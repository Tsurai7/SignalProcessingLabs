import matplotlib.pyplot as plt
import numpy as np
import math
from shared.my_math import (my_cos, my_sin, integral, power_fast,
                            DFT, DIF_FFT, IDFT, IFFT, my_log2, my_sqrt)
import time
from shared.wav import save_wave


a1, b1, v1 = 1, 5, 1
a2, b2, v2 = 2, 1, 2
phi0 = -1.4
N = 128
D = 84

epsilon = 1e-15

attenuation_factor = 0.5
cutoff_frequency = 1


def U1(a1, b1, v1, t, phi0):
    return a1 * power_fast(my_sin(v1 * t + phi0), b1)


def U2(a2, b2, v2, t, phi0):
    return a2 * power_fast(my_cos(v2 * t + phi0), b2)


def product_U1_U2(t, a1, b1, v1, a2, b2, v2, phi0):
    return U1(a1, b1, v1, t, phi0) * U2(a2, b2, v2, t, phi0)


def f(t, a1, b1, v1, a2, b2, v2, phi0):
    return U1(a1, b1, v1, t, phi0) + U2(a2, b2, v2, t, phi0)


def modify_spectrum(Y, attenuation_factor, cutoff_frequency, sample_rate):
    """Modifies the amplitude in the frequency domain and zeroes out low frequencies."""
    N = len(Y)
    freq_step = sample_rate / N
    modified_Y = np.array(Y, dtype=complex)

    for i in range(N):
        freq = i * freq_step

        if freq < cutoff_frequency:
            modified_Y[i] = 0
        else:
            modified_Y[i] *= attenuation_factor

    return modified_Y


def calculate_fft_efficiency(N):
    """Calculates efficiency."""
    operations_fft = N * my_log2(N)
    operations_dft = N ** 2
    speedup = operations_dft / operations_fft

    x = np.random.random(N)
    start_time = time.time()
    DFT(x)
    dft_time = time.time() - start_time

    start_time = time.time()
    DIF_FFT(x)
    fft_time = time.time() - start_time

    print(f"Number of points: {N}")
    print(f"DFT operations: {operations_dft:.0f}")
    print(f"DIF FFT operations: {operations_fft:.0f}")
    print(f"DIF FFT is {speedup:.2f} times faster than DFT")
    print(f"DFT time: {dft_time:.6f} sec")
    print(f"DIF FFT time: {fft_time:.6f} sec")
    print(f"DIF FFT is {dft_time - fft_time:.6f} sec faster than DFT")


result = integral(product_U1_U2, phi0, phi0 + 2 * math.pi, N, a1, b1,
                  v1, a2, b2, v2, phi0)
print("Integration result:", result)

if abs(result) < epsilon:
    print(f"The integration result is close to 0 with accuracy {epsilon}, " +
          "therefore the functions are orthogonal")
else:
    print(f"The integration result is not equal to 0 with accuracy {epsilon} " +
          "therefore the functions are not orthogonal")

scalar_product_u1_u2 = integral(product_U1_U2, phi0, phi0 + 2 * math.pi, N, a1, b1, v1, a2, b2, v2, phi0)
scalar_product_u2_u1 = integral(product_U1_U2, phi0, phi0 + 2 * math.pi, N, a2, b2, v2, a1, b1, v1, phi0)

print(f"Scalar product U1 * U2: {scalar_product_u1_u2}")
print(f"Scalar product U2 * U1: {scalar_product_u2_u1}")

t = np.linspace(0, D, N)
y = f(t, a1, b1, v1, a2, b2, v2, phi0)

save_wave("y.wav", y.real, round(N/D))
print("Audio file 'y.wav' saved.")

# Discrete Fourier Transform (DFT)
start_time = time.time()
y_dft = DFT(y)
dft_time = time.time() - start_time

# Fast Fourier Transform (DIF FFT)
start_time = time.time()
y_dif_fft = DIF_FFT(y)
fft_time = time.time() - start_time

# Inverse DFT
start_time = time.time()
y_reconstructed_dft = IDFT(y_dft).real
idft_time = time.time() - start_time

# Inverse DIF FFT
start_time = time.time()
y_reconstructed_dif_fft = IFFT(y_dif_fft).real
ifft_time = time.time() - start_time

calculate_fft_efficiency(N)


def calculate_frequencies(N, D):
    sample_rate = N / D
    freq = [(i * sample_rate) / N for i in range(N // 2)]
    return freq


freq = calculate_frequencies(N, D)


def calculate_amplitude(Y):
    amplitude = [my_sqrt(y.real**2 + y.imag**2) for y in Y]
    return amplitude


amplitude_dft = calculate_amplitude(y_dft)
amplitude_dif_fft = calculate_amplitude(y_dif_fft)

y_dif_fft_modified = modify_spectrum(y_dif_fft, attenuation_factor,
                                     cutoff_frequency, N / D)
amplitude_dif_fft_modified = calculate_amplitude(y_dif_fft_modified)

print(f"DFT time: {dft_time:.5f} sec")
print(f"DIF FFT time: {fft_time:.5f} sec")
print(f"DIF FFT is {dft_time - fft_time:.5f} sec faster than DFT")
print(f"Inverse DFT time: {idft_time:.5f} sec")
print(f"Inverse DIF FFT time: {ifft_time:.5f} sec")
print(f"Inverse DIF FFT is {idft_time - ifft_time:.5f} sec faster than inverse DFT")

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(t, y, label="Original f(t)")
plt.title("Graph of f(t)")
plt.legend()
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(freq[:N//2], amplitude_dif_fft_modified[:N//2],
         label="Modified Amplitude Spectrum (DIF)", color='purple')
plt.title("Amplitude Spectrum after modifications (DIF)")
plt.legend()
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(freq[:N//2], amplitude_dft[:N//2], label="Amplitude Spectrum (DFT)", color='b')
plt.title("Amplitude Spectrum (DFT)")
plt.legend()
plt.grid()

plt.subplot(3, 2, 4)
plt.plot(freq[:N//2], amplitude_dif_fft[:N//2], label="Amplitude Spectrum (DIF FFT)", color='r')
plt.title("Amplitude Spectrum (DIF FFT)")
plt.legend()
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(t, y_reconstructed_dft,
         label="Reconstructed Signal (Inverse DFT)", color='g')
plt.title("Reconstructed Signal (Inverse DFT)")
plt.legend()
plt.grid()

plt.subplot(3, 2, 6)
plt.plot(t, y_reconstructed_dif_fft,
         label="Reconstructed Signal (Inverse DIF FFT)", color='m')
plt.title("Reconstructed Signal (Inverse DIF FFT)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

save_wave("y_reconstructed_dft.wav", y_reconstructed_dft.real, round(N/D))
print("Audio file 'y_reconstructed_dft.wav' saved.")

save_wave("y_reconstructed_dif_fft.wav", y_reconstructed_dif_fft.real, round(N/D))
print("Audio file 'y_reconstructed_dif_fft.wav' saved.")