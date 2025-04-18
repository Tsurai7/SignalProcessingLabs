import math
import numpy as np
import random

def integral(f, a, b, N, *params):
    """Расчёт интеграла."""
    dx = (b - a) / N
    result = 0.5 * f(a, *params) + 0.5 * f(b, *params)

    for i in range(1, N):
        x_i = a + i * dx
        result += f(x_i, *params)

    integral_value = result * dx
    return integral_value

def x_fun(a1, b1, v1, t, phi0):
    return a1 * np.power(np.sin(v1 * t + phi0), b1)

def y_fun(a2, b2, v2, t, phi0):
    return a2 * np.power(np.cos(v2 * t + phi0), b2)

def product_x_y(t, a1, b1, v1, a2, b2, v2, phi0):
    return x_fun(a1, b1, v1, t, phi0) * y_fun(a2, b2, v2, t, phi0)

def sum_signals(a1, b1, v1, a2, b2, v2, t, phi0):
    return x_fun(a1, b1, v1, t, phi0) + y_fun(a2, b2, v2, t, phi0)

def manual_uniform_noise(size, low=-1.0, high=1.0):
    random.seed()  
    return np.array([random.uniform(low, high) for _ in range(size)])


def d2_transform(signal):
    """Одномерное прямое преобразование Добеши D2 (аналог Хаара)"""
    n = len(signal)
    if n % 2 != 0:
        signal = np.append(signal, 0)  # дополнение нулем при нечетной длине

    # Фильтры Добеши D2
    h = [1 / np.sqrt(2), 1 / np.sqrt(2)]  # низкочастотный фильтр
    g = [1 / np.sqrt(2), -1 / np.sqrt(2)]  # высокочастотный фильтр

    approx = np.convolve(signal, h, mode='valid')[::2]  # приближающие коэффициенты
    detail = np.convolve(signal, g, mode='valid')[::2]  # детализирующие коэффициенты

    return approx, detail


def inverse_d2(approx, detail):
    """Обратное преобразование Добеши D2"""
    n = len(approx)
    signal = np.zeros(2 * n)

    # Обратные фильтры
    h_inv = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    g_inv = [1 / np.sqrt(2), -1 / np.sqrt(2)]

    # Восстановление сигнала
    for i in range(n):
        signal[2 * i] += approx[i] * h_inv[0] + detail[i] * g_inv[0]
        if 2 * i + 1 < len(signal):
            signal[2 * i + 1] += approx[i] * h_inv[1] + detail[i] * g_inv[1]

    return signal


def manual_d2_decomposition(signal, level=2):
    """Многоуровневое разложение Добеши D2"""
    coeffs = []
    current_signal = signal.copy()

    for _ in range(level):
        if len(current_signal) < 2:
            break
        approx, detail = d2_transform(current_signal)
        coeffs.append(detail)
        current_signal = approx

    coeffs.append(current_signal)
    return coeffs

def reconstruct_specific_level(coeffs, target_level):
    """Восстановление конкретной компоненты"""
    A = coeffs[0]
    for i in range(1, len(coeffs)-target_level):
        A = inverse_d2(A, coeffs[i])
    return A
