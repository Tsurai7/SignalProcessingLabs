import tkinter as tk
from PIL import Image, ImageTk
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from tkinter import ttk

from function import sum_signals, manual_uniform_noise, d2_transform,inverse_d2,manual_d2_decomposition

from fun_wave import save_wave, read_wave

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../res")))

def update_output(text):
    result_label.config(text=text)

def draw_discretizatio_f():
    a1 = float(entries["a1"].get())
    a2 = float(entries["a2"].get())
    b1 = int(entries["b1"].get())
    b2 = int(entries["b2"].get())
    v1 = int(entries["v1"].get())
    v2 = int(entries["v2"].get())
    phi0 = float(entries["phi0"].get())
    D = float(entries["D"].get())
    d = int(entries["d"].get())
    N = int(D / (2 * math.pi)) * d
    t = np.linspace(0, D, N)
    f = sum_signals(a1, b1, v1, a2, b2, v2, t, phi0)
    plt.figure(figsize=(10, 6))
    plt.plot(t, f,
             label=f"f(t) (N={N} точек)",
             color="black",
             linewidth=1,
             marker='o',
             markersize=2,
             markeredgecolor='black',
             markerfacecolor='black')

    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title(f"Дискретизация сигнала (D={D} с, d={d} точек/2π)")
    plt.legend()
    plt.grid(True)
    plt.margins(x=0.05, y=0.1)
    plt.show()
    save_wave("res\\audio\\signal_f.wav",f)

def draw_noise_f():
    try:
        f, sample_rate = read_wave("res\\audio\\signal_f.wav")
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return
    noise = manual_uniform_noise(len(f))
    f += noise
    #f = np.clip(f + noise, -1.0, 1.0)
    D = float(entries["D"].get())
    d = int(entries["d"].get())
    N = int(D / (2 * math.pi)) * d
    t = np.linspace(0, D, N)
    plt.figure(figsize=(10, 6))
    plt.plot(t, f,
             label=f"f(t) (N={N} точек)",
             color="black",
             linewidth=1,
             marker='o',
             markersize=2,
             markeredgecolor='black',
             markerfacecolor='black')
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title(f"noise signal")
    plt.legend()
    plt.grid(True)
    plt.margins(x=0.05, y=0.1)
    plt.show()
    save_wave("res\\audio\\noise_signal_f.wav",f)


def dvp():
    f, sample_rate = read_wave("res\\audio\\noise_signal_f.wav")
    if f is None:
        print("Ошибка: не удалось загрузить файл")
        if 'output' in entries:
            entries['output'].delete(1.0, tk.END)
            entries['output'].insert(tk.END, "Ошибка загрузки файла")
        return

    f = f / np.max(np.abs(f))

    D = float(entries["D"].get())
    d = int(entries["d"].get())
    N = int((D / (2 * math.pi)) * d)

    adjusted_N = min((N // 4) * 4, len(f))
    if adjusted_N < 4:
        error_msg = "Ошибка: слишком короткий сигнал (минимум 4 точки)"
        print(error_msg)
        if 'output' in entries:
            entries['output'].delete(1.0, tk.END)
            entries['output'].insert(tk.END, error_msg)
        return

    t = np.linspace(0, D, adjusted_N)
    signal = f[:adjusted_N]

    print(f"Длина исходного сигнала: {len(signal)} точек") # to output

    coeffs = manual_d2_decomposition(signal, level=2)
    if len(coeffs) != 3:
        error_msg = f"Ошибка: ожидалось 3 коэффициента, получено {len(coeffs)}"
        print(error_msg)
        if 'output' in entries:
            entries['output'].delete(1.0, tk.END)
            entries['output'].insert(tk.END, error_msg)
        return

    cA2, cD2, cD1 = coeffs
    print(f"Размеры коэффициентов: {[len(c) for c in coeffs]}") # to output

    def exact_reconstruct(a, d, target_len):
        """Восстановление с жестким контролем длины"""
        min_len = min(len(a), len(d))
        reconstructed = inverse_d2(a[:min_len], d[:min_len])
        return reconstructed[:target_len]

    # Аппроксимация уровня 2
    level2 = exact_reconstruct(cA2, np.zeros_like(cA2), len(cA2)*2)
    A2 = exact_reconstruct(level2, np.zeros_like(level2), adjusted_N)

    # Аппроксимация уровня 1
    A1 = exact_reconstruct(cA2, np.zeros_like(cA2), adjusted_N)

    # Детали уровня 2
    D2 = exact_reconstruct(np.zeros_like(cD2), cD2, adjusted_N)

    # Детали уровня 1
    D1 = exact_reconstruct(np.zeros_like(cD1), cD1, adjusted_N)

    print(f"Проверка размеров перед визуализацией:") # to output
    print(f"t: {len(t)}, signal: {len(signal)}")
    print(f"A2: {len(A2)}, A1: {len(A1)}, D2: {len(D2)}, D1: {len(D1)}")

    plt.figure(figsize=(12, 10))
    if len(t) != len(signal):
        signal = signal[:len(t)]
    plt.subplot(5, 1, 1)
    plt.plot(t, signal, 'b')
    plt.title(f'Исходный сигнал (N={adjusted_N})')
    plt.grid(True)

    components = [
        (A2[:len(t)], 'r', 'Аппроксимирующая (Уровень 1)'),
        (A1[:len(t)], 'c', 'Аппроксимация (Уровень 2)'),
        (D2[:len(t)], 'g', 'Детали (Уровень 2)'),
        (D1[:len(t)], 'm', 'Детали (Уровень 1)')
    ]
    for i, (comp, color, title) in enumerate(components, 2):
        plt.subplot(5, 1, i)
        plt.plot(t[:len(comp)], comp, color)
        plt.title(title)
        plt.grid(True)
        plt.ylim(-1.2, 1.2)
    plt.tight_layout()
    plt.show()


def dvp():
    f, sample_rate = read_wave("res\\audio\\noise_signal_f.wav")
    if f is None:
        print("Ошибка: не удалось загрузить файл")
        if 'output' in entries:
            entries['output'].delete(1.0, tk.END)
            entries['output'].insert(tk.END, "Ошибка загрузки файла")
        return

    f = f / np.max(np.abs(f))

    D = float(entries["D"].get())
    d = int(entries["d"].get())
    N = int((D / (2 * math.pi)) * d)

    adjusted_N = min((N // 8) * 8, len(f))  # Увеличиваем кратность для 4 уровней
    if adjusted_N < 8:
        error_msg = "Ошибка: слишком короткий сигнал (минимум 8 точек)"
        print(error_msg)
        if 'output' in entries:
            entries['output'].delete(1.0, tk.END)
            entries['output'].insert(tk.END, error_msg)
        return

    t = np.linspace(0, D, adjusted_N)
    signal = f[:adjusted_N]

    print(f"Длина исходного сигнала: {len(signal)} точек")

    # Изменяем на 4 уровня декомпозиции
    coeffs = manual_d2_decomposition(signal, level=4)
    if len(coeffs) != 5:
        error_msg = f"Ошибка: ожидалось 5 коэффициентов, получено {len(coeffs)}"
        print(error_msg)
        if 'output' in entries:
            entries['output'].delete(1.0, tk.END)
            entries['output'].insert(tk.END, error_msg)
        return

    cA4, cD4, cD3, cD2, cD1 = coeffs
    print(f"Размеры коэффициентов: {[len(c) for c in coeffs]}")

    def exact_reconstruct(a, d, target_len):
        """Восстановление с жестким контролем длины"""
        min_len = min(len(a), len(d))
        reconstructed = inverse_d2(a[:min_len], d[:min_len])
        return reconstructed[:target_len]

    # Функция для восстановления компонент на нужном уровне
    def reconstruct_component(approx_coeffs, detail_coeffs, target_level, target_len):
        current = approx_coeffs
        for level in range(target_level, 0, -1):
            if level == target_level:
                current = exact_reconstruct(np.zeros_like(detail_coeffs[target_level - 1]),
                                            detail_coeffs[target_level - 1],
                                            len(current) * 2)
            else:
                current = exact_reconstruct(current, np.zeros_like(current), len(current) * 2)
        return current[:target_len]

    # Восстановление всех компонент
    components = {}

    # Аппроксимации
    components['A4'] = exact_reconstruct(cA4, np.zeros_like(cA4), adjusted_N)
    components['A3'] = exact_reconstruct(
        exact_reconstruct(cA4, np.zeros_like(cA4), len(cA4) * 2),
        np.zeros_like(cA4),
        adjusted_N
    )
    components['A2'] = exact_reconstruct(
        exact_reconstruct(
            exact_reconstruct(cA4, np.zeros_like(cA4), len(cA4) * 2),
            np.zeros_like(cA4),
            len(cA4) * 4
        ),
        np.zeros_like(cA4),
        adjusted_N
    )
    components['A1'] = exact_reconstruct(cA4, np.zeros_like(cA4), adjusted_N)

    # Детали
    components['D4'] = reconstruct_component(cA4, [cD4, cD3, cD2, cD1], 4, adjusted_N)
    components['D3'] = reconstruct_component(cA4, [cD4, cD3, cD2, cD1], 3, adjusted_N)
    components['D2'] = reconstruct_component(cA4, [cD4, cD3, cD2, cD1], 2, adjusted_N)
    components['D1'] = reconstruct_component(cA4, [cD4, cD3, cD2, cD1], 1, adjusted_N)

    print(f"Проверка размеров перед визуализацией:")
    print(f"t: {len(t)}, signal: {len(signal)}")
    for name, comp in components.items():
        print(f"{name}: {len(comp)}")

    plt.figure(figsize=(12, 16))

    # Исходный сигнал
    plt.subplot(9, 1, 1)
    plt.plot(t, signal[:len(t)], 'b')
    plt.title(f'Исходный сигнал (N={adjusted_N})')
    plt.grid(True)

    # Компоненты для отображения (добавляем уровни 3 и 4)
    comp_list = [
        (components['A4'], 'r', 'Аппроксимирующая (Уровень 4)'),
        (components['A3'], 'c', 'Аппроксимация (Уровень 3)'),
        (components['A2'], 'm', 'Аппроксимация (Уровень 2)'),
        (components['A1'], 'y', 'Аппроксимация (Уровень 1)'),
        (components['D4'], 'g', 'Детали (Уровень 4)'),
        (components['D3'], 'b', 'Детали (Уровень 3)'),
        (components['D2'], 'k', 'Детали (Уровень 2)'),
        (components['D1'], 'orange', 'Детали (Уровень 1)')
    ]

    for i, (comp, color, title) in enumerate(comp_list, 2):
        plt.subplot(9, 1, i)
        plt.plot(t[:len(comp)], comp, color)
        plt.title(title)
        plt.grid(True)
        plt.ylim(-1.2, 1.2)

    plt.tight_layout()
    plt.show()

def reconstruct_and_compare():
    try:
        f, sample_rate = read_wave("res\\audio\\noise_signal_f.wav")
        f = f / np.max(np.abs(f))

        D = float(entries["D"].get())
        N = min(len(f), int(D * sample_rate))
        adjusted_N = (N // 4) * 4
        if adjusted_N < 4:
            raise ValueError("Слишком короткий сигнал (минимум 4 точки)") # to output

        t = np.linspace(0, D, adjusted_N)
        signal = f[:adjusted_N]

        print(f"Длина исходного сигнала: {len(signal)} точек") # to output

        coeffs = manual_d2_decomposition(signal, level=2)
        print(f"Размеры коэффициентов: {[len(c) for c in coeffs]}") # to output

        if len(coeffs) != 3:
            raise ValueError(f"Ожидалось 3 коэффициента, получено {len(coeffs)}") # to output

        cD1, cD2, cA2 = coeffs

        def correct_reconstruction(cA2, cD2, cD1, target_len):

            level2 = inverse_d2(cA2, cD2)

            full_signal = inverse_d2(level2, cD1)

            return full_signal[:target_len]

        reconstructed = correct_reconstruction(cA2, cD2, cD1, len(signal))

        print(f"Длина восстановленного сигнала: {len(reconstructed)} точек") # to output

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(t, signal, 'b', linewidth=1)
        plt.title(f'Оригинальный зашумленный сигнал\nДлина: {adjusted_N} точек')
        plt.xlabel('Время, с')
        plt.ylabel('Амплитуда')
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        plt.plot(t, reconstructed, 'r', linewidth=1)
        plt.title('Восстановленный сигнал после вейвлет-преобразования')
        plt.xlabel('Время, с')
        plt.ylabel('Амплитуда')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e: # to output
        error_msg = f"Ошибка: {str(e)}"
        print(error_msg)
        if 'output' in entries:
            entries['output'].delete(1.0, tk.END)
            entries['output'].insert(tk.END, error_msg)




def process_and_compare():
    try:
        noisy_signal, sample_rate = read_wave("res\\audio\\noise_signal_f.wav")
        noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))

        clean_signal, _ = read_wave("res\\audio\\signal_f.wav")
        clean_signal = clean_signal[:len(noisy_signal)]
        clean_signal = clean_signal / np.max(np.abs(clean_signal))
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    D = float(entries["D"].get())
    N = min(len(noisy_signal), int(D * sample_rate))
    t = np.linspace(0, D, N)
    noisy_signal = noisy_signal[:N]
    clean_signal = clean_signal[:N]

    def modified_haar_decomposition(signal, level=2):
        coeffs = []
        current = signal.copy()

        for _ in range(level):
            if len(current) < 2:
                break
            approx, detail = d2_transform(current)
            detail_padded = np.zeros(len(current))
            detail_padded[:len(detail)] = detail
            coeffs.append(detail_padded)
            current = approx

        approx_padded = np.zeros(len(signal))
        approx_padded[:len(current)] = current
        coeffs.append(approx_padded)

        return coeffs

    coeffs = modified_haar_decomposition(noisy_signal, level=2)
    if len(coeffs) != 3:
        print("Неверное количество коэффициентов")
        return

    cA4_full, cD4_full, cD3_full = coeffs

    cD3_full *= np.zeros_like(cD3_full)
    cD4_full *= 2

    def full_length_reconstruction(cA, cD_list):
        """Восстановление с сохранением исходной длины"""
        reconstructed = cA.copy()
        for cD in reversed(cD_list):
            level_len = len(reconstructed)
            if level_len % 2 != 0:
                reconstructed = np.append(reconstructed, 0)
                level_len += 1

            pairs = level_len // 2
            restored = np.zeros(level_len)

            for i in range(pairs):
                a = reconstructed[i]
                d = cD[i] if i < len(cD) else 0
                restored[2*i] = (a + d) / np.sqrt(2)
                restored[2*i+1] = (a - d) / np.sqrt(2)

            reconstructed = restored

        return reconstructed[:len(cA)]

    processed_signal = full_length_reconstruction(cA4_full, [cD4_full, cD3_full])

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, clean_signal, 'g')
    plt.title(f'Оригинальный сигнал (N={len(t)})')
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_signal, 'b', alpha=0.7)
    plt.title('Зашумленный сигнал')
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(t, processed_signal, 'r')
    plt.title('После обработки (L1×2, L2=0)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def application():
    try:
        noisy_signal, sample_rate = read_wave("res\\audio\\noise_signal_f.wav")
        noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))

        clean_signal, _ = read_wave("res\\audio\\signal_f.wav")
        clean_signal = clean_signal[:len(noisy_signal)]
        clean_signal = clean_signal / np.max(np.abs(clean_signal))
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    D = float(entries["D"].get())
    N = min(len(noisy_signal), int(D * sample_rate))
    t = np.linspace(0, D, N)
    noisy_signal = noisy_signal[:N]
    clean_signal = clean_signal[:N]

    def modified_haar_decomposition(signal, level=2):
        coeffs = []
        current = signal.copy()

        for _ in range(level):
            if len(current) < 2:
                break
            approx, detail = d2_transform(current)
            detail_padded = np.zeros(len(current))
            detail_padded[:len(detail)] = detail
            coeffs.append(detail_padded)
            current = approx

        approx_padded = np.zeros(len(signal))
        approx_padded[:len(current)] = current
        coeffs.append(approx_padded)

        return coeffs

    coeffs = modified_haar_decomposition(noisy_signal, level=2)
    if len(coeffs) != 3:
        print("Неверное количество коэффициентов")
        return

    cA2_full, cD2_full, cD1_full = coeffs

    # Получаем параметры обработки для каждого уровня
    l1_operation = level_operations["L1_operation"].get()
    l1_value = float(level_operations["L1_value"].get())
    l2_operation = level_operations["L2_operation"].get()
    l2_value = float(level_operations["L2_value"].get())

    # Применяем операции к уровням
    if l1_operation == "*":
        cD1_full *= l1_value
    elif l1_operation == "/":
        cD1_full /= l1_value
    elif l1_operation == "+":
        cD1_full += l1_value
    elif l1_operation == "-":
        cD1_full -= l1_value

    if l2_operation == "*":
        cD2_full *= l2_value
    elif l2_operation == "/":
        cD2_full /= l2_value
    elif l2_operation == "+":
        cD2_full += l2_value
    elif l2_operation == "-":
        cD2_full -= l2_value
    elif l2_operation == "0":  # Обнуление
        cD2_full = np.zeros_like(cD2_full)

    def full_length_reconstruction(cA, cD_list):
        """Восстановление с сохранением исходной длины"""
        reconstructed = cA.copy()
        for cD in reversed(cD_list):
            level_len = len(reconstructed)
            if level_len % 2 != 0:
                reconstructed = np.append(reconstructed, 0)
                level_len += 1

            pairs = level_len // 2
            restored = np.zeros(level_len)

            for i in range(pairs):
                a = reconstructed[i]
                d = cD[i] if i < len(cD) else 0
                restored[2*i] = (a + d) / np.sqrt(2)
                restored[2*i+1] = (a - d) / np.sqrt(2)

            reconstructed = restored

        return reconstructed[:len(cA)]

    processed_signal = full_length_reconstruction(cA2_full, [cD2_full, cD1_full])

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, clean_signal, 'g')
    plt.title(f'Оригинальный сигнал (N={len(t)})')
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_signal, 'b', alpha=0.7)
    plt.title('Зашумленный сигнал')
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(t, processed_signal, 'r')
    plt.title(f'После обработки (L1:{l1_operation}{l1_value}, L2:{l2_operation}{l2_value})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from tkinter import ttk

# Создаем главное окно
root = tk.Tk()
root.title("App")
root.configure(bg="#f0f0f0")

# Устанавливаем окно на полный экран
root.geometry("1200x900")

# Создаем стиль для кнопок
button_style = {
    "bg": "#4CAF50",
    "fg": "white",
    "activebackground": "#45a049",
    "font": ("Arial", 10),
    "width": 25,  # Фиксированная ширина для всех кнопок
    "anchor": "center",  # Выравнивание текста по центру
    "bd": 0,  # Без границы
    "padx": 10,
    "pady": 5
}

# Создаем основной контейнер
main_container = tk.Frame(root, bg="#f0f0f0")
main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Левая панель (управление)
control_frame = tk.Frame(main_container, bg="#f0f0f0", width=300)
control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

# Правая панель (результаты)
result_frame = tk.Frame(main_container, bg="#ffffff", relief=tk.GROOVE, borderwidth=2)
result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Заголовок с изображением
image = Image.open("res\\img\\image.png")
image = image.resize((200, 100), Image.Resampling.LANCZOS)
photo = ImageTk.PhotoImage(image)

image_label = tk.Label(control_frame, image=photo, bg="#f0f0f0")
image_label.image = photo
image_label.pack(pady=(0, 20))

# Поля ввода параметров
entries = {}
for key, value in {"a1": 1.0, "a2": 2.0, "b1": 5, "b2": 1, "v1": 1, "v2": 2, "phi0": -1.4, "D": 84, "d": 128}.items():
    frame = tk.Frame(control_frame, bg="#f0f0f0")
    frame.pack(fill=tk.X, pady=3)
    tk.Label(frame, text=f"{key}:", bg="#f0f0f0").pack(side=tk.LEFT)
    entries[key] = tk.Entry(frame)
    entries[key].pack(side=tk.RIGHT, expand=True, fill=tk.X)
    entries[key].insert(tk.END, str(value))

# Поля для операций с уровнями
level_operations = {}

# Поля для L3
tk.Label(control_frame, text="L3 Операция:", bg="#f0f0f0").pack()
level_operations["L3_operation"] = ttk.Combobox(control_frame, values=["*", "/", "+", "-", "0"])
level_operations["L3_operation"].pack(fill=tk.X, pady=3)
level_operations["L3_operation"].set("*")

tk.Label(control_frame, text="L3 Значение:", bg="#f0f0f0").pack()
level_operations["L3_value"] = tk.Entry(control_frame)
level_operations["L3_value"].pack(fill=tk.X, pady=3)
level_operations["L3_value"].insert(tk.END, "2.0")

# Поля для L4
tk.Label(control_frame, text="L4 Операция:", bg="#f0f0f0").pack()
level_operations["L4_operation"] = ttk.Combobox(control_frame, values=["*", "/", "+", "-", "0"])
level_operations["L4_operation"].pack(fill=tk.X, pady=3)
level_operations["L4_operation"].set("0")

tk.Label(control_frame, text="L4 Значение:", bg="#f0f0f0").pack()
level_operations["L4_value"] = tk.Entry(control_frame)
level_operations["L4_value"].pack(fill=tk.X, pady=3)
level_operations["L4_value"].insert(tk.END, "1.0")




# Список кнопок с командами
buttons = [
    ("Дискретизация сигнала f(t)", draw_discretizatio_f),
    ("Зашумленный сигнал f'(t)", draw_noise_f),
    ("ДВП", dvp),
    ("ОДВП", reconstruct_and_compare),
    ("ОДВП 2", process_and_compare),
    ("Приложение", application)
]

# Создаем кнопки одинаковой ширины
for text, command in buttons:
    btn = tk.Button(control_frame, text=text, command=command, **button_style)
    btn.pack(pady=5, fill=tk.X)  # fill=tk.X делает кнопки одинаковой ширины внутри контейнера

# Метка для вывода результатов
result_label = tk.Label(result_frame,
                      text="Ожидание команды...",
                      bg="#ffffff",
                      fg="#333",
                      font=("Arial", 12),
                      justify="left",
                      anchor="nw",
                      wraplength=800,
                      padx=10,
                      pady=10)
result_label.pack(fill=tk.BOTH, expand=True)

# Привязываем клавишу Escape для выхода из полноэкранного режима
root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))

root.mainloop()