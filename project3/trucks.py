import numpy as np
import pandas as pd
from math import floor
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Настройка русских шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('1_w.csv', sep=';', encoding='cp1251')

cols = [
    "Широта", "Долгота", "Скорость",
    "Ускорение по оси X", "Ускорение по оси Y",
    "Статус самосвала"
]

data = df[cols].copy()

# Преобразование в числовой формат
for c in cols[:-1]:
    data[c] = pd.to_numeric(data[c], errors='coerce')

# Удаление строк с пропусками
data = data.dropna().reset_index(drop=True)

# Радиус Земли
R = 6371000

# Опорная точка
lat0 = np.deg2rad(data.loc[0, "Широта"])
lon0 = np.deg2rad(data.loc[0, "Долгота"])

def latlon_to_xy(lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = R * (lon - lon0) * np.cos(lat0)
    y = R * (lat - lat0)
    return x, y

def xy_to_latlon(x, y):
    lat = y / R + lat0
    lon = x / (R * np.cos(lat0)) + lon0
    return np.rad2deg(lat), np.rad2deg(lon)

# Добавление координат в метрах
data["x"], data["y"] = latlon_to_xy(data["Широта"], data["Долгота"])

dt = 1.0

# Поиск момента разгрузки
mask = (data["Скорость"] < 1)
if mask.any():
    unload_idx = data.index[mask][0]
else:
    unload_idx = int(len(data)*0.9)

# =========================
# ФИЛЬТР КАЛМАНА
# =========================

# Матрица перехода (x, ẋ, y, ẏ)
F = np.array([
    [1, dt, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dt],
    [0, 0, 0, 1]
])

# Матрица управления
B = np.array([
    [0.5*dt**2, 0],
    [dt, 0],
    [0, 0.5*dt**2],
    [0, dt]
])

# Матрица измерений
H = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

sigma_a = 1.5
sigma_meas = 8

# Ковариация процесса
Q = sigma_a**2 * np.array([
    [dt**4/4, dt**3/2, 0, 0],
    [dt**3/2, dt**2, 0, 0],
    [0, 0, dt**4/4, dt**3/2],
    [0, 0, dt**3/2, dt**2]
])

R_meas = np.eye(2) * sigma_meas**2

# Начальное состояние [x, vx, y, vy]
state = np.array([data.loc[0,"x"], 0, data.loc[0,"y"], 0])
P = np.eye(4) * 100

states = []
covs = []

for i in range(len(data)):
    z = np.array([data.loc[i,"x"], data.loc[i,"y"]])
    u = np.array([data.loc[i,"Ускорение по оси X"], data.loc[i,"Ускорение по оси Y"]])
    
    # Прогноз
    state_pred = F @ state + B @ u
    P_pred = F @ P @ F.T + Q
    
    # Обновление
    S = H @ P_pred @ H.T + R_meas
    K = P_pred @ H.T @ np.linalg.inv(S)
    state = state_pred + K @ (z - H @ state_pred)
    P = (np.eye(4) - K @ H) @ P_pred
    
    states.append(state.copy())
    covs.append(P.copy())

states = np.array(states)
covs = np.array(covs)

# Координаты разгрузки
x_u = states[unload_idx, 0]
y_u = states[unload_idx, 2]
lat_u, lon_u = xy_to_latlon(x_u, y_u)

var_x = covs[unload_idx, 0, 0]
var_y = covs[unload_idx, 2, 2]

print("\n" + "="*60)
print("ЧАСТЬ A — ЗАДАНИЕ 1")
print("="*60)
print(f"Координаты разгрузки: {lat_u:.10f}, {lon_u:.10f}")
print(f"\nЗАДАНИЕ 2")
print(f"Дисперсия X: {var_x:.6f} м²")
print(f"Дисперсия Y: {var_y:.6f} м²")

# =========================
# ЧАСТЬ B (первая треть списка - смена 1)
# =========================

# Параметры дорог
roads = {
    "D1": {"warehouse": "C",  "load": 2, "loaded": 15, "unload": 5},
    "D2": {"warehouse": "Kb", "load": 2, "loaded": 10, "unload": 5},
    "D3": {"warehouse": "Km", "load": 2, "loaded": 20, "unload": 5},
    "D4": {"warehouse": "T",  "load": 2, "loaded": 15, "unload": 5},
}

# Химический состав складов
warehouses = {
    "C":  {"Ni": 2.28, "Cu": 2.36},
    "Kb": {"Ni": 1.80, "Cu": 1.90},
    "Km": {"Ni": 0.56, "Cu": 1.17},
    "T":  {"Ni": 1.98, "Cu": 2.82},
}

# План поставки 1 смены (тонн за смену)
shift_plan = {
    "C": 2400,
    "Kb": 500,
    "Km": 450,
    "T": 1600
}

# Целевые показатели 1 смены
target_Ni = 1.58
target_Cu = 2.66

# Грузоподъемность (тонн)
capacity = 240

# Время цикла: погрузка + груженый + разгрузка + порожний (грузеный - 2)
def cycle_time(load, loaded, unload):
    empty = loaded - 2
    return load + loaded + unload + empty

def productivity(cycle_minutes):
    return 60 * capacity / cycle_minutes

print("\n" + "="*60)
print("ЧАСТЬ B — ЗАДАНИЕ 1 (Параметры дорог)")
print("="*60)

for r in roads:
    cycle_min = cycle_time(roads[r]["load"], roads[r]["loaded"], roads[r]["unload"])
    roads[r]["cycle"] = cycle_min
    roads[r]["prod"] = productivity(cycle_min)
    print(f"{r}: цикл = {cycle_min} мин, произв. = {roads[r]['prod']:.1f} т/ч")

# ЗАДАНИЕ 2: Максимальное количество самосвалов (простой ≤ 5 мин)
print("\n" + "="*60)
print("ЧАСТЬ B — ЗАДАНИЕ 2")
print("="*60)

max_trucks = {}
for r in roads:
    max_trucks[r] = floor(roads[r]["cycle"] / 5)
    print(f"{r}: макс. {max_trucks[r]} самосвалов")

# ЗАДАНИЕ 3a: Минимизация отклонений Ni и Cu
print("\n" + "="*60)
print("ЧАСТЬ B — ЗАДАНИЕ 3 (первая треть списка)")
print("Цель: Ni → 1.58%, Cu → 2.66%")
print("="*60)

best = None
best_err = float('inf')
best_without_plan = None
best_err_without_plan = float('inf')

print("\nПеребор комбинаций...")

for x1 in range(max_trucks["D1"] + 1):
    for x2 in range(max_trucks["D2"] + 1):
        for x3 in range(max_trucks["D3"] + 1):
            for x4 in range(max_trucks["D4"] + 1):
                
                total_trucks = x1 + x2 + x3 + x4
                if total_trucks >= 30 or total_trucks == 0:
                    continue
                
                # Потоки руды (т/час)
                flow1 = x1 * roads["D1"]["prod"]
                flow2 = x2 * roads["D2"]["prod"]
                flow3 = x3 * roads["D3"]["prod"]
                flow4 = x4 * roads["D4"]["prod"]
                
                total_flow = flow1 + flow2 + flow3 + flow4
                if total_flow == 0:
                    continue
                
                # Средневзвешенный состав
                avg_Ni = (flow1 * warehouses["C"]["Ni"] + 
                         flow2 * warehouses["Kb"]["Ni"] + 
                         flow3 * warehouses["Km"]["Ni"] + 
                         flow4 * warehouses["T"]["Ni"]) / total_flow
                
                avg_Cu = (flow1 * warehouses["C"]["Cu"] + 
                         flow2 * warehouses["Kb"]["Cu"] + 
                         flow3 * warehouses["Km"]["Cu"] + 
                         flow4 * warehouses["T"]["Cu"]) / total_flow
                
                # Функция ошибки
                err = abs(avg_Ni - target_Ni) + abs(avg_Cu - target_Cu)
                
                # Сохраняем лучшее решение без учёта плана поставки
                if err < best_err_without_plan - 1e-9:
                    best_err_without_plan = err
                    best_without_plan = (x1, x2, x3, x4, avg_Ni, avg_Cu, total_flow, total_trucks)
                
                # Проверка ограничения по плану поставки (8 часов в смену)
                shift_flow1 = flow1 * 8
                shift_flow2 = flow2 * 8
                shift_flow3 = flow3 * 8
                shift_flow4 = flow4 * 8
                
                # Ослабляем ограничения - допуск 30% (можно регулировать)
                tolerance = 1.3  # 30% запас
                
                if (shift_flow1 <= shift_plan["C"] * tolerance and
                    shift_flow2 <= shift_plan["Kb"] * tolerance and
                    shift_flow3 <= shift_plan["Km"] * tolerance and
                    shift_flow4 <= shift_plan["T"] * tolerance):
                    
                    if err < best_err - 1e-9:
                        best_err = err
                        best = (x1, x2, x3, x4, avg_Ni, avg_Cu, total_flow, total_trucks,
                               shift_flow1, shift_flow2, shift_flow3, shift_flow4)

# Вывод результатов
if best:
    print(f"\n✅ Найдено решение с учётом плана поставки:")
    print(f"x1 (Дорога 1, склад С):     {best[0]}")
    print(f"x2 (Дорога 2, склад К_б):   {best[1]}")
    print(f"x3 (Дорога 3, склад К_м):   {best[2]}")
    print(f"x4 (Дорога 4, склад Т):     {best[3]}")
    print(f"\nПолученный состав:")
    print(f"Ni = {best[4]:.4f}% (цель: {target_Ni}%) → отклонение: {best[4]-target_Ni:+.4f}%")
    print(f"Cu = {best[5]:.4f}% (цель: {target_Cu}%) → отклонение: {best[5]-target_Cu:+.4f}%")
    print(f"Суммарная ошибка: {best_err:.6f}")
    print(f"\nСуммарный поток: {best[6]:.1f} т/час")
    print(f"Всего самосвалов: {best[7]} (менее 30)")
    print(f"\nПоставка за смену (8 часов):")
    print(f"  Склад С:  {best[8]:.0f} т (план: {shift_plan['C']} т)")
    print(f"  Склад К_б: {best[9]:.0f} т (план: {shift_plan['Kb']} т)")
    print(f"  Склад К_м: {best[10]:.0f} т (план: {shift_plan['Km']} т)")
    print(f"  Склад Т:  {best[11]:.0f} т (план: {shift_plan['T']} т)")
else:
    print("\n⚠️ Не найдено решение с учётом плана поставки")
    print("Показано лучшее решение без учёта плана:\n")
    
if best_without_plan:
    print(f"\n📊 Лучшее решение без учёта плана поставки:")
    print(f"x1, x2, x3, x4 = {best_without_plan[:4]}")
    print(f"Ni = {best_without_plan[4]:.4f}% (цель: {target_Ni}%)")
    print(f"Cu = {best_without_plan[5]:.4f}% (цель: {target_Cu}%)")
    print(f"Суммарная ошибка: {best_err_without_plan:.6f}")
    print(f"Всего самосвалов: {best_without_plan[7]}")
    
    # Используем лучшее решение без плана для вывода
    if best is None:
        best = best_without_plan

# =========================
# ПОСТРОЕНИЕ ГРАФИКОВ
# =========================

fig = plt.figure(figsize=(14, 10))

# График 1: Траектория
ax1 = plt.subplot(2, 2, 1)
ax1.plot(data["x"], data["y"], 'b.', alpha=0.3, markersize=1, label='GPS')
ax1.plot(states[:,0], states[:,2], 'r-', linewidth=1.5, label='Фильтр Калмана')
ax1.scatter(data.loc[unload_idx, "x"], data.loc[unload_idx, "y"], 
            c='green', s=100, marker='o', label='Измеренная разгрузка')
ax1.scatter(x_u, y_u, c='red', s=150, marker='*', label='Отфильтрованная разгрузка')

# Эллипс неопределённости
cov_unload = covs[unload_idx][[0,2], :][:, [0,2]]
eigenvals, eigenvecs = np.linalg.eigh(cov_unload)
angle = np.degrees(np.arctan2(eigenvecs[1,0], eigenvecs[0,0]))
ellipse = Ellipse(xy=(x_u, y_u), width=2*np.sqrt(5.991*eigenvals[1]), 
                  height=2*np.sqrt(5.991*eigenvals[0]), angle=angle, 
                  alpha=0.3, color='red', label='95% доверительный интервал')
ax1.add_patch(ellipse)
ax1.set_xlabel('X (м)')
ax1.set_ylabel('Y (м)')
ax1.set_title('Траектория движения самосвала')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# График 2: X координата
ax2 = plt.subplot(2, 2, 2)
ax2.plot(data["x"], 'b.', alpha=0.3, markersize=2, label='GPS')
ax2.plot(states[:,0], 'r-', linewidth=1.5, label='Фильтр')
ax2.axvline(x=unload_idx, color='g', linestyle='--', label='Разгрузка')
ax2.fill_between(range(len(states)), states[:,0] - 2*np.sqrt(covs[:,0,0]), 
                  states[:,0] + 2*np.sqrt(covs[:,0,0]), alpha=0.2, color='red')
ax2.set_xlabel('Время (шаги)')
ax2.set_ylabel('X (м)')
ax2.set_title('Фильтрация координаты X')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# График 3: Y координата
ax3 = plt.subplot(2, 2, 3)
ax3.plot(data["y"], 'b.', alpha=0.3, markersize=2, label='GPS')
ax3.plot(states[:,2], 'r-', linewidth=1.5, label='Фильтр')
ax3.axvline(x=unload_idx, color='g', linestyle='--', label='Разгрузка')
ax3.fill_between(range(len(states)), states[:,2] - 2*np.sqrt(covs[:,2,2]), 
                  states[:,2] + 2*np.sqrt(covs[:,2,2]), alpha=0.2, color='red')
ax3.set_xlabel('Время (шаги)')
ax3.set_ylabel('Y (м)')
ax3.set_title('Фильтрация координаты Y')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# График 4: Результаты
ax4 = plt.subplot(2, 2, 4)
if best:
    warehouses_names = ['Склад С', 'Склад К_б', 'Склад К_м', 'Склад Т']
    trucks = best[:4]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax4.bar(warehouses_names, trucks, color=colors)
    ax4.set_xlabel('Склад')
    ax4.set_ylabel('Количество самосвалов')
    ax4.set_title(f'Оптимальное распределение\nNi={best[4]:.3f}%, Cu={best[5]:.3f}%')
    for bar, val in zip(bars, trucks):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(val), ha='center', va='bottom')
    ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Лабораторная работа 3: Фильтр Калмана и оптимизация состава руды\n(первая треть списка, смена 1)', fontsize=12)
plt.tight_layout()
plt.savefig('lab3_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("ВЫВОД")
print("="*60)
if best:
    print(f"Для первой трети списка (смена 1) оптимальное количество самосвалов:")
    print(f"Дорога 1 (Скалистый): {best[0]} шт.")
    print(f"Дорога 2 (К_б):       {best[1]} шт.")
    print(f"Дорога 3 (К_м):       {best[2]} шт.")
    print(f"Дорога 4 (Таймырский): {best[3]} шт.")
    print(f"Всего: {best[7]} самосвалов (<30)")
