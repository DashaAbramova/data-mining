import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import rfft, irfft
from math import floor
from matplotlib.patches import Ellipse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# ============================================================
# 1. ЗАГРУЗКА И ПЕРВИЧНАЯ ОБРАБОТКА ДАННЫХ
# ============================================================

# Настройка параметров отображения DataFrame
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Загрузка Excel-файла и просмотр листов
excel_file = pd.ExcelFile('PQM - podatki za pilotni projekt (OCR12VM) - ENG.xlsx')
print("Доступные листы:", excel_file.sheet_names)

# --- Обработка основной таблицы (плавки, химия, добавки) ---
main_data = pd.read_excel('PQM - podatki za pilotni projekt (OCR12VM) - ENG.xlsx', 
                          sheet_name='Table1 (basic)', 
                          skiprows=[0])

# Убираем служебные колонки, не влияющие на химсостав
columns_to_drop = ['QualityRequirement', 'QualNo', 'CustID', 'CustVer', 
                   'InternalVer', 'MetalRavneQualityName', 'SteelGroup', 
                   'Month', 'Year']
main_data.drop(columns_to_drop, axis=1, inplace=True)

# Оставляем только уникальные номера плавок
main_data.drop_duplicates(subset=['HeatNo'], keep='first', inplace=True)

# Чистим пропуски в ключевых легирующих элементах
main_data.dropna(subset=['Cr_Last_EOP', 'Mo_Last_EOP', 'Ni_Last_EOP', 
                         'V_Last_EOP', 'W_Last_EOP'], inplace=True)

# --- Таблица с нормативными пределами ---
limits_data = pd.read_excel('PQM - podatki za pilotni projekt (OCR12VM) - ENG.xlsx',
                            sheet_name='Table2 (limits)',
                            skiprows=[0])

# Удаляем те же метаданные
limits_data.drop(columns_to_drop, axis=1, inplace=True)

# Убираем финальные значения, чтобы избежать дублирования с main_data
limits_data.drop(['Date', 'Cr_Final', 'Ni_Final', 'Mo_Final', 
                  'V_Final', 'W_Final'], axis=1, inplace=True)

limits_data.drop_duplicates(subset=['HeatNo'], keep='first', inplace=True)

# Слияние основной информации с лимитами
main_data = main_data.merge(limits_data, on='HeatNo', how='inner')

# --- Таблица легирующих добавок на LFVD (в кг) ---
alloys_data = pd.read_excel('PQM - podatki za pilotni projekt (OCR12VM) - ENG.xlsx',
                            sheet_name='Table4 (alloys)',
                            skiprows=[0])

# Исключаем часть ферросплавов из основного набора
ferro_to_remove = ['LFVD_FeCrA', 'LFVD_FeCrC', 'LFVD_NiGran', 'LFVD_NiKatode',
                   'LFVD_FeMo', 'LFVD_Polymox', 'LFVD_FeV', 'LFVD_FeW72', 'LFVD_WPaketi']
main_data.drop(ferro_to_remove, axis=1, inplace=True)

alloys_data.drop_duplicates(subset=['HeatNo'], keep='first', inplace=True)
main_data = main_data.merge(alloys_data, on='HeatNo', how='inner')

# --- Данные по завалке обычного лома (порционно, до 6 шагов) ---
scrap_data = pd.read_excel('PQM - podatki za pilotni projekt (OCR12VM) - ENG.xlsx',
                           sheet_name='Table8 (scrap)',
                           skiprows=[0])
scrap_data.drop_duplicates(subset=['HeatNo'], keep='first', inplace=True)
main_data = main_data.merge(scrap_data, on='HeatNo', how='inner')

# --- Информация о легированном ломе (Cr, Mo, V и пр.) ---
alloyed_scrap = pd.read_excel('PQM - podatki za pilotni projekt (OCR12VM) - ENG.xlsx',
                              sheet_name='Table9 (all. scr.)',
                              skiprows=[0])
alloyed_scrap.drop_duplicates(subset=['HeatNo'], keep='first', inplace=True)
main_data = main_data.merge(alloyed_scrap, on='HeatNo', how='inner')

# --- Детализация по углеродистому (нелегированному) лому ---
unalloyed_scrap = pd.read_excel('PQM - podatki za pilotni projekt (OCR12VM) - ENG.xlsx',
                                sheet_name='Table10 (unall. scr.)',
                                skiprows=[0])
unalloyed_scrap.drop_duplicates(subset=['HeatNo'], keep='first', inplace=True)
main_data = main_data.merge(unalloyed_scrap, on='HeatNo', how='inner')

# Вывод информации о результирующем датафрейме
print(main_data.head(3))
print(f'Общее количество записей: {len(main_data)}')
print(f'Число признаков: {len(main_data.columns)}')

# Сохранение в CSV
main_data.to_csv('итоговая таблица.csv', sep=';', index=False)

# ============================================================
# 2. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ЛЕГИРУЮЩИХ ЭЛЕМЕНТОВ
# ============================================================

# Определяем интересующие нас химические элементы
target_elements = ["Cr_Final", "Ni_Final", "Mo_Final", "V_Final", "W_Final"]

# Отбираем только существующие колонки из датафрейма
available_elements = [elem for elem in target_elements if elem in main_data.columns]

# Создаём гистограммы для визуализации распределения легирующих компонентов
fig, axes = plt.subplots(1, len(available_elements), figsize=(16, 6))
if len(available_elements) == 1:
    axes = [axes]

# Построение каждого графика
for idx, element in enumerate(available_elements):
    axes[idx].hist(main_data[element].dropna(), bins=20, color='steelblue', 
                   edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Распределение {element}', fontsize=12)
    axes[idx].set_xlabel('Содержание, %')
    axes[idx].set_ylabel('Частота')
    axes[idx].grid(True, linestyle='--', alpha=0.3)

plt.suptitle('Анализ распределения легирующих элементов в финальной продукции', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 3. ОЧИСТКА ДАННЫХ
# ============================================================

# Загрузка предварительно обработанных данных
dataset = pd.read_csv('итоговая таблица.csv', sep=';')

# Просмотр статистической сводки
dataset.describe()

# Поиск константных признаков (столбцы с нулевым стандартным отклонением)
constant_columns = []
for col in dataset.describe().columns:
    if dataset.describe().loc['std', col] == 0:
        constant_columns.append(col)

# Исключения: сохраняем целевые пределы, даже если они константные
columns_to_keep = ['Ni_LowerLimit', 'Ni_Target', 'W_LowerLimit', 'W_Target']
for col in columns_to_keep:
    if col in constant_columns:
        constant_columns.remove(col)

# Удаляем все найденные константные столбцы
dataset.drop(constant_columns, axis=1, inplace=True)

# Очистка данных о шихтовке лома (информация о порционной загрузке на 6 этапах)
scrap_columns = [
    'Fill#1_ScrapName', 'Fill#1_ScrapWeight',
    'Fill#2_ScrapName', 'Fill#2_ScrapWeight',
    'Fill#3_ScrapName', 'Fill#3_ScrapWeight',
    'Fill#4_ScrapName', 'Fill#4_ScrapWeight',
    'Fill#5_ScrapName', 'Fill#5_ScrapWeight',
    'Fill#6_ScrapName', 'Fill#6_ScrapWeight'
]
dataset.drop(scrap_columns, axis=1, inplace=True)

# Вывод итоговой информации
print(f'Количество наблюдений: {len(dataset)}')
print(f'Число признаков после очистки: {len(dataset.columns)}')

# Сохранение очищенного датасета
dataset.to_csv('итоговая таблица_испр.csv', sep=';', index=False)

# ============================================================
# 4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ============================================================

# Загрузка очищенного датасета
df = pd.read_csv('итоговая таблица_испр.csv', sep=';')

# Исключаем нечисловые и идентифицирующие поля перед корреляционным анализом
df_numeric = df.drop(['HeatNo', 'Date'], axis=1)

# Расчёт матрицы парных корреляций Пирсона (только для числовых признаков)
correlation_matrix = df_numeric.corr()

print(f'Количество признаков в корреляционной матрице: {len(correlation_matrix)}')
print("\nПервые 3 строки корреляционной матрицы:")
print(correlation_matrix.head(3))

# ============================================================
# 5. МАТРИЦА КОРРЕЛЯЦИИ (ТЕПЛОВАЯ КАРТА)
# ============================================================

# Определяем перечень анализируемых параметров (до и после обработки)
parameters = [
    "Cr_Last_EOP", "Ni_Last_EOP", "Mo_Last_EOP", "V_Last_EOP", "W_Last_EOP",
    "Cr_Final", "Ni_Final", "Mo_Final", "V_Final", "W_Final"
]

# Построение тепловой карты для визуализации взаимосвязей
plt.figure(figsize=(10, 8))
sns.heatmap(main_data[parameters].corr(), 
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            square=True)

plt.title("Матрица корреляции: содержание элементов\nна промежуточном и финальном этапах", 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 6. АНАЛИЗ ОТКЛОНЕНИЙ ОТ ЦЕЛЕВЫХ ЗНАЧЕНИЙ
# ============================================================

# Расчёт отклонений
main_data['Cr_exceed'] = main_data['Cr_Final'] - main_data['Cr_Target']
main_data['Ni_exceed'] = main_data['Ni_Final'] - main_data['Ni_Target']

# Статистика
cr_margin = (main_data['Cr_Target'] - main_data['Cr_LowerLimit']).mean()
ni_margin = (main_data['Ni_Target'] - main_data['Ni_LowerLimit']).mean()
print(f'Средний запас по хрому (Target - LowerLimit) = {cr_margin:.4f}')
print(f'Средний запас по никелю (Target - LowerLimit) = {ni_margin:.4f}')

# Построение графиков с помощью matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Гистограмма для хрома
ax1.hist(main_data['Cr_exceed'].dropna(), bins=50, color='steelblue', 
         edgecolor='black', alpha=0.7, range=(-0.2, 0.2))
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Целевое значение')
ax1.set_title('Отклонения по хрому (Cr)', fontsize=12)
ax1.set_xlabel('Отклонение от цели, %')
ax1.set_ylabel('Количество плавок')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Гистограмма для никеля
ax2.hist(main_data['Ni_exceed'].dropna(), bins=50, color='darkorange', 
         edgecolor='black', alpha=0.7, range=(-0.5, 0.5))
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Целевое значение')
ax2.set_title('Отклонения по никелю (Ni)', fontsize=12)
ax2.set_xlabel('Отклонение от цели, %')
ax2.set_ylabel('Количество плавок')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Распределение отклонений финального содержания от целевых значений', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 7. ОПТИМИЗАЦИЯ ЦЕЛЕВЫХ ЗНАЧЕНИЙ (СОВЕТЧИК)
# ============================================================

# Создание "рекомендованных" целевых значений (средняя точка между Target и LowerLimit)
main_data['Cr_optimized_target'] = (main_data['Cr_Target'] + main_data['Cr_LowerLimit']) / 2
main_data['Ni_optimized_target'] = (main_data['Ni_Target'] + main_data['Ni_LowerLimit']) / 2

# Расчёт отклонений от оптимизированной цели ("советчика")
main_data['Cr_deviation'] = main_data['Cr_Final'] - main_data['Cr_optimized_target']
main_data['Ni_deviation'] = main_data['Ni_Final'] - main_data['Ni_optimized_target']

# Вывод статистики по технологическим допускам
cr_range = (main_data['Cr_Target'] - main_data['Cr_LowerLimit']).mean()
ni_range = (main_data['Ni_Target'] - main_data['Ni_LowerLimit']).mean()
print(f'Средний диапазон Cr (Target - LowerLimit) = {cr_range:.4f}')
print(f'Средний диапазон Ni (Target - LowerLimit) = {ni_range:.4f}')

# Создание графиков с помощью matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График для хрома
ax1.hist(main_data['Cr_deviation'].dropna(), bins=50, 
         color='forestgreen', edgecolor='black', alpha=0.7, range=(-0.2, 0.2))
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Рекомендованная цель')
ax1.set_title('Отклонения по Cr от оптимизированного ориентира', fontsize=12)
ax1.set_xlabel('Отклонение от рекомендованной цели, %')
ax1.set_ylabel('Количество плавок')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График для никеля
ax2.hist(main_data['Ni_deviation'].dropna(), bins=50, 
         color='purple', edgecolor='black', alpha=0.7, range=(-0.5, 0.5))
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Рекомендованная цель')
ax2.set_title('Отклонения по Ni от оптимизированного ориентира', fontsize=12)
ax2.set_xlabel('Отклонение от рекомендованной цели, %')
ax2.set_ylabel('Количество плавок')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Распределение отклонений от рекомендованных целевых значений\n(средняя точка между Target и LowerLimit)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 8. АНАЛИЗ ВСЕХ ЭЛЕМЕНТОВ
# ============================================================

# Список анализируемых элементов
elements = ["Cr", "Ni", "Mo", "V", "W"]

# Словарь с настройками бинов для каждого элемента
bin_configs = {
    "Cr": dict(start=-0.2, end=0.2, size=0.005),
    "Ni": dict(start=-0.5, end=0.5, size=0.005),
    "Mo": dict(start=-0.2, end=0.2, size=0.005),
    "V": dict(start=-0.1, end=0.1, size=0.002),
    "W": dict(start=-0.3, end=0.3, size=0.005)
}

# Расчёт для каждого элемента
for el in elements:
    main_data[f'{el}_adviser_target'] = (main_data[f'{el}_Target'] + main_data[f'{el}_LowerLimit']) / 2
    main_data[f'{el}_exceed'] = main_data[f'{el}_Final'] - main_data[f'{el}_adviser_target']
    diff_mean = (main_data[f'{el}_Target'] - main_data[f'{el}_LowerLimit']).mean()
    print(f'Средняя разница между {el}_Target и {el}_LowerLimit = {diff_mean:.4f}')

# Создание сетки графиков (5 строк, 1 столбец)
fig, axes = plt.subplots(5, 1, figsize=(10, 16))
fig.suptitle('Анализ отклонений по 5 элементам от рекомендованной цели', 
             fontsize=14, fontweight='bold', y=0.995)

# Построение графиков
for i, el in enumerate(elements):
    config = bin_configs.get(el, dict(start=-0.2, end=0.2, size=0.005))
    
    # Создание бинов для гистограммы
    bins = np.arange(config['start'], config['end'] + config['size'], config['size'])
    
    # Построение гистограммы
    axes[i].hist(main_data[f'{el}_exceed'].dropna(), bins=bins, 
                 color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Рекомендованная цель')
    axes[i].set_title(f'Гистограмма превышения {el}', fontsize=11)
    axes[i].set_xlabel('Отклонение от рекомендованной цели, %')
    axes[i].set_ylabel('Количество плавок')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    # Добавление статистики на график
    mean_val = main_data[f'{el}_exceed'].mean()
    median_val = main_data[f'{el}_exceed'].median()
    axes[i].text(0.95, 0.95, f'среднее: {mean_val:.4f}\nмедиана: {median_val:.4f}', 
                 transform=axes[i].transAxes, verticalalignment='top',
                 horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# ============================================================
# 9. АНАЛИЗ ФЕРРОХРОМА
# ============================================================

# Вывод образца данных
print("Первые 3 записи (вес слитков, Cr на EOP, Cr финальный, добавки FeCr):")
print(main_data[['TotalIngotsWeight', 'Cr_Last_EOP', 'Cr_Final', 'LFVD_FeCrA', 'LFVD_FeCrC']].head(3))

# Анализ плавок, где не производилось легирование феррохромом
no_chromium_additions = main_data[
    (main_data['LFVD_FeCrA'] == 0) & 
    (main_data['LFVD_FeCrC'] == 0) & 
    (main_data['LFVD_FeCrCSi'] == 0) & 
    (main_data['LFVD_FeCrC51'] == 0)
]
print(f'Количество плавок без добавления феррохрома (легирование не требовалось) = {len(no_chromium_additions)}')

# ============================================================
# 10. ЭКОНОМИЧЕСКАЯ ОЦЕНКА ПЕРЕРАСХОДА
# ============================================================

# Отбор плавок с превышением финального содержания хрома
overuse_heats = main_data[(main_data['Cr_Final'] > main_data['Cr_Last_EOP']) & 
                           (main_data['Cr_Final'] > main_data['Cr_Target'])].copy()

# Расчёт удельного расхода феррохрома на 1% прироста хрома
overuse_heats['FeCr_per_1pct'] = overuse_heats.apply(
    lambda row: (row['LFVD_FeCrA'] + row['LFVD_FeCrC']) / 
                (row['Cr_Final'] - row['Cr_Last_EOP']) / 
                row['TotalIngotsWeight'] * 100, 
    axis=1
)

# Вычисление объёма перерасхода феррохрома
overuse_heats['Cr_overuse_kg'] = overuse_heats.apply(
    lambda row: (row['LFVD_FeCrA'] + row['LFVD_FeCrC']) - 
                (row['Cr_Target'] - row['Cr_Last_EOP']) * 
                row['TotalIngotsWeight'] * row['FeCr_per_1pct'] / 100, 
    axis=1
)

# Вывод проверочных данных
print("Контрольные расчёты по перерасходу феррохрома (3 примера):")
check_columns = ['TotalIngotsWeight', 'Cr_Last_EOP', 'Cr_Target', 'Cr_Final', 
                 'LFVD_FeCrA', 'LFVD_FeCrC', 'FeCr_per_1pct', 'Cr_overuse_kg']
print(overuse_heats[check_columns].head(3).round(2))

# Экономическая оценка: годовой перерасход феррохрома
total_overuse_kg = overuse_heats['Cr_overuse_kg'].sum()
annual_overuse_kg = total_overuse_kg / 10
price_per_kg = 300
annual_cost_million = annual_overuse_kg * price_per_kg / 1_000_000

print("\n" + "="*55)
print("ЭКОНОМИЧЕСКАЯ ОЦЕНКА ПЕРЕРАСХОДА ФЕРРОХРОМА")
print("="*55)
print(f"Годовой перерасход FeCr: {annual_overuse_kg:,.0f} кг")
print(f"Ориентировочная стоимость: {annual_cost_million:.2f} млн руб")
print("="*55)

# ============================================================
# 11. АНАЛИЗ ПЕРЕРАСХОДА ВСЕХ ЭЛЕМЕНТОВ
# ============================================================

# Загрузка подготовленного датасета
dataset = pd.read_csv('итоговая таблица_испр.csv', sep=';')

# Формирование компромиссного целевого ориентира
dataset['Cr_compromise_target'] = (dataset["Cr_LowerLimit"] + dataset["Cr_Target"]) / 2

# Отбор плавок с превышением финального Cr
overuse_vs_mid = dataset[(dataset['Cr_Final'] > dataset['Cr_Last_EOP']) & 
                          (dataset['Cr_Final'] > dataset['Cr_compromise_target'])].copy()

# Расчёт эффективности легирования
overuse_vs_mid['FeCr_efficiency'] = overuse_vs_mid.apply(
    lambda row: (row['LFVD_FeCrA'] + row['LFVD_FeCrC']) / 
                (row['Cr_Final'] - row['Cr_Last_EOP']) / 
                row['TotalIngotsWeight'] * 100 
                if (row['Cr_Final'] - row['Cr_Last_EOP']) > 0 else 0, 
    axis=1
)

# Расчёт избыточного расхода феррохрома
overuse_vs_mid['Cr_excess_vs_mid'] = overuse_vs_mid.apply(
    lambda row: (row['LFVD_FeCrA'] + row['LFVD_FeCrC']) - 
                (row['Cr_compromise_target'] - row['Cr_Last_EOP']) * 
                row['TotalIngotsWeight'] * row['FeCr_efficiency'] / 100, 
    axis=1
)

# Вывод контрольных данных
print("="*90)
print("КОНТРОЛЬНЫЕ РАСЧЁТЫ (первые 5 плавок с перерасходом)")
print("="*90)
check_columns = ['HeatNo', 'TotalIngotsWeight', 'Cr_Last_EOP', 'Cr_LowerLimit', 
                 'Cr_compromise_target', 'Cr_Target', 'Cr_Final', 
                 'LFVD_FeCrA', 'LFVD_FeCrC', 'FeCr_efficiency', 'Cr_excess_vs_mid']
print(overuse_vs_mid[check_columns].head(5).round(3))

# Экономическая оценка перерасхода
total_excess = overuse_vs_mid['Cr_excess_vs_mid'].sum()
annual_excess = total_excess / 10
annual_cost_million = annual_excess * price_per_kg / 1_000_000

print("\n" + "="*60)
print("ЭКОНОМИЧЕСКАЯ ОЦЕНКА ПЕРЕРАСХОДА")
print("="*60)
print(f"Суммарный перерасход FeCr за весь период: {total_excess:,.0f} кг")
print(f"Годовой перерасход FeCr: {annual_excess:,.0f} кг")
print(f"Годовая стоимость перерасхода: {annual_cost_million:.3f} млн руб")
print("="*60)

print(f"\nСтатистика по выборке:")
print(f"  - Количество плавок с перерасходом: {len(overuse_vs_mid)}")
print(f"  - Средний перерасход на плавку: {overuse_vs_mid['Cr_excess_vs_mid'].mean():.0f} кг")
print(f"  - Медианный перерасход: {overuse_vs_mid['Cr_excess_vs_mid'].median():.0f} кг")

# ============================================================
# 12. АНАЛИЗ ДРУГИХ ЭЛЕМЕНТОВ (Mo, Ni, V, W)
# ============================================================

material_prices = {
    'Mo': [['LFVD_FeMo', 'LFVD_Polymox'], 300],
    'Ni': [['LFVD_NiGran'], 300],
    'V':  [['LFVD_FeV'], 300],
    'W':  [['LFVD_FeW72'], 300]
}

for element, config in material_prices.items():
    alloy_columns = config[0]
    price_per_kg = config[1]

    # Создание компромиссного целевого ориентира
    target_mid = f'{element}_mid_target'
    if target_mid not in dataset.columns:
        dataset[target_mid] = (dataset[f"{element}_LowerLimit"] + dataset[f"{element}_Target"]) / 2

    # Суммирование всех добавок
    dataset[f'Total_{element}_Added'] = dataset[alloy_columns].sum(axis=1)

    # Фильтрация плавок с потенциальным перерасходом
    filtered_data = dataset[
        (dataset[f'{element}_Final'] > dataset[f'{element}_Last_EOP']) &
        (dataset[f'{element}_Final'] > dataset[target_mid]) &
        (dataset[f'Total_{element}_Added'] > 0)
    ].copy()

    if filtered_data.empty:
        print(f"\n--- Элемент {element}: данные для анализа перерасхода отсутствуют ---")
        continue

    # Расчёт удельной эффективности легирования
    filtered_data[f'{element}_efficiency'] = filtered_data.apply(
        lambda row: row[f'Total_{element}_Added'] / 
                    (row[f'{element}_Final'] - row[f'{element}_Last_EOP']) / 
                    row['TotalIngotsWeight'] * 100
                    if (row[f'{element}_Final'] - row[f'{element}_Last_EOP']) > 0 else 0, 
        axis=1
    )

    # Расчёт избыточного расхода
    filtered_data[f'{element}_excess_mid'] = filtered_data.apply(
        lambda row: row[f'Total_{element}_Added'] - 
                    (row[target_mid] - row[f'{element}_Last_EOP']) * 
                    row['TotalIngotsWeight'] * row[f'{element}_efficiency'] / 100,
        axis=1
    )

    # Вывод контрольных данных
    print(f"\n{'='*60}")
    print(f"АНАЛИЗ ПЕРЕРАСХОДА ДЛЯ ЭЛЕМЕНТА: {element}")
    print(f"{'='*60}")
    
    display_columns = ['HeatNo', 'TotalIngotsWeight', f'{element}_Last_EOP', 
                       f'{element}_LowerLimit', target_mid, f'{element}_Target', 
                       f'{element}_Final'] + alloy_columns + [f'{element}_efficiency', 
                       f'{element}_excess_mid']
    print(filtered_data[display_columns].head(5).round(3))

    # Экономическая оценка
    total_excess_kg = filtered_data[f'{element}_excess_mid'].sum()
    annual_excess_kg = total_excess_kg / 10
    annual_cost_million = annual_excess_kg * price_per_kg / 1_000_000

    print(f"\nФИНАНСОВЫЙ АНАЛИЗ ПО ЭЛЕМЕНТУ {element}:")
    print(f"  Суммарный перерасход за весь период: {total_excess_kg:,.0f} кг")
    print(f"  Годовой перерасход: {annual_excess_kg:,.0f} кг")
    print(f"  Годовая стоимость (при цене {price_per_kg} руб/кг): {annual_cost_million:.3f} млн руб")
    print("-" * 60)

# ============================================================
# 13. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИРОВАНИЯ
# ============================================================

print(f"Импорт данных выполнен: {dataset.shape[0]} записей, {dataset.shape[1]} полей")

# Заполнение пропусков в технологических добавках и ломе
ferroalloy_cols = [col for col in dataset.columns if col.startswith('LFVD_')]
scrap_cols = [col for col in dataset.columns if col.startswith('PV_')]
dataset[ferroalloy_cols + scrap_cols] = dataset[ferroalloy_cols + scrap_cols].fillna(0)

# Целевые переменные
target_variables = ['Cr_Final', 'Ni_Final', 'Mo_Final', 'V_Final', 'W_Final']

# Базовые характеристики
initial_conditions = [
    'TotalIngotsWeight',
    'Cr_Last_EOP', 'Ni_Last_EOP', 'Mo_Last_EOP', 'V_Last_EOP', 'W_Last_EOP',
    'Cr_LowerLimit', 'Cr_Target', 'Cr_UpperLimit',
    'Ni_LowerLimit', 'Ni_Target', 'Ni_UpperLimit',
    'Mo_LowerLimit', 'Mo_Target', 'Mo_UpperLimit',
    'V_LowerLimit',  'V_Target', 'V_UpperLimit',
    'W_LowerLimit',  'W_Target', 'W_UpperLimit'
]

# Управляемые параметры
alloy_additives = [col for col in dataset.columns if col.startswith('LFVD_')]
scrap_materials = [col for col in dataset.columns if col.startswith('PV_')]

# Формирование полного набора признаков
model_features = initial_conditions + alloy_additives + scrap_materials

# Удаление идентификационных полей
model_data = dataset.drop(columns=['HeatNo', 'Date'], errors='ignore')

# Разделение на матрицу признаков и вектор целей
X_matrix = model_data[model_features]
y_matrix = model_data[target_variables]

print("\n" + "="*50)
print("СТРУКТУРА ДАННЫХ ДЛЯ МОДЕЛИРОВАНИЯ")
print("="*50)
print(f"Признаки (X): {X_matrix.shape[1]} колонок, {X_matrix.shape[0]} строк")
print(f"Цели (y):    {y_matrix.shape[1]} колонок, {y_matrix.shape[0]} строк")
print("="*50)

# ============================================================
# 14. ОБУЧЕНИЕ МОДЕЛЕЙ ГРАДИЕНТНОГО БУСТИНГА
# ============================================================

# Разделение данных: 80% на обучение, 20% на тестирование
X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_matrix, test_size=0.2, random_state=42)

print("="*60)
print("ЗАПУСК ОБУЧЕНИЯ МОДЕЛЕЙ ГРАДИЕНТНОГО БУСТИНГА")
print("="*60)
print(f"Обучающая выборка: {X_train.shape[0]} записей")
print(f"Тестовая выборка:   {X_test.shape[0]} записей")
print(f"Количество признаков: {X_train.shape[1]}")
print("="*60)

# Хранилище для моделей и метрик
trained_models = {}
performance_records = []

# Обучение модели для каждого целевого элемента
for target in target_variables:
    print(f"\nОбработка элемента: {target}")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    model.fit(X_train, y_train[target])
    print(f"Модель обучена")
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test[target], y_pred)
    bias = np.mean(y_pred - y_test[target])
    
    trained_models[target] = model
    performance_records.append({
        'Элемент': target,
        'ME': round(bias, 5),
        'MAE': round(mae, 5),
    })
    
    filename = f'model_{target}.joblib'
    joblib.dump(model, filename)
    print(f"Модель сохранена в {filename}")
    print(f"MAE = {mae:.5f}, ME = {bias:.5f}")

# Вывод таблицы с метриками
print("\n" + "="*60)
print("ИТОГОВЫЕ МЕТРИКИ ТОЧНОСТИ МОДЕЛЕЙ")
print("="*60)
metrics_df = pd.DataFrame(performance_records)
print(metrics_df.to_string(index=False))
print("="*60)

# Сохранение списка признаков
joblib.dump(model_features, 'model_features_list.joblib')
print(f"\nСписок признаков сохранён в model_features_list.joblib")
print(f"  (всего {len(model_features)} признаков)")

# ============================================================
# 15. ФУНКЦИЯ-СОВЕТЧИК
# ============================================================

def get_advisor_recommendation(current_heat_data, historical_database, known_features_list, adjustable_features_list):
    """
    Рекомендует оптимальный режим легирования на основе исторических данных
    
    Параметры:
    - current_heat_data: известные параметры текущей плавки
    - historical_database: DataFrame с историческими данными
    - known_features_list: список известных признаков (базовые условия)
    - adjustable_features_list: список управляемых признаков (добавки)
    
    Возвращает:
    - best_option: словарь с оптимальной рекомендацией
    """
    
    best_total_cost = np.inf
    optimal_recommendation = None
    
    # Перебор исторических решений
    for hist_idx in historical_database.index:
        candidate_solution = historical_database.loc[hist_idx]
        
        # Формирование вектора признаков
        test_vector = current_heat_data.copy()
        for additive_col in adjustable_features_list:
            test_vector[additive_col] = candidate_solution[additive_col]
        
        # Подготовка входа для модели
        X_input = pd.DataFrame([test_vector])[model_features]
        
        # Прогнозирование финального состава
        predicted_composition = {}
        for target_element in target_variables:
            predicted_composition[target_element] = trained_models[target_element].predict(X_input)[0]
        
        # Проверка соответствия технологическим допускам
        within_spec = True
        for element in ['Cr', 'Ni', 'Mo', 'V', 'W']:
            lower_bound = test_vector.get(f'{element}_LowerLimit', 0)
            upper_bound = test_vector.get(f'{element}_UpperLimit', 100)
            predicted_value = predicted_composition[f'{element}_Final']
            
            if not (lower_bound - 0.03 <= predicted_value <= upper_bound + 0.05):
                within_spec = False
                break
        
        if not within_spec:
            continue
        
        # Расчёт суммарного расхода добавок
        total_additives = sum(test_vector[col] for col in adjustable_features_list)
        
        # Обновление лучшего решения
        if total_additives < best_total_cost:
            best_total_cost = total_additives
            optimal_recommendation = {
                'recommended_additions': {col: test_vector[col] for col in adjustable_features_list if test_vector[col] > 0},
                'predicted_final_composition': predicted_composition,
                'total_additions_kg': total_additives,
                'source_heat_id': hist_idx
            }
    
    return optimal_recommendation

# ============================================================
# 16. ПРИМЕР ИСПОЛЬЗОВАНИЯ СОВЕТЧИКА
# ============================================================

test_melt_idx = 11
sample_melt_known = main_data[initial_conditions].iloc[test_melt_idx].to_dict()

recommendation = get_advisor_recommendation(
    current_heat_data=sample_melt_known,
    historical_database=main_data,
    known_features_list=initial_conditions,
    adjustable_features_list=alloy_additives + scrap_materials
)

print("\n" + "="*60)
print("РЕЗУЛЬТАТ РАБОТЫ СОВЕТЧИКА")
print("="*60)

if recommendation is None:
    print("Советчик не нашёл решение!")
else:
    print("\nРекомендация сформирована")
    print(f"\nСуммарный вес добавок: {recommendation['total_additions_kg']:.1f} кг")
    print(f"Историческая плавка №{recommendation['source_heat_id']}")
    
    print("\nРекомендуемый состав добавок:")
    for material, weight in recommendation['recommended_additions'].items():
        if weight > 0:
            print(f"   - {material}: {weight:.1f} кг")
    
    print("\nПрогнозируемый финальный состав:")
    for target_name, predicted_val in recommendation['predicted_final_composition'].items():
        element = target_name.replace('_Final', '')
        print(f"   - {element}: {predicted_val:.4f}%")
