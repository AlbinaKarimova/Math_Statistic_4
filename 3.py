import pandas as pd
import numpy as np
from scipy.stats import t

data_1 = pd.read_excel('var10_Z2.xls', sheet_name='Sheet2', usecols='R')
data_1 = data_1[data_1['Z14,15,16_x'].notna()]
data_1_arr = list(data_1['Z14,15,16_x'])

data_2 = pd.read_excel('var10_Z2.xls', sheet_name='Sheet2', usecols='S')
data_2 = data_2[data_2['Z14,15,16_y'].notna()]
data_2_arr = list(data_2['Z14,15,16_y'])

alpha = 0.1

disp1 = np.var(data_1_arr, ddof=0) # Смещенная дисперсия
disp2 = np.var(data_2_arr, ddof=0)

std1 = np.sqrt(disp1)
std2 = np.sqrt(disp2)

mean1 = np.mean(data_1_arr)
mean2 = np.mean(data_2_arr)

n1 = len(data_1)
n2 = len(data_2)

print(f"Среднее: for x = {mean1}, for y = {mean2}")
print(f"Дисперсия: for x = {disp1}, for y = {disp2}")
print(f"Стандартное отклонение: for x = {std1}, for y = {std2}")
print(f"Объем выборки: for x = {n1}, for y = {n2}")

def corr(d1, d2):
    res = 0
    for i in range(len(d1)):
        res += (d1[i] - mean1) * (d2[i] - mean2)
    return res / (len(d1) * std1 * std2)

R = corr(data_1_arr, data_2_arr)
print(f"Коэффициент корреляции R = {R}")

t_pr = np.sqrt(n1 - 2) * R / np.sqrt(1 - R**2)
print(f"Преобразование Стьюдента t = {t_pr}")

# КРИТИЧЕСКАЯ ОБЛАСТЬ: |t| > C, так как нулевая гипотеза утверждает, что corr=0
C = t.ppf(1-alpha/2, n1-2) # - КРИТИЧЕСКАЯ КОНСТАНТА(нижняя квантиль Стьюдента)
print(f"Критическая константа: {C}")
if (np.abs(t_pr) > C):
    print("Нулевая гипотеза отвергается")
else:
    print("Нулевая гипотеза принимается")

p_value = 2 * (1 - t.cdf(np.abs(t_pr), n1-2))

print(f"Критический уровень значимости: {p_value}")

if (p_value > alpha):
    print("Отклонение от нулевой гипотезы не значимо, принимается нулевая гипотеза")
else:
    print("Отклонение от нулевой гипотезы значимо, принимается альтернатива")