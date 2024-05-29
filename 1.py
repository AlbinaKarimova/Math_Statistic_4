import pandas as pd
import numpy as np
from scipy.stats import chi2

data_1 = pd.read_excel('var10_Z2.xls', sheet_name='Sheet2', usecols='R')
data_1 = data_1[data_1['Z14,15,16_x'].notna()]
data_1_arr = list(data_1['Z14,15,16_x'])
data_1_arr.sort()

data_2 = pd.read_excel('var10_Z2.xls', sheet_name='Sheet2', usecols='S')
data_2 = data_2[data_2['Z14,15,16_y'].notna()]
data_2_arr = list(data_2['Z14,15,16_y'])
data_2_arr.sort()

alpha = 0.01
X_1 = 116.05
X_r = 124.05
Y_1 = 81.05
Y_s = 87.05
r = 3
s = 4

bins_X = np.linspace(X_1, X_1 + ((X_r - X_1)/r) * r, r-1) # Разбиение чиловой прямой на r групп
bins_Y = np.linspace(Y_1, Y_1 + ((Y_s - Y_1)/s) * s, s-1) # Разбиение чиловой прямой на s групп

# Количество попаданий в интервал
def count_v(data, left, right):
    v = 0
    v_data = []
    for i in range(len(data)):
        if data[i] > left and data[i] <= right: # Берем интервалы типа ( ; ]
            v += 1
            v_data.append(data[i])
    return v, v_data

# Массив из количества попаданий
def find_v(data, bins_, count):
    v = []
    v_data = []
    check = count_v(data, -1, bins_[0])
    v.append(check[0])
    v_data.append(check[1])
    for i in range(0, count-2):
        check = count_v(data, bins_[i], bins_[i+1])
        v.append(check[0])
        v_data.append(check[1])
    v.append(len(data)-sum(v))
    v_data.append(count_v(data, bins_[count-2], 100000)[1])
    return v, v_data


def count_v_all(data1, data2):
    v = np.array([[0] * r for i in range(s)])
    for j in range(r):
        v[0, j] = find_v(data1[1][j], bins_Y, s)[0][s - 1]
    for i in range(1, s):
        v[i, 0] = find_v(data2[1][i], bins_X, r)[0][0]
    return v

d1 = find_v(data_1_arr, bins_X, r)[1]
d2 = find_v(data_2_arr, bins_Y, s)[1]
d2 = d2[::-1]
union_array = []
for i in range(len(d2)):
    row = []
    for j in range(len(d1)):
        row.append(d1[j] + d2[i])
    union_array.append(row)

def create_bins(bins):
    new_bins = []
    new_bins.append(-1)
    for i in range(len(bins)):
        new_bins.append(bins[i])
    new_bins.append(200)
    return new_bins

new_bins_x = create_bins(bins_X)
new_bins_y = create_bins(bins_Y)
print(new_bins_x)
print(new_bins_y)
# ИЗНАЧАЛЬНО К РАЗБИЕНИЯМ ДОБАВИТЬ ГРАНИЦЫ
# new_bins_x = [-1, 116.05, 124.05, 200]
# new_bins_y = [-1, 81.05, 84.05, 87.05, 200]

def create_table(data, bins_x, bins_y):
    bins_y = bins_y[::-1]
    v = np.array([[0] * r for i in range(s)])
    for i in range(len(bins_y)-1):
        for j in range(len(bins_x)-1):
            count = count_v(data[i][j], bins_y[i], bins_x[j])
            v[i, j] = count
    return v

H, Y_bins, X_bins = np.histogram2d(data_2_arr, data_1_arr, bins=[new_bins_y, new_bins_x])

print("Частоты попаданий в ячейки:")
#меняем порядок строк
ar = np.copy(H)
H[0] = ar[3]
H[1] = ar[2]
H[2] = ar[1]
H[3] = ar[0]
print(H)

# Считаем сумму столбцов
def sum_col(arr):
    return list(map(sum, zip(*arr)))

#Считаем сумму строк
def sum_row(arr):
    return [sum(i) for i in arr]

def statistic(v):
    res = 0
    n = len(data_1_arr)
    for k in range(r):
        for j in range(s):
            col = sum_col(v)
            row = sum_row(v)
            res += (n * v[j, k] - col[k] * row[j]) ** 2 / (col[k] * row[j])
    return res / n

X_stat = statistic(H)
print(f"Статистика: {round(X_stat, 3)}")
# КРИТИЧЕСКАЯ ОБЛАСТЬ: X^2 > C

C = chi2.ppf(1-alpha, (s-1)*(r-1)) # Критическая константа-верхняя альфа квантиль
print(f"{alpha*100}%-я критическая область: X^2 > {round(C, 3)}")

if (X_stat > C): # то есть если попадаем в крит область, то нулевая гипотеза отвергается
    print("Нулевая гипотеза отвергается")
else:
    print("Нулевая гипотеза принимается")

p_value = 1 - chi2.cdf(X_stat, (s-1)*(r-1))
if p_value < 0.00001:
    print(f"Критический уровень значимости: p_value < 0.00001")
else:
    print(f"Критический уровень значимости: {p_value}")

if (p_value > alpha):
    print("Отклонение от нулевой гипотезы не значимо, принимается нулевая гипотеза")
else:
    print("Отклонение от нулевой гипотезы значимо, принимается альтернатива")
