import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_1 = pd.read_excel('var10_Z2.xls', sheet_name='Sheet2', usecols='R')
data_1 = data_1[data_1['Z14,15,16_x'].notna()]
data_1_arr = list(data_1['Z14,15,16_x'])

data_2 = pd.read_excel('var10_Z2.xls', sheet_name='Sheet2', usecols='S')
data_2 = data_2[data_2['Z14,15,16_y'].notna()]
data_2_arr = list(data_2['Z14,15,16_y'])

# X через Y
Y = 82

# Станд. отклонение
std1 = np.std(data_1_arr, axis=0)
std2 = np.std(data_2_arr, axis=0)

mean1 = np.mean(data_1_arr)
mean2 = np.mean(data_2_arr)

cov = np.mean((data_1_arr-mean1)*(data_2_arr-mean2))

corr = cov / (std1 * std2)
print(f"Коэффициент корреляции: {corr}")

betta = corr*std1/std2
print(f"Коэффициент линейной регрессии: {betta}")

print(f"Уравнение регрессии X на Y: x*(y) = {betta} * y + {betta*(-mean2) + mean1}")

prediction_X = betta * (Y - mean2) + mean1
print(f"Прогноз при у = 82: {prediction_X}")

print(f"Стандартное отклонение наблюдений прочности: S_x = {round(std1, 3)}")


def err():
    res = 0
    for i in range(len(data_1)):
        res += (data_1_arr[i]-data_2_arr[i])**2
    return np.sqrt(res/len(data_1))

print(f"Ошибка прогноза: {err()}")
#Построение эллипса
x_min = mean1 - 2 * std1
x_max = mean1 + 2 * std1

# Шаг изменения x
step = 0.0001

# Создать массив значений x
x = np.arange(x_min, x_max, step)
y = np.arange(70, 100, step)

y1 = mean2 + corr * std2 * (x - mean1) / std1 - std2 * np.sqrt(1 - corr ** 2) * np.sqrt(4 * (std1 ** 2) - (x - mean1)**2) / std1
y2 = mean2 + corr * std2 * (x - mean1) / std1 + std2 * np.sqrt(1 - corr ** 2) * np.sqrt(4 * (std1 ** 2) - (x - mean1)**2) / std1
x_reg = corr*std1/std2 * (y - mean2) + mean1

r = 3
s = 4
X_1 = 116.05
X_r = 124.05
Y_1 = 81.05
Y_s = 87.05
bins_X = np.linspace(X_1, X_1 + ((X_r - X_1)/r) * r, r-1) # Разбиение чиловой прямой на r групп
bins_Y = np.linspace(Y_1, Y_1 + ((Y_s - Y_1)/s) * s, s-1) # Разбиение чиловой прямой на s групп

fig, ax = plt.subplots()
for i in range(r-1):
    ax.vlines(bins_X[i], 70, 100, color="Gray", linewidth=0.5, linestyle='dashed')
for i in range(s-1):
    ax.hlines(bins_Y[i], 111, 129, color="Gray", linewidth=0.5, linestyle='dashed')

ax.hlines(Y, 111, prediction_X, color="Blue", linewidth=0.7, linestyles='dashed')
ax.vlines(prediction_X, 70, Y, color="Blue", linewidth=0.7, linestyles='dashed')
plt.plot(x, y1, color="Green", linewidth=0.5)
plt.plot(x, y2, color="Green", linewidth=0.5)
ax.plot(x_reg, y, color="Red", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Поле данных, линия регрессии х на у, эллипс рассеяния")
plt.show()