import numpy as np
import matplotlib.pyplot as plt
from time import process_time


start_time = process_time()

# инициализация параметров
mu = 0.25
lam = 0.33
t_start = 0
t_finish = 10
N = 7 # кол-во аппаратов
epsilon = 0.0000001

d = 5 # метод Рунге-Кутта-Флеминга обеспечивает 5-ый порядок точности
rtol = 10 ** (-12) # относительная погрешность
atol = 10 ** (-24) # абсолютная погрешность
tol = 0 # локальная точность

p = np.zeros((N + 1, 1)) # массив вероятностей системы состояний: число аппаратов + нулевое состояние
p[0][0] = 1 # начальные условия
time = np.array([t_start]) # массив временных точек

# система дифференциальных уравнений Колмогорова
def sys_dif_eq(t, p):
    p_der = np.array([-lam * p[0] + mu * p[1]])
    for i in range(1, N):
        p_der = np.append(p_der, lam * p[i - 1] - (lam + i * mu) * p[i] + (i + 1) * mu * p[i + 1])
    p_der = np.append(p_der, lam * p[N - 1] - N * mu * p[N])
    return p_der

# метод Рунге-Кутта-Флеминга
def Runge_Kutta_Felberg(f, p, t, step):
    K = np.zeros((6, N + 1))
    result = np.zeros(N + 1)
    check = 0

    K[0] = step * f(t, p)
    K[1] = step * f(t + (1/4) * step, p + (1/4) * K[0])
    K[2] = step * f(t + (3/8) * step, p + (3/32) * K[0] + (9/32) * K[1])
    K[3] = step * f(t + (12/13) * step, p + (1932/2197) * K[0] - (7200/2197) * K[1] + (7296/2197) * K[2])
    K[4] = step * f(t + step, p + (439/216) * K[0] - 8 * K[1] + (3680/513) * K[2] - (845/4104) * K[3])
    K[5] = step * f(t + (1/2) * step,
                    p - (8/27) * K[0] + 2 * K[1] - (3544/2565) * K[2] + (1859/4104) * K[3] - (11/40) * K[4])

    result = ((16/135) * K[0] + (6656/12825) * K[2] + (28561/56430) * K[3] - (9/50) * K[4] + (2/55) * K[5]) + p

    # проверка условия равенства единице суммы вероятностей в один момент времени
    check = sum(result)
    if abs(check - 1) <= epsilon:
        return result
    else:
        print('No solution')
        quit(0)

'''____________________________________________________________'''

# Выбор начального шага
norm_p = np.sqrt(sum(list(map(lambda x: x ** 2, p[:, -1]))))
tol = rtol * norm_p + atol
norm_func = np.sqrt(sum(list(map(lambda x: x ** 2, sys_dif_eq(time[-1:], p[:, -1])))))
delta = ((1 / np.fabs(t_finish)) ** (d + 1)) + (norm_func ** (d + 1))
step_0 = (tol / delta) ** (1 / (d + 1))

    # один шаг методом Эйлера, т.к. начальные условия в особом положении (более половины компонент = 0)
time_0 = time[-1:] + step_0
p_0 = list(map(lambda x, y: x + step_0 * sys_dif_eq(time[-1:], p[:, -1])[y], p[:, -1], np.arange(N + 1)))

    # пересчет начального шага
norm_func = np.sqrt(sum(list(map(lambda x: x ** 2, sys_dif_eq(time[-1:], p_0)))))
delta = ((1 / np.fabs(t_finish)) ** (d + 1)) + (norm_func ** (d + 1))
step_00 = (tol / delta) ** (1 / (d + 1))
step_full = min(step_0, step_00) # ===> получили начальный шаг

'''____________________________________________________________'''

# Высчитаваем начальную погрешность
    # Приближение методом Рунге-Кутта-Флеминга за один шаг step_full
u1 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full)

    # Приближение методом Рунге-Кутта-Флеминга за два шага, каждый = step_full / 2
u_02 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full / 2)
u2 = Runge_Kutta_Felberg(sys_dif_eq, u_02, time[-1:] + step_full / 2, step_full / 2)

    # Оценка локальной погрешности
norm_r = np.sqrt(sum(list(map(lambda x, y: ((x - y) / (1 - 2 ** d)) ** 2, u2, u1))))

    # Локальная точность относительно u1
norm_p = np.sqrt(sum(list(map(lambda x: x ** 2, u1))))
tol = rtol * norm_p + atol

'''____________________________________________________________'''

# Алгоритм удвоения и деления шага пополам
while time[-1:] < t_finish:

    if norm_r > tol * (2 ** d):
        # ни u1, ни u2 не обеспечивают необходимую точность ===> повтор n-го шага с меньшей длинной
        step_full = step_full / 2
        u1 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full)
        u_02 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full / 2)
        u2 = Runge_Kutta_Felberg(sys_dif_eq, u_02, time[-1:] + step_full / 2, step_full / 2)
        norm_r = np.sqrt(sum(list(map(lambda x, y: ((x - y) / (1 - 2 ** d)) ** 2, u2, u1))))

        norm_p = np.sqrt(sum(list(map(lambda x: x ** 2, u1))))
        tol = rtol * norm_p + atol

    elif (norm_r > tol) and (norm_r <= (tol * (2 ** d))):
        # приближение u2 дает нужную точность ===> принимаем результаты сделанного шага, но большая вероятность того,
        #что новый шаг той же длины приведет к к слишком большой величине погрешности ===> меняем длину шага
        p = np.column_stack((p, u2))
        time = np.append(time, time[-1:] + step_full)

        step_full = step_full / 2
        u1 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full)
        u_02 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full / 2)
        u2 = Runge_Kutta_Felberg(sys_dif_eq, u_02, time[-1:] + step_full / 2, step_full / 2)
        norm_r = np.sqrt(sum(list(map(lambda x, y: ((x - y) / (1 - 2 ** d)) ** 2, u2, u1))))

        norm_p = np.sqrt(sum(list(map(lambda x: x ** 2, p[:, -1]))))
        tol = rtol * norm_p + atol

    elif (norm_r >= tol * 1 / (2 ** (d + 1))) and (norm_r <= tol):
        # приближение u2 дает нужную точность ===> принимаем сделанный шаг
        p = np.column_stack((p, u1))
        time = np.append(time, time[-1:] + step_full)

        # full_step не изменяется
        u1 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full)
        u_02 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full / 2)
        u2 = Runge_Kutta_Felberg(sys_dif_eq, u_02, time[-1:] + step_full / 2, step_full / 2)
        norm_r = np.sqrt(sum(list(map(lambda x, y: ((x - y) / (1 - 2 ** d)) ** 2, u2, u1))))

        norm_p = np.sqrt(sum(list(map(lambda x: x ** 2, p[:, -1]))))
        tol = rtol * norm_p + atol

    elif norm_r < tol * (1 / (2 ** (d + 1))):
        # приближение u2 слишком точное ===> принимаем сделанный шаг, но удваиваем step_full
        p = np.column_stack((p, u1))
        time = np.append(time, time[-1:] + step_full)

        step_full = 2 * step_full
        u1 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full)
        u_02 = Runge_Kutta_Felberg(sys_dif_eq, p[:, -1], time[-1:], step_full / 2)
        u2 = Runge_Kutta_Felberg(sys_dif_eq, u_02, time[-1:] + step_full / 2, step_full / 2)
        norm_r = np.sqrt(sum(list(map(lambda x, y: ((x - y) / (1 - 2 ** d)) ** 2, u2, u1))))

        norm_p = np.sqrt(sum(list(map(lambda x: x ** 2, p[:, -1]))))
        tol = rtol * norm_p + atol

print('process_time = ', process_time() - start_time, "seconds")


# вывод графика
fig = plt.figure(figsize=(12, 8))
fig.canvas.set_window_title('СМО с отказами')
ax = fig.add_subplot()

labels = np.arange(N + 1)
for i in range(N + 1):
    plt.plot(time, p[i], label = 'State ' + str(labels[i]))

ax.set_title('Решение методом Рунге-Кутта-Фельберга')
ax.set_xlabel('Время')
ax.set_ylabel('Вероятность')
plt.legend(loc = 'best')
plt.grid(True)

plt.show()












