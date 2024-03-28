import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

start = time.time()

# Константы
ALPHA = 0.001
DELTA = 0.00000001
POINTS_SIZE = 5
GOLDEN_RATIO = scipy.constants.golden_ratio
METHOD = "fib"
START_X = -2
START_Y = 0


# Функция
def fun(x, y):
    return x ** 2 + y ** 2


# Координата x градиента
def grad_x(x, y):
    return 2 * x


# Координата y градиента
def grad_y(x, y):
    return 2 * y


def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def add_point(x, y):
    x_points.append(x)
    y_points.append(y)
    z_points.append(fun(x, y))
    s.append(POINTS_SIZE)


def add_fib_point(x, y):
    x_fib_points.append(x)
    y_fib_points.append(y)
    z_fib_points.append(fun(x, y))
    s_fib.append(POINTS_SIZE)


# Метод золотого сечения
def fib_search(a_x, a_y, b_x, b_y):
    global f_count
    p1_x = b_x - (b_x - a_x) / GOLDEN_RATIO
    p1_y = b_y - (b_y - a_y) / GOLDEN_RATIO
    p2_x = a_x + (b_x - a_x) / GOLDEN_RATIO
    p2_y = a_y + (b_y - a_y) / GOLDEN_RATIO
    add_fib_point(p1_x, p1_y)
    add_fib_point(p2_x, p2_y)
    f_count += 2
    y1 = fun(p1_x, p1_y)
    y2 = fun(p2_x, p2_y)
    while distance(a_x, a_y, b_x, b_y) > DELTA:
        if y1 >= y2:
            a_x = p1_x
            a_y = p1_y
            p1_x = p2_x
            p1_y = p2_y
            p2_x = a_x + (b_x - a_x) / GOLDEN_RATIO
            p2_y = a_y + (b_y - a_y) / GOLDEN_RATIO
            add_fib_point(p2_x, p2_y)
            y1 = y2
            y2 = fun(p2_x, p2_y)
        else:
            b_x = p2_x
            b_y = p2_y
            p2_x = p1_x
            p2_y = p1_y
            p1_x = b_x - (b_x - a_x) / GOLDEN_RATIO
            p1_y = b_y - (b_y - a_y) / GOLDEN_RATIO
            add_fib_point(p2_x, p2_y)
            y2 = y1
            y1 = fun(p1_x, p1_y)
        f_count += 1
    return (a_x + b_x) / 2, (a_y + b_y) / 2

# Градиентный спуск
def gradient_descent(method):
    # начальная точка
    x = START_X
    y = START_Y
    global grad_count
    global count
    while len(z_points) < 1 or distance(x_points[-1], y_points[-1], x, y) > DELTA:
        add_point(x, y)
        x_grad = grad_x(x, y)
        y_grad = grad_y(x, y)
        grad_count += 2
        if method == "const":
            x -= ALPHA * x_grad
            y -= ALPHA * y_grad
        elif method == "fib":
            x, y = fib_search(x, y, x - x_grad, y - y_grad)
        count += 1
        # print("{:.6f} {:.6f}".format(x, y))


# Поиск экстремума
x_points = []
y_points = []
z_points = []
x_fib_points = []
y_fib_points = []
z_fib_points = []
s = []  # размер точек
s_fib = []
count = 0  # число итераций
f_count = 0  # количество вычислений значений функции
grad_count = 0  # количество вычислений градиента

# Оптимизация
gradient_descent(METHOD)
result = scipy.optimize.minimize(fun, START_X, START_Y, method="nelder-mead")

# Построение графика
x_range = np.arange(-2, 2, 0.1)
y_range = np.arange(-2, 2, 0.1)
x_grid, y_grid = np.meshgrid(x_range, y_range)
z_grid = fun(x_grid, y_grid)

# Вывод графика и точек
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x_grid, y_grid, z_grid, alpha=0.2)
ax.scatter(x_points, y_points, z_points, s=s, c="#FF0000")
ax.scatter(x_fib_points, y_fib_points, z_fib_points, s=s_fib, c="#00FF00")
ax.view_init(40, -20)
plt.show()
fig = plt.figure()
# Вывод линий уровня
ax1 = fig.add_subplot(111)
plt.contour(x_grid, y_grid, z_grid, levels=np.arange(-100, 100, 1))
ax1.scatter(x_points, y_points, s=s, c="#FF0000")
ax1.scatter(x_fib_points, y_fib_points, s=s_fib, c="#00FF00")
plt.show()

# Вывод результатов эксперимента
print("РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ ЭКСПЕРИМЕНТА")
print()
print("ГРАДИЕНТНЫЙ СПУСК:")
print("Число итераций:", len(s))
print("Количество вычислений значений функции:", f_count)
print("Количество вычислений градиента:", grad_count)
print("Реальное значение: (0, 0)")
print("Найденное значение:")
print("x =", x_points[-1], "y =", y_points[-1])
print("Время работы:", time.time() - start, "сек")
print("МЕТОД НЕЛДЕРА-МИДА")
print()
print("Количество вычислений значений функции:", result['nfev'])
print("Найденное значение:")
print("x =", result["final_simplex"][0][0][0], "y =", result["final_simplex"][0][1][0])
