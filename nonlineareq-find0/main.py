import math
import numpy
import matplotlib.pyplot as plt

e = 1e-10

# бисекция и секущей
# ищет только 1 ноль на отрезке, больше (?)


def func(arg):
    return math.cos(arg)


def half_divide_method(a, b, f):
    x = (a + b) / 2
    it = 0
    while math.fabs(f(x)) >= e:
        x = (a + b) / 2
        a, b = (a, x) if f(a) * f(x) < 0 else (x, b)
        it += 1
    print("0 in ", x, "find in ", it, "iteration")


def secant(f, x0, x1, tol=e, max_iterations=100):
    # (!) problem with sign, if segment by x includes 0
    fx0 = f(x0)
    fx1 = f(x1)

    while abs(fx1) >= tol and max_iterations != 0:
        x2 = (x0 * fx1 - x1 * fx0) / (fx1 - fx0)
        x0, x1 = x1, x2
        fx0, fx1 = fx1, f(x2)
        max_iterations -= 1
    print("0 in ", x1, "find in ", 100 - max_iterations, "iteration")


start = 2.0
end = 6.0
step = 0.1
x = numpy.arange(start, end, step)
y = []
for i in x:
    y.append(func(i))
half_divide_method(start, end, func)
secant(func, start, end)
plt.plot(x, y, c="black")
plt.show()
