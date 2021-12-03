import math
import numpy
import matplotlib.pyplot as plt

e = 1e-10

# бисекция и секущей


def func(arg):
    # for more nulls
    # return math.sin(arg)
    # return (arg-1)*(arg-2)*(arg-3)
    # return math.log(arg) #0.5-4
    return math.exp(arg) + arg


# for one point
def half_divide_method(a, b, f):
    x = (a + b) / 2
    it = 0
    while math.fabs(f(x)) >= e and f(a)*f(x) > 0 or f(b)*f(x) > 0:
        x = (a + b) / 2
        a, b = (a, x) if f(a) * f(x) < 0 else (x, b)
        it += 1
    if f(a) * f(x) < 0 and f(b) * f(x) < 0:
        print("So many nulls")
        return 0
    else:
        print("0 in ", x, "find in ", it, "iteration")
        return x


# for one point
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
    return x1


# for more points (some problem, do not use in this form)
def divide_method(a, b, f, it):
    x = (a + b) / 2
    if math.fabs(f(x)) >= e:
        if math.fabs(a-b) > e:
            divide_method(a, x, f, it+1)
            divide_method(x, b, f, it+1)
    else:
        print("0 in ", x, it)


start = -2.0  # start x
end = 4.0  # end x (for more points type 25)
step = 0.1
x = numpy.arange(start, end, step)
y = []
for i in x:
    y.append(func(i))
result1 = half_divide_method(start, end, func)
result2 = secant(func, start, end)
# print("Diff: ", abs(result1 - result2))
print("")
e = 1e-5  # change epsilon, because calculations so long...
divide_method(start, end, func,0)
plt.plot(x, y, c="black")
plt.plot(x, numpy.zeros(len(x)))
plt.show()


# c.99 численное интегирование Прямоугольники, трапеции и симпсона
# (дробление шага)