# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy
import matplotlib.pyplot as plt

e = 1e-10

def func(i):
    return math.cos(i)

def half_divide_method(a, b, f):
    x = (a + b) / 2
    if math.fabs(f(x)) >= e:
        # print(x)
        # x = (a + b) / 2
        if f(a) * f(x) < 0:
            half_divide_method(a,x,f)
        if f(x) * f(b) < 0:
            half_divide_method(x,b,f)
    else:
        print("0 in ", (a + b) / 2)

x = numpy.arange(-2.0, 4.0, 0.1)
y = []
start = -2.0
end = 4.0
step = 0.1
for i in x:
    y.append(func(i))
half_divide_method(start, end, func)
plt.plot(x, y, c="black", label="Интерполируемая функция")
plt.show()

# print ('root of the equation half_divide_method %s' % half_divide_method(a1, b1, func_glob))
# print ('root of the equation newtons_method %s' % newtons_method(a1, b1, func_glob, func_first))

# бисекция и секущей
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
