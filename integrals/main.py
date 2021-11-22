import numpy as np
import matplotlib.pyplot as plt
import math


def func(x):
    return math.cos(x)
    # return -x * x
    # return 1/(1+(x*x))


# middle quads
def riemann_sum(f, a, b, N):
    dx = (b - a)/N
    x = np.linspace(a, b, N+1)
    x_mid = (x[:-1] + x[1:])/2
    sum = 0
    for i in x_mid:
        sum += f(i)
    return sum*dx


# trapezoidal rule (заменяем интервал на простейший многочлен)
def trap(f, a, b, n):
    g = 0
    if b > a:
        h = (b-a)/float(n)
    else:
        h = (a-b)/float(n)
    for i in range(0, n):
        k = 0.5 * h * (f(a + i*h) + f(a + (i+1)*h))
        g = g + k

    return g


def simpson(f, a, b, n):
    h = (b-a)/n
    k = 0.0
    x = a + h
    for i in range(1, int(n/2) + 1):
        k += 4*f(x)
        x += 2*h
    x = a + 2*h
    for i in range(1, int(n/2)):
        k += 2*f(x)
        x += 2*h

    return (h/3)*(f(a)+f(b)+k)


a = -1
b = 7
e = 1e-5
# automatic find optimal n
funcArray = [riemann_sum, trap, simpson]
nameArray = ["quad", "trapezoidal", "simpson"]
nArray = 0

for f in funcArray:
    n = 4
    resLast = f(func, a, b, int(n/2))
    resCurr = f(func, a, b, n)
    while abs(resLast - resCurr) > e:
        n *= 2
        resLast = resCurr
        resCurr = f(func, a, b, n)
    print("For", nameArray[nArray], "rule:")
    print("n =", n, ", result:", resCurr)
    nArray += 1

x = np.arange(a, b, abs((b-a)/n))
x = np.append(x, b)
y = []
for i in x:
    y.append(func(i))
plt.plot(x, y, c="black")
plt.plot(x, np.zeros(len(x)))
plt.show()
