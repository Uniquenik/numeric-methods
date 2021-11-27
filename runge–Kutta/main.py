import math
import numpy as np
import matplotlib.pyplot as plt


def func(x, y):
    # return x*x + math.cos(y)
    # return y*y - x  # (?)
    # return math.log(x) - y
    return math.cos(x) + math.sin(y)


def runge_kutta(f, y0, x0, N, x_end):
    h = (x_end - x0) / N
    x = np.zeros((N,))
    y = np.zeros((N,))
    y[0] = y0
    x[0] = x0

    for k in range(N-1):
        k1 = f(x[k], y[k])
        k2 = f(x[k] + 0.5 * h, y[k] + 0.5 * h * k1)
        k3 = f(x[k] + 0.5 * h, y[k] + 0.5 * h * k2)
        k4 = f(x[k] + h, y[k] + h * k3)
        y[k + 1] = y[k] + (1 / 6) * h * (k1 + 2 * k2 + 2 * k3 + k4)
        x[k+1] = x[k] + h

    return x, y


e = 1e-5
y0 = 0.0
x0 = 1.0
x1 = 6.0
plt.grid()
h = 4
prevRes = runge_kutta(func, y0, x0, h, x1)
plt.plot(prevRes[0], prevRes[1], c=np.random.rand(3,), label=h)
h *= 2
currentRes = runge_kutta(func, y0, x0, h, x1)
plt.plot(currentRes[0], currentRes[1], c=np.random.rand(3,), label=h)
# without diff doesn't work
diff = currentRes[1]
diff2 = prevRes[1]

# increase number of partitions while error is greater than epsilon
while np.linalg.norm(diff[::2] - diff2) > e:
    print(np.linalg.norm(diff[::2] - diff2))
    h *= 2
    prevRes = currentRes
    currentRes = runge_kutta(func, y0, x0, h, x1)
    plt.plot(currentRes[0], currentRes[1], c=np.random.rand(3,), label=h)
    diff = currentRes[1]
    diff2 = prevRes[1]

plt.legend(loc='upper left', frameon=False)
plt.show()


