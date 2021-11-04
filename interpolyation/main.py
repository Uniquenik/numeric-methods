import numpy as np
import matplotlib.pyplot as plt
import math

def cubic_interp1d(x0, x, y):
    x = np.asfarray(x)
    y = np.asfarray(y)

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # allocate buffer matrices
    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = math.sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0 # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = math.sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = math.sqrt(2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0 # natural boundary
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    # solve [L.T][x] = [y]
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    # find index
    index = x.searchsorted(x0)
    # np.clip(index, 1, size-1, index)

    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)
    return f0


def lagrange_interpolation(x, y, xx, n):
    sum = 0
    # lagrange sum
    for i in range(n):
        product = y[i]
        # basic polynomial
        for j in range(n):
            if i != j:
                product = product * (xx - x[j]) / (x[i] - x[j])
        sum = sum + product
    return sum


start = 2
dots = 7
diff_bt_alldots = 0.5
new_dots = 10
stop = start + ((dots - 1) * new_dots * diff_bt_alldots + dots * diff_bt_alldots)
x2 = np.arange(start, stop, diff_bt_alldots)
x1 = x2[::new_dots + 1]
y11 = []
for i in x2:
    y11.append(math.log(i) ** 0.5)
y1 = y11[::new_dots + 1]
y2, y3, y4 = [], [], []

for i in x2:
    if i in x1:
        # search index by element (some problem in np)
        index, = np.where(x1 == i)
        y2.append(y1[index.tolist()[0]])
        y3.append(y1[index.tolist()[0]])
        #y4.append(y1[index.tolist()[0]])
    else:
        y2.append(lagrange_interpolation(x1, y1, i, len(x1)))
        y3.append(cubic_interp1d(i, x1, y1))
        # a = cubic_interp1d(i, x1, y1)
        #y4.append(lagrange_interpolation(x1, y1, i, len(x1) - 2))

plt.grid()
print(x1, y1)
print(x2, y2)
print("Inaccuracy Lagrange: ", np.linalg.norm(np.array(y2) - np.array(y11)))
print("Inaccuracy Spline: ", np.linalg.norm(np.array(y3) - np.array(y11)))
plt.plot(x1, y1, c="black", label="Интерполируемая функция")
plt.plot(x2, y2, 'k--', c="green", label="Лагранж")
plt.plot(x2, y3, 'k--', c="orange", label="Сплайны")
plt.plot(x2, y11,  c="blue", label="Исходная")
plt.legend(loc='upper left', frameon=False)
# , x2, y3, x2, y4
plt.show()
