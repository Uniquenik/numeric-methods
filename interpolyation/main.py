import numpy as np
import matplotlib.pyplot as plt
import math


def lagrange_interpolation(x, y, xx, n):
    sum = 0
    for i in range(n):
        product = y[i]
        for j in range(n):
            if i != j:
                product = product*(xx - x[j])/(x[i]-x[j])
        sum = sum + product
    return sum

start = 2
length = 7
diff = 1
newDots = 7
stop = start+(length*(newDots-1))+length
x2 = np.arange(start, stop, diff)
x1 = x2[::newDots+1]
y11 = []
for i in x2:
    y11.append(math.log(i))
y1 = y11[::newDots+1]
y2, y3, y4 = [], [], []

for i in x2:
    if i in x1:
        index, = np.where(x1 == i)
        y2.append(y1[index.tolist()[0]])
        y3.append(y1[index.tolist()[0]])
        y4.append(y1[index.tolist()[0]])
    else:
        y2.append(lagrange_interpolation(x1, y1, i, len(x1)))
        y3.append(lagrange_interpolation(x1, y1, i, len(x1)-1))
        y4.append(lagrange_interpolation(x1, y1, i, len(x1)-2))

plt.grid()
print(x1, y1)
print(x2, y2)
print("Inaccuracy: ", np.linalg.norm(np.array(y2) - np.array(y11)))
plt.plot(x1, y1, x2, y2)
#, x2, y3, x2, y4
plt.show()