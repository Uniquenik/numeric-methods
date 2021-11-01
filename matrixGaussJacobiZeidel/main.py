import numpy as np

ITER_LIMIT = 1000

# def Gauss(a, n, x):
#     for i in range(n):
#         maxvalue = 0
#         maxi = i
#         for i1 in range(i, n):
#             if abs(a[i1][i]) > maxvalue:
#                 maxvalue = abs(a[i1][i])
#                 maxi = i1
#         a[[i, maxi]] = a[[maxi, i]]
#
#         for j in range(i + 1, n):
#             r = a[j][i] / a[i][i]
#             for k in range(n + 1):
#                 a[j][k] = a[j][k] - r * a[i][k]
#         #print(a)
#
#     x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]
#
#     for i in range(n - 2, -1, -1):
#         x[i] = a[i][n]
#         for j in range(i + 1, n):
#             x[i] = x[i] - a[i][j] * x[j]
#         x[i] = x[i] / a[i][i]
#     # print("Gauss: ", x)
#     return x

def Jacobi(a, x1):
    x1 = x1.transpose()[0]
    x = np.zeros_like(x1)

    for it in range(ITER_LIMIT):
        x_new = np.zeros_like(x)
        for i in range(a.shape[0]):
            s1 = np.dot(a[i, :i], x[:i])
            s2 = np.dot(a[i, i+1:], x[i+1:])
            x_new[i] = (x1[i] - s1 - s2) / a[i, i]
        diff = np.linalg.norm(x_new - x)
        if abs(diff) < abs(1e-10):
            print("Jacobi: ", it, "iteration")
            break
        x = x_new
    return x_new


def Zeidel(a, x1):
    x = np.zeros_like(x1)
    n = len(a)
    for iter in range(ITER_LIMIT):
        for i in range(n):
            s1 = 0
            temp = x.copy()
            for j in range(n):
                if i != j:
                    s1 += x[j] * a[i][j]
            x[i] = (b[i] - s1) / a[i][i]
        gap = max(abs(x - temp))

        if gap < 1e-10:
            print("Zeidel: ", iter, "iteration")
            break
    return x


a1 = int(input("Size:"))
x1 = np.zeros(a1)
matrixA = np.random.rand(a1, a1+1)
for i in range(a1):
    matrixA[i][i] *= 1000

invA = np.linalg.inv(matrixA[:, :-1])
b = matrixA[:, -1:]
result = invA.dot(b)
print("Equ:", np.transpose(result))

m = np.array(matrixA[:, :-1])
x = np.array(matrixA[:, -1:])

m1 = np.array(matrixA[:, :-1])
x1 = np.array(matrixA[:, -1:])

# resultGauss = Gauss(matrixA, a1, x1)
resultJacobi = Jacobi(m, x)
resultZeidel = Zeidel(m1, x1).transpose()

# print("Gauss", resultGauss)
# print("Jacobi result:", resultJacobi)
print("Jacobi accuracy: ", np.linalg.norm(np.transpose(result) - resultJacobi, ord=1))

# print("Zeidel result:", resultZeidel)
print("Zeidel accuracy: ", np.linalg.norm(np.transpose(result) - resultZeidel, ord=1))
