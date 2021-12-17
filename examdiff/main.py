import matplotlib.pyplot as plt
import numpy as np
import math

plt.figure(figsize=(7, 7))
plt.grid()
T = 2.0
partsT = 1000
drawLineCount = 10
deltaT = T / (partsT - 1)
partsX = math.floor(math.pi / 2 / math.sqrt(2 * deltaT) + 1)
deltaX = math.pi / 2 / (partsX - 1)


matrix = np.zeros((partsT, partsX))

for xIdx in range(partsX):
    x = xIdx * deltaX
    matrix[0, xIdx] = math.sin(x)

for tIdx in range(partsT):
    t = tIdx * deltaT
    matrix[tIdx, 0] = 0.0
    matrix[tIdx, partsX - 1] = math.exp(-t)

courantValue = deltaT / deltaX / deltaX

for tIdx in range(1, partsT):
    for xIdx in range(1, partsX - 1):
        value = courantValue * matrix[tIdx - 1, xIdx + 1]
        value += (1 - 2 * courantValue) * matrix[tIdx - 1, xIdx]
        value += courantValue * matrix[tIdx - 1, xIdx - 1]
        matrix[tIdx, xIdx] = value


xValues = np.linspace(0.0, math.pi / 2, partsX)#.toList()


answerU = np.zeros((partsT, partsX))
for i in range(partsT):
    for j in range(partsX):
        answerU[i][j] = math.exp(-i*deltaT)*math.sin(j*deltaX)
print(answerU)
dataX = xValues

data = {}
for tIdx in range(0, partsT, int(partsT / drawLineCount)):
    data[tIdx] = matrix[tIdx]
    data["answer"+str(tIdx)] = answerU[tIdx]

    #var p = letsPlot(data)

for tIdx in range(400, 500, int(partsT / drawLineCount)):
    plt.plot(dataX, data[tIdx], c=np.random.rand(3, ), label=tIdx)
    plt.plot(dataX, data[tIdx], c=np.random.rand(3, ), label="answer"+str(tIdx))

maxError = 0.0
maxErrorT = 0.0
for tIdx in range(partsT):
    arr = np.stack((matrix[tIdx], answerU[tIdx]), axis=1)
    curError = max([abs(i[0]-i[1]) for i in arr])
    if maxError < curError:
        maxError = curError
        maxErrorT = tIdx*deltaT

print("Max inaccurance: ", maxError)
print("Max inaccurance time: ", maxErrorT)
plt.legend(loc='upper left', frameon=False)
plt.show()
