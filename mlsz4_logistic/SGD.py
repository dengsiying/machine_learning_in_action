import numpy as np
import random
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = dataMatrix.shape
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
