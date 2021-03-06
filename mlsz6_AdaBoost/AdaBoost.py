import numpy as np
import  matplotlib.pylab as plt
import pandas as pd
def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                        [1.5, 1.6],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

# dataArr,classLabels = loadSimpData()
#
# df = pd.DataFrame(dataArr,columns=['X1','X2'])
# df['class'] = classLabels
# plt.scatter(df['X1'],df['X2'],c=df['class'],s=40,cmap=plt.cm.Spectral);
#plt.show()


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
        单层决策树分类函数
        Parameters:
            dataMatrix - 数据矩阵
            dimen - 第dimen列，也就是第几个特征
            threshVal - 阈值
            threshIneq - 标志
        Returns:
            retArray - 分类结果
        """
    retArray = np.ones((dataMatrix.shape[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray



def buildStump(dataArr,classLabels,D):
    """
        找到数据集上最佳的单层决策树
        Parameters:
            dataArr - 数据矩阵
            classLabels - 数据标签
            D - 样本权重
        Returns:
            bestStump - 最佳单层决策树信息
            minError - 最小误差
            bestClasEst - 最佳的分类结果
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf')  #最小误差初始为正无穷大
    for i in range(n): #遍历所有特征
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps #计算步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']: #大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin+float(j)*stepSize) #计算阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) #计算分类结果
                errArr = np.mat(np.ones((m,1))) #初始化误差矩阵
                errArr[predictedVals==labelMat] = 0 #分类正确的,赋值为0
                weightedError = D.T*errArr #计算误差
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt = 40):
    weakClassArr = []
    m = dataArr.shape[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #print('D:',D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16))) #计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        bestStump['alpha'] = alpha  #存储弱学习算法权重
        weakClassArr.append(bestStump) #存储单层决策树
        #print('classEst',classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha*classEst
        #print('aggClassEst:',aggClassEst.T)#乘以分类器权重alpha的Class
        aggErrors  = np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        #print('aggErrors',aggErrors)
        errorRate = aggErrors.sum()/m
        #print('total error',errorRate)
        if errorRate == 0.0:break
    return weakClassArr,aggClassEst

# if __name__ == '__main__':
#     dataArr,classLabels = loadSimpData()
#     weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
#     print('weakClassArr',weakClassArr)
#     print('aggClassEst',aggClassEst)

def adaClassify(datToClass,classifierArr):
    """
    AdaBoost分类函数
    Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
    Returns:
        分类结果
    """
    dataMatrix = np.mat(datToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)
if __name__ == '__main__':
    dataArr,classLabels = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    print(adaClassify([[0,0],[5,5]], weakClassArr))











