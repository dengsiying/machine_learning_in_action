import  numpy as np
import  matplotlib.pyplot as plt
import operator
from os import listdir

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

testVector = img2vector('testDigits/0_13.txt')

def classify0(inX,dataSet,labels,k):#在距离最近的k个点中选择类别数最多的那个类别
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#行方向上相加 返回1000*
    distances = sqDistances**0.5
    sortedDisIndicies = np.argsort(distances)#np.argsort函数返回的是数组值从小到大的索引值
    classCount = {}#字典 类别：个数
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        # 字典的get（） 第二个参数default -- 可选参数，如果指定键的值不存在时，返回该值，默认为 None。第二个参数为0，即指定的voteIlabel键值不存在时，返回0值
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1 #相当于类别计数器
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#sorted 从小到大 reverse=True 则从大到小
    return sortedClassCount[0][0]

def handwritingClassTest():
    hwLabels = [] #存放向量对应的真实数字
    trainingFileList = listdir('trainingDigits')#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNamestr = trainingFileList[i]
        fileStr = fileNamestr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNamestr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNamestr = testFileList[i]
        fileStr = fileNamestr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'%fileNamestr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('tne classifier came back with :%d,the real answer is :%d'%(classifierResult,classNumStr))
        if (classifierResult!=classNumStr):
            errorCount += 1.0
        print('the total number of errors is :%d'%errorCount)
        print('the total error rate is:%f'%(errorCount/float(mTest)))

handwritingClassTest()
















