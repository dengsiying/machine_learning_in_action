import  numpy as np
import  matplotlib.pyplot as plt
import pandas as pd
import operator
import seaborn as sns
#sns.set(context='notebook',style='whitegrid')

def  file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index =0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
data = pd.DataFrame(datingDataMat,columns=['Flydistance','Gametime','Icecream'])
data['class'] = datingLabels

#sns.lmplot('Gametime','Icecream',hue='class',data=data.loc[:,['Gametime','Icecream','class']],size=6,fit_reg=False,scatter_kws={'s':30})

largeDoses = data[data['class'].isin([3])]
smallDoses = data[data['class'].isin([2])]
didntLike = data[data['class'].isin([1])]
fig,ax = plt.subplots(figsize=(10,6))
ax.scatter(largeDoses['Flydistance'],largeDoses['Gametime'],c='g',marker='o',label='largeDoses')
ax.scatter(smallDoses['Flydistance'],smallDoses['Gametime'],c='b',marker='x',label='smallDoses')
ax.scatter(didntLike['Flydistance'],didntLike['Gametime'],c='r',marker='*',label='didntLike')
ax.legend()
ax.set_xlabel('Flydistance')
ax.set_ylabel('Gametime')
#plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)#参数0使得函数从每列中取得最小值，而不是行。返回1*3
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = np.zeros(data.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet-np.tile(minVals,(m,1))#把minVals复制扩充为m行1列的数组
    normDataSet = normDataSet/np.tile(ranges,(m,1))#特征值相除
    return normDataSet,ranges,minVals

normMat,ranges,minVals = autoNorm(datingDataMat)


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


def datingClassTest():
    hoRatio = 0.05
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
       classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
       print('the classifier came back with: %d,the real answer is : %d' %(classifierResult,datingLabels[i]))
       if(classifierResult !=datingLabels[i]):
           errorCount +=1.0
    print('the total error rate is: %f'%(errorCount/float(numTestVecs)))

#datingClassTest()
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    precentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')# 训练集
    normMat, ranges, minVals = autoNorm(datingDataMat)#对训练集数据进行预处理
    inArr = np.array([ffMiles, precentTats, iceCream])
    classfierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print('You will probably like this person :',resultList[classfierResult-1])


classifyPerson()































