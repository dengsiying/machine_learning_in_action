

from functools import reduce
import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]      #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec
"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型

"""
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:print('this word:%s is not in my Vocabulary!'%word)
    return returnVec

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表

"""
def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet|set(document) #两个set之间取并集
    return list(vocabSet)


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 非侮辱类的条件概率数组
    p1Vect - 侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率

"""


def trainNBO(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #计算训练的文档数目 6
    numWords = len(trainMatrix[0]) #计算每篇文档的词条数 32
    pAbusive = sum(trainCategory)/float(numTrainDocs) #文档属于侮辱类的概率 3/6=0.5
    #p0Num = np.zeros(numWords)
    #p1Num = np.zeros(numWords)
    #p0Denom = 0.0
    #p1Denom = 0.0
    # 拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):#统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)··
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)··
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            #取对数 防止下
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return  p0Vect,p1Vect,pAbusive

postingList ,classVec = loadDataSet()
print('postingList:',postingList)
myVocabList = createVocabList(postingList)
print('myVocabList:',myVocabList)
trainMat = []
for postinDoc in postingList:
    trainMat.setOfWappend(ords2Vec(myVocabList,postinDoc))
print('trainMat:',trainMat)
p0V,p1V,pAb = trainNBO(trainMat,classVec)
#p0V存放的是每个单词属于类别0，也就是非侮辱类词汇的概率 p1V存放的就是各个单词属于侮辱类的条件概率。pAb就是先验概率。
print('p0V:',p0V)
print('p1V:',p1V)
print('classVec',classVec)
print('pAb:',pAb)



"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 侮辱类的条件概率数组
    p1Vec -非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类

"""
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) +np.log(pClass1)  #log p(w1|1)+log p(w2|1) +... = log  p(w1|1)* p(w2|1)*...
    p0 = sum(vec2Classify*p0Vec) +np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNBO(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    p1 = sum(thisDoc * p1V) + np.log(pAb)
    print(p1)
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'1')
    else:
        print(testEntry,'0')
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '1')
    else:
        print(testEntry, '0')

