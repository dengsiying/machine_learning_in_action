import re
import bayes
import numpy as np
import random

"""
函数说明:接收一个大字符串并将其解析为字符串列表

"""
def textParse(bigString):
    listOfTokens = re.split(r'\W*',bigString) # \W 匹配任何非单词字符  * 匹配前面的子表达式零次或多次。
    return [tok.lower() for tok in listOfTokens if len(tok)>2]#除了单个字母，例如大写的I，其它单词变成小写



"""
函数说明:根据vocabList词汇表，构建词袋模型

Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词袋模型

"""

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    trainingSet = list(range(50));testSet = [] # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0,len(trainingSet)))#从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = bayes.trainNBO(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.setOfWords2Vec(vocabList,docList[docIndex])
        if bayes.classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print('wrong testSet:',docList[docIndex])
    print('wrong rate:%.2f%%'%(float(errorCount)/len(testSet)*100))

spamTest()
