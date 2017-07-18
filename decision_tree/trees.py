'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator
#创建数据集，输出数据集形式：[数据+标签]
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

#计算数据集的香农熵，返回结果越大，数据集越无序
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  #数据集样本数
    labelCounts = {}
    #针对每个标签计算该标签样本的个数
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]   #获取标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #对所有类别计算信息后累加
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2  #公式计算
    return shannonEnt
 
 #根据数据集中的某一个属性划分数据集，例如输入（dataset,0,1）则返回数据集中第一个属性的值为1的数据样本
def splitDataSet(dataSet, axis, value):
    retDataSet = [] 
    for featVec in dataSet:  #对于每个数据样本
        if featVec[axis] == value:   #如果对应属性等于对应值
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:]) #去掉该属性值，然后加入列表中
            retDataSet.append(reducedFeatVec)
    return retDataSet  

#选择最好的数据集划分方式，即调用一个返回该数据集使用哪个特征进行划分最好（用该特征划分后，信息熵变化越明显）   
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #返回数据集的特征数  #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)  #计算初始数据集的信息熵
    bestInfoGain = 0.0; bestFeature = -1   #初始化
    for i in range(numFeatures):       #遍历每个属性（特征）     #iterate over all the features
        featList = [example[i] for example in dataSet]   #返回一个列表，该列表包含了数据集中i特征那一列的所有值     #create a list of all the examples of this feature
        uniqueVals = set(featList)      #set语句去重，获取第i个属性所有的可能值   #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:  #对于第i个属性的所有可能取值
            subDataSet = splitDataSet(dataSet, i, value)   #根据 该属性=其所有可能值   都划分一遍数据集（根据属性值，将原始数据集划分为几块）
            prob = len(subDataSet)/float(len(dataSet))     #计算该取值下的数据集概率（每一块取值概率）
            newEntropy += prob * calcShannonEnt(subDataSet)   #计算对应划分的数据集下的信息熵，然后相加，即获得了以这个属性划分完数据集后的总信息增益   
        infoGain = baseEntropy - newEntropy      #划分前后的信息增益差，越大说明增益降低越明显，越好   #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i            #信息增益差最大的划分方式对应的属性
    return bestFeature                      #returns an integer

#若最后还没有统一结果，投票方式决定类别
def majorityCnt(classList):     
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#创建决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  #所有类标签列表
    if classList.count(classList[0]) == len(classList):     #所有类标签完全相同
        return classList[0]#stop splitting when all of the classes are equal #返回该标签
    if len(dataSet[0]) == 1:  #数据集的每个特征都划分完了（这里每划分一次，就删除该标签）    #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)  #投票决定
    bestFeat = chooseBestFeatureToSplit(dataSet)   #选择当前这个数据集的最好划分属性
    bestFeatLabel = labels[bestFeat]    #该属性值都有哪些标签 
    myTree = {bestFeatLabel:{}}     
    del(labels[bestFeat])   #删除该属性的类别
    featValues = [example[bestFeat] for example in dataSet]  #该属性值下所有的取值，该属性值对应的那一列
    uniqueVals = set(featValues)  #去重
    for value in uniqueVals: 
        subLabels = labels[:]     #复制类别标签，保证每个递归操作不改变原始列表的内容           #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)   #递归操作
    return myTree                            
    
#用训练的分类器分类新的数据的函数（输入inputTree表示训练的决策树  featLabels代表输入的属性类别   testVec代表对应属性类别的值，列表表示，如[0,1]   ）  
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

#将构建号的决策树存储在硬盘文件中，这样每次分类时直接导入该决策树即可
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
#导出决策树    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
