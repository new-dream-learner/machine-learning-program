'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *
#数据预处理，将所有数据放在一个列表中，而不是每个目标变量放在不同列表
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))  #每行数据映射为浮点数      #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

#输入：数据集合、待切分特征，切分阈值
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]   #返回大于阈值的数据集
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

#建立叶子节点函数，生成叶子节点
#当不再对数据进行切分时，在每个数据集上调用该函数得到叶子节点
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

#误差估计函数，总方差
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#模型树叶子节点生成
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

#切分函数，输入数据集，找到最佳的二元切分方式，找不到，则调用函数生成叶子节点
#如果找到最佳切分方式，返回特征编号和切分特征值
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]    #控制函数停止时机的输入参数，分别为容许的误差下降值以及切分的最小样本数
    #if all the target variables are the same value: quit and return value
    #统计剩余的不同特征的数量，数量为1则不再切分直接返回
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)    #数据集大小
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)   #数据集总体误差，用于与新切分误差进行对比
    bestS = inf; bestIndex = 0; bestValue = 0  
    for featIndex in range(n-1):    #对于每个特征
        for splitVal in set(dataSet[:,featIndex]):  #对该特征下所有不同的取值  
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)   #以该特征的某个取值切分数据集
            #判断是否切分的最小样本数小于给定值，小于的话跳出循环，不使用该切分方式
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)   #计算切分后数据的新误差 
            if newS < bestS:    #新误差更小，更新
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:    #判断误差降低度，若降低度小于给定值，则不切分数据，直接创建叶子节点
        return None, leafType(dataSet) #返回到节点生成函数  exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)   #获取最佳切分方式的两个数据子集 
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): #小于给定参数，也直接退出切分，直接创建叶子节点    #exit cond 3  
        return None, leafType(dataSet)
    return bestIndex,bestValue  #返回最佳切分方式  #returns the best feature to split on
                                  #and the value used for that split
#创建数方式，输入：数据集、叶子节点生成函数、误差函数、切分停止参数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)   #选择该数据集下最好的切分方式，或者直接返回none          #choose the best split
    if feat == None: return val    #如果无需切分，直接返回叶子节点           #if the splitting hit a stop condition return val
    #需要切分数据集
    retTree = {}  #  
    retTree['spInd'] = feat  #切分特征
    retTree['spVal'] = val   #切分值
    lSet, rSet = binSplitDataSet(dataSet, feat, val)   #切分后数据集
    #对两个切分后的数据集分成两棵树，直到无需切分为止
    retTree['left'] = createTree(lSet, leafType, errType, ops)   
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree   #返回切分后的左右子树··  

#判断是否是一棵树，即当前节点是否是叶结点
def isTree(obj):
    return (type(obj).__name__=='dict')

#从上往下遍历，如果找到两个叶子节点就计算其平均值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)   #判断测试集是否为空   #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):              #if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    #递归对数据集进行切分
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else: return tree
    else: return tree

#输入数据点，根据模型返回一个预测值
def regTreeEval(model, inDat):
    return float(model)

#对叶结点数据预测
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

#遍历整棵树，直到命中叶子节点
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

#多次调用上述函数，返回一组预测值       
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat