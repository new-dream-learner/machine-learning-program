'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度下降
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  #样本（100*3）转化为numpy矩阵             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose()   #样本类别 转化为numpy矩阵，并转置      #convert to NumPy matrix
    m,n = shape(dataMatrix)   #行列数
    alpha = 0.001
    maxCycles = 500   #迭代次数
    weights = ones((n,1))    #权值初始为1
    for k in range(maxCycles):          #每次迭代     #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #在每个样本中代入权值，代入sigmoid函数，求出预测值          #matrix mult
        error = (labelMat - h)              #和原始标签差值得到每个数据点的误差     #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error  #这里是矩阵运算（dataMatrix.transpose()* error相当于数据集每一个特征对应乘以误差后得到一个矩阵（相当于在每个特征上的最大梯度））      #matrix mult
    return weights

#根据权值画出拟合曲线以及初始的散点
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

#随机梯度下降，与梯度下降不同的是，每次更新权值只用了数据集中的一个样本的误差（但此时每次迭代样本的选取是按顺序的）
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))   #样本i的输出
        error = classLabels[i] - h      #样本i的误差
        weights = weights + alpha * error * dataMatrix[i]   #更新权值
    return weights

#改进的随机梯度下降：1.步长动态调整   2.每次更新权值用了数据集中随机的一个样本
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001   #步长动态变化 #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))   #数据集中随机选个数据           #go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))      
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])  #对于已经更新权值过的数据，删除
    return weights

#分类函数，将输入的特征值放入参数，输出类别
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

#分类器的训练和测试，输出在测试集上的错误率
def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')   #打开数据
    #数据的预处理，格式处理
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)   #使用梯度下降训练权值
    errorCount = 0; numTestVec = 0.0   
    #判断训练集中错误分类的样本数量     
    for line in frTest.readlines():    
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

#重复10次测试实验，然后取平均误差
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
        