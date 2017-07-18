'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *
#数据生成
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

#文件数据导入以及数据预处理
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#构建单层决策树：只对一个特征做判断，该特征大于阈值，得出结果，小于阈值，得出结果，如果有多个特征，就在多个特征上都实行所述操作
#输入数据集，要阈值判定的特征，输出该阈值下的的预测标签
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))   #ones函数后面接一个元组，shape返回行数和列数，将数组所有元素设为1
    if threshIneq == 'lt':  #判定大于阈值或小于阈值该样本应该属于哪个类别
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0    #将小于等于阈值的元素编成-1，就得到了该特征下对应该阈值下的预测标签 
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
#遍历所有特征，所有阈值取值，找到数据集上的最佳决策树（这个决策树是针对某个特征下的最佳阈值得来的决策树）
#输入数据集，标签以及初始的样本标签权值
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    #bestStump存储最佳决策树信息（哪一维度，哪个阈值，阈值下是哪个类别，对应错误率多大）
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions   #n为特征数，在每个特征上遍历
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();  #取该个特征下样本取值的上界和下界
        stepSize = (rangeMax-rangeMin)/numSteps   #决定步长，即阈值每次迭代后增加多少
        for j in range(-1,int(numSteps)+1):    #遍历该特征下所有步数        #loop over all range in current dimension
            for inequal in ['lt', 'gt']:       #迭代方式：是低于阈值变-1还是高于变-1    #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)     #j的作用所在，刚开始阈值取下界减去一个步长，这样能迭代所有情况
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)    #代入上述特征阈值判断函数，得到该特征该阈值下的预测标签     #call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))       #现将所有样本行置1
                errArr[predictedVals == labelMat] = 0   #预测标签与真实标签一样的样本行置0
                weightedError = D.T*errArr      #把错误向量和权重向量相应元素相乘并求和，计算加权错误率， #calc total error multiplied by D  
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:     #如果该误差小于目前最小误差
                    minError = weightedError     #当前误差变成最小误差
                    bestClasEst = predictedVals.copy()   #最佳预测样本值
                    bestStump['dim'] = i                 #最佳决策树的维度维度
                    bestStump['thresh'] = threshVal      #最佳阈值
                    bestStump['ineq'] = inequal          #当前阈值判断方向
    return bestStump,minError,bestClasEst

#算法训练，训练多个弱分类器，完成类别权值更新，alpha计算（每个弱分类器投票权重）等
#输入数据样本，样本标签，默认决策树弱分类器数量为40
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []  #存储所有弱分类器
    m = shape(dataArr)[0]  #样本数量
    D = mat(ones((m,1))/m)    #初始权值向量    #init D to all equal
    aggClassEst = mat(zeros((m,1)))    #每个数据点的类别估计累加值
    for i in range(numIt):  #迭代次数，训练多少个弱分类器上界
        bestStump,error,classEst = buildStump(dataArr,classLabels,)   #得出当前迭代下的最佳决策树  #build Stump
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))          #根据错误率计算alpha值                    #calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha                                    #alpha值存入    
        weakClassArr.append(bestStump)                  #当前分类器所有信息加入分类器组     # store Stump Params in Array
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)  #                        #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #                        Calc New D for next iteration
        D = D/D.sum()    #得到更新后的权重向量
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst  #当前apha值乘以当前决策树预测值
        #print "aggClassEst: ",aggClassEst.T
        #错误率累加计算
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0: break    #总错误率为0，结束
    return weakClassArr,aggClassEst

#用各弱分类器进行分类
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst    #类别估计值*alpha后累加
        print aggClassEst 
    return sign(aggClassEst)

#ROC曲线绘制以及AUC计算函数
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor  #绘制光标位置
    ySum = 0.0 #variable to calculate AUC  #AUC值
    numPosClas = sum(array(classLabels)==1.0)    #计算正例的数量
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)  #x，y轴步长计算
    sortedIndicies = predStrengths.argsort()    #获取排好序的索引   #get sorted index, it's reverse 
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep
