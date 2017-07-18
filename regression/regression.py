'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *
#数据导入函数，预处理函数
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#标准回归函数，普通最小二乘法求出权值w
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T  #T代表转置
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  #判断能不能求逆
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

#局部加权线性回归，每次预测都要根据参数选出对应数据子集，每个训练数据点都有对应的权值
#给定x空间任意一点，以及训练集，输出对应预测值
#k值越小，则对预测点附近的点赋予更大权值，即在每次运算中，只有该点附近的点占据主要效果
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T   #训练集化为矩阵
    m = shape(xMat)[0]    #训练样本数
    weights = mat(eye((m)))    #eye函数方返回一个对角矩阵，对角值为1，其他值为0 
    for j in range(m):         #对于每个训练样本             #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #与预测样本的差值 
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))    #代入核函数求出该样本对应的权值，放入之前的对角矩阵中
    xTx = xMat.T * (weights * xMat)   #局部加权回归公式代入 
    if linalg.det(xTx) == 0.0:  #判断是否能求逆
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))     #返回预测权值
    return testPoint * ws    

#对数据集中的每个点都调用前面的LWLR函数进行预测
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat   #返回测试集预测的结果

#绘图预处理
def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

#误差计算函数
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

#岭回归（特征比数据点还多）：引入一个参数限制所有权值的和，通过引入惩罚项，减少不重要的参数
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam    #代入领回归公式
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
#在一组lam参数值下测试回归效果    
def ridgeTest(xArr,yArr):
    #数据预处理，标准化处理
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30    #30个不同的lam值
    wMat = zeros((numTestPts,shape(xMat)[1]))  #回归系数存储矩阵，每一行代表一个lam下的w权值系数
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))   #领回归计算权值
        wMat[i,:]=ws.T
    return wMat

#数据标准化操作
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

#前向逐步回归
#输入：数据集、预测值、迭代调整步长、迭代
次数
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    #数据标准化
    xMat = mat(xArr); yMat=mat(yArr).T 
    yMean = mean(yMat,0) 
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)   #数据样本数和特征数
    #returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()   #给定该数据集一个初始的权值，后续每个权值都一一进行更新得到最好权值
    #每次迭代，分别计算增加和减少某个特征对误差的影响
    for i in range(numIt):   #迭代次数
        print ws.T
        lowestError = inf; 
        for j in range(n):   #对于需要增加或者减少的每个特征
            for sign in [-1,1]:  #增加或减少
                wsTest = ws.copy()   
                wsTest[j] += eps*sign  #改变该特征的系数
                yTest = xMat*wsTest    #重新计算预测值
                rssE = rssError(yMat.A,yTest.A)   #改变后的新误差 
                if rssE < lowestError:    #误差变小了 
                    lowestError = rssE  #最低误差
                    wsMax = wsTest  #最佳权值 
        ws = wsMax.copy()  
        returnMat[i,:]=ws.T  #每次迭代的最优权值
    return returnMat   #返回迭代后的所有权值

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()

#购物信息获取函数
from time import sleep
import json
import urllib2
#Google API 收集数据，从返回的JSON中抽取价格
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)  #防止短时间有过多API调用
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    #查询的URL字符串，添加API的Key和带查询的套装信息
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)   #抓取内容 
    retDict = json.loads(pg.read())   #打开和解析内容操作，得到所有信息组成的一个字典
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]  
            if currItem['product']['condition'] == 'new':  #搜寻新套装
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']  #产品数组
            for item in listOfInv:  #对于所有套装
                sellingPrice = item['price']   #每个套装的价格 
                #如果一个套装比原始价格低一半，说明该套装不完整，过滤该信息
                if  sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    #解析成功后的套装
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print 'problem with item %d' % i

#多次调用上述函数    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

#交叉验证领回归
#输入：数据集中的X,Y值以及交叉验证次数
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)      #数据点个数                     
    indexList = range(m)   #打乱数据点的时候用到
    errorMat = zeros((numVal,30))  #初始误差矩阵 #create error mat 30columns numVal rows
    for i in range(numVal):  
        #训练集和测试集列表容器
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)  #将数据点混洗，实现训练集和测试集数据点随机选取

        #构造训练集和测试集
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: #（0.9代表测试集和训练集之间比率）  
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #用30个lam参数得到30个岭回归权值    #get 30 weight vectors from ridge
        for k in range(30):  #对于30个权值      #loop over all of the ridge estimates
            #测试集标准化，必须和训练集相同的参数
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)  #训练集平均值 
            varTrain = var(matTrainX,0)    #训练集方差
            matTestX = (matTestX-meanTrain)/varTrain  #测试集标准化     #regularize test with training params
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)   #代入每个岭回归权值，得到测试数据的预测标签              #test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY))   #每次交叉验证下不同权值的误差矩阵，行代表交叉验证数索引  列代表权值索引
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)   #计算10次交叉验证后的平均误差        #calc avg performance of the different ridge weight vectors  
    minMean = float(min(meanErrors))      
    bestWeights = wMat[nonzero(meanErrors==minMean)]   #取出10次交叉验证后具有最小误差的预测误差下的对应一组权值
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX   #之前的权值是数据标准化后的权值，返回最佳权值
    print "the best model from Ridge Regression is:\n",unReg    
    print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)  #常数项的值 