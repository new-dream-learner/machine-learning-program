'''
Created on Mar 8, 2011

@author: Peter
'''
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

#三种相似度计算方法
#欧式距离 
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

#皮尔逊系数
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

#余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

#基于物品相似度的推荐引擎，给定相似度计算方法，用户对物品的估计评分值
#输入：数据矩阵、用户编号、相似度计算方法、需要估计的物品编号
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]   #数据样本
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):   #遍历行中的每个物品：具体功能是对用户评过分的物品进行遍历，将其与其他物品进行比较
        userRating = dataMat[user,j]  #获取该用户对该物品的评分值
        if userRating == 0: continue     #如果用户没对该物品进行评分，跳过该物品
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]   #找到对物品item和j都进行打过分的用户（把第 item 列与 j 列求 与, 并取第一个不为 0 的行作为 overLap）
        if len(overLap) == 0: similarity = 0     #没有重合，相似度为0（没有用户同时点评过这两道菜）
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])   #计算两个物品评分相似度
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity   #总相似度
        ratSimTotal += similarity * userRating   #相似度*打分，累加
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal       
  
#基于SVD的评分估计
#现实中的评分矩阵很稀疏，用SVD进行转化成低维度的矩阵，再进行相似度计算 
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]   #菜品数量
    simTotal = 0.0; ratSimTotal = 0.0   #初始化
    U,Sigma,VT = la.svd(dataMat)    #对初始矩阵进行SVD分解
    Sig4 = mat(eye(4)*Sigma[:4])   #只利用前四个奇异值（包含90%能量值），并存储为矩阵形式   arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I    #将高维转化为低维度，构建经过转换后的物品情况    #create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal    #返回预测评分值

#产生最高的N个推荐结果
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]                   #用户未评分的物品列表   find unrated items 
    if len(unratedItems) == 0: return 'you rated everything'          #用户已经全部评分
    itemScores = []                  #编号和估计值存放列表
    for item in unratedItems:        #对所有用户没吃过的菜
        estimatedScore = estMethod(dataMat, user, simMeas, item)   #计算这个菜用户的估计打分值
        itemScores.append((item, estimatedScore))    #存入列表
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]   #对列表中每个元组取第2个值进行排序，得到排序后的N个评分值

#打印矩阵，函数遍历矩阵元素，大于阈值，输出1，小于输出0
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print (1),
            else: print (0),
        print ('')

#图像压缩函数，输入奇异值数量和打印图像的阈值
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    #输入图像，并转化为列表方式表示的像素
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)  #转化为矩阵表示
    print ("****original matrix******")
    printMat(myMat, thresh)   #矩阵包含浮点数，限定阈值，打印矩阵
    U,Sigma,VT = la.svd(myMat)   #图像矩阵进行SVD分解
    SigRecon = mat(zeros((numSV, numSV)))   #建立奇异值初始矩阵
    for k in range(numSV):     #对每个奇异值     construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]  #将奇异值放入初四矩阵对角线中
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]  #重构初始的图像
    print ("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)