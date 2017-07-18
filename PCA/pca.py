'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
#PCA降维步骤：1.去除平均值    2.计算样本协方差矩阵     3.计算样本协方差矩阵的特征值和特征向量    
#4.将特征值从大到小排序    5.保留最上面的N个特征向量       6.将数据转化到上述N各特征向量的新空间完成降维为N维

from numpy import *

#导入数据,输出矩阵表示的浮点型数据，每个数据一个列表
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list[map(float,line)] for line in stringArr]
    return mat(datArr)
#PCA降维，输入数据矩阵以及返回的特征数
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)  #数据平均值
    meanRemoved = dataMat - meanVals #数据减去平均值   remove mean
    covMat = cov(meanRemoved, rowvar=0)   #计算其协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))    #计算出特征值和特征向量
    eigValInd = argsort(eigVals)            #特征值进行排序                  sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]   #指定个数的最大特征向量索引    #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]    #返回指定个数特征向量   #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  #原始数据映射至特征向量构建的新空间，降维之后的矩阵   #transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals   #重构之后的数据集
    return lowDDataMat, reconMat

#将nan替换为平均值
def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
