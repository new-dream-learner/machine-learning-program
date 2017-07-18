'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *

#数据导入，最后返回列表，列表中元素为列表，并代表一个数据
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))   #变成浮点数对象后返回一列表 #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

#计算两个向量之间的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

#为数据集构建包含k个随机质心的的集合
def randCent(dataSet, k):
    n = shape(dataSet)[1]   #数据点维度
    centroids = mat(zeros((k,n)))    #k个质心存储矩阵          #create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])  #每个特征下的最小值 
        rangeJ = float(max(dataSet[:,j]) - minJ)    #最大值减去最小值获得范围
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))    #随机在该范围内生成随机数，生成质心
    return centroids

#k-means聚类 ，返回聚类中心以及每个数据点相对于最终质心的距离的平方
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]   #数据点个数
    clusterAssment = mat(zeros((m,2)))  #               #create mat to assign data points 
                                                        #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment

#二分k-均值算法，把所有店看成一个簇，根据误差平方决定判定接下来如何划分簇
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]  #数据点个数
    clusterAssment = mat(zeros((m,2)))   #簇分配结果以及误差存放矩阵
    centroid0 = mean(dataSet, axis=0).tolist()[0]  #创建一个初始的簇
    centList =[centroid0]         #初始质心放入质心列表                   #create a list with one centroid
    for j in range(m):#calc initial Error    #遍历所有点，计算每个点到质心的误差值，存入分配矩阵第二列
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf   #初始误差，设为极大
        for i in range(len(centList)):  #遍历质心列表中所有簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]   #获取该簇下的所有点当作一个小数据集                 #get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  #对该数据集分成两个簇
            sseSplit = sum(splitClustAss[:,1])       #划分后的总误差               #compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])   #没有进行划分簇的数据集的误差（剩余数据集）
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:    #小于初始误差，更新参数
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #将数据点的簇编号修改为划分簇或新加簇编号
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]             #replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])   #添加新的质心
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss     #重新分配簇分配情况以及误差情况#reassign new clusters, and SSE
    return mat(centList), clusterAssment

import urllib
import json
#输入地址和城市输出该位置的信息
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)   #将上述字典转化为可以通过URL进行传递的字符串格式
    yahooApi = apiStem + url_params      #print url_params
    print yahooApi                #API形式，两个相加
    c=urllib.urlopen(yahooApi)     #打开URL读取返回值
    return json.loads(c.read())    #返回值是Json格式，因此可以对其进行解码为一个字典

from time import sleep
#打开一个文件，获取第二列和第三列的结果，输入上述函数中，输出该位置的字典，然后提取该字典中对应的经纬度值，然后添加到原先的对应行中，重新写入文件中
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])  #输入至上述函数，获取返回的该位置的信息字典
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])   #获取经纬度值
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))   #写入新文件place.txt中
        else: print "error fetching"  
        sleep(1)
    fw.close()

#球面余弦定理计算两个经维度之间距离 
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #返回两个经纬度之间的距离       pi is imported with numpy

#簇绘图函数
import matplotlib
import matplotlib.pyplot as plt
#输入参数为希望得到的簇的数量
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        #打开列表，获取所有位置的经纬度列表
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)  #经纬度矩阵
    #聚类操作，指定距离计算方式为上述函数
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()  #新建一张图
    rect=[0.1,0.1,0.8,0.8]  #图中新建一个矩形
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']      #标记形状的列表
    axprops = dict(xticks=[], yticks=[])          
    ax0=fig.add_axes(rect, label='ax0', **axprops)  3#画矩阵配置
    imgP = plt.imread('Portland.png')   #基于一幅图像创建矩阵
    ax0.imshow(imgP) #绘制该矩阵
    ax1=fig.add_axes(rect, label='ax1', frameon=False)  #在同一张图上绘制一张新的图，允许使用两套坐标系统
    #遍历每一个簇，将其画在第二个坐标系中
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]   #每个簇从初始数据集中找到其数据点，画图
        markerStyle = scatterMarkers[i % len(scatterMarkers)]   #绘制点的形式矩阵
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)   #对每个簇中的点画散点图
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)   #绘制每个簇的聚类中心
    plt.show()   #图显示
