'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *
#数据生成
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#为数据集中的每个元素，返回单个集的固定列表，例如数据中共有1 2 3 4 5 6中商品，返回已上商品单个组成的列表集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])  #不重复的加入
                
    C1.sort()
    return list(map(frozenset, C1))   #将列表中的值变成不可变类型  #use frozen set so we
                            #can use it as a key in a dict    

#输入：数据集、一个项集以及最小支持度，返回不小于该支持度的所有项集以及对应的支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:    #所有数据
        for can in Ck:   #所有项集
            if can.issubset(tid):
                #计算该元素在数据中出现的次数
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems   #每个项集元素的支持度
        if support >= minSupport:
            retList.insert(0,key)     #大于阈值。加入新集合
        supportData[key] = support     #每个项集元素的对应支持度存储字典
    return retList, supportData    

#输入频繁项集列表以及项集元素个数，输出组合后的所有项集组成的列表
#例如：1 2 3输入，输出12 13 23      12  13 23输入。输出123
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)   #项集中元素个数
    for i in range(lenLk):  #对于所有项集元素
        for j in range(i+1, lenLk):  #所选项集的所有下一个元素
        #如果这两个集合的前k-2项元素都相等，将两个集合合成大小为k的集合
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]    
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union  #合并集合
    return retList

#频繁项集挖掘函数
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)  #创建基础项集
    D = list(map(set, dataSet))   #将原始数据集映射为集合的形式
    L1, supportData = scanD(D, C1, minSupport)   #获取单元素的频繁集
    L = [L1]  #存储所有频繁项集的列表 
    k = 2  
    #寻找所有频繁项集
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)   #更新新增项集的支持度
        L.append(Lk)               #加入频繁项集列表
        k += 1
    return L, supportData         #返回频繁项集以及所有频繁项集对应的支持度（不同元素长度的频繁项分别组成一个列表存在一个大列表里）

#关联规则计算函数：输入频繁项集的列表，对应支持度字典以及最小置信度，输出生成的候选规则集合
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []  #候选规则存储列表
    for i in range(1, len(L)):#对于每个不同元素个数的频繁项集           #only get the sets with two or more items
        for freqSet in L[i]:  #对于每个超过两个元素的频繁项集
            H1 = [frozenset([item]) for item in freqSet]     #为每个多元素频繁项集创建只包含单个元素集合的列表
            if (i > 1):       #如果频繁项集中的元素超过2
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)   #进一步合并函数
            else:  #该频繁集里只有两个元素
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)    #计算可信度
    return bigRuleList         

#计算规则可信度，并返回满足阈值的规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #计算单个元素之间的可信度calc confidence
        if conf >= minConf:   
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH   #返回满足要求的规则

#从最初项集生成更多关联规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])   #频繁项集H大小
    if (len(freqSet) > (m + 1)): #判断该频繁项集是否达到可以移除m大小的子集try further merging
        Hmp1 = aprioriGen(H, m+1)      #生成H中元素的无重复组合    #create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)   #计算关联度
        if (len(Hmp1) > 1):    #need at least two sets to merge     不止一条规则满足要求
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)   #进一步组合规则
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print (itemMeaning[item])
        print ("           -------->")
        for item in ruleTup[1]:
            print (itemMeaning[item])
        print ("confidence: %f" % ruleTup[2])
        print()       #print a blank line
        
            
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning
