# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN
import os
os.chdir('D:\\MLlearning\\Machine-Learning\\kNN\\3.数字识别')
"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
	filename - 文件名
Returns:
	returnVect - 返回的二进制图像的1x1024向量

Modify:
	2017-07-15
"""
def img2vector(filename):
    fr=open(filename)

    #first Parameter of np.zeros, is shape, a two element group
    reVector=np.zeros((1,1024))
    for i in range(32):#读取第i行的内容
        fileline=fr.readline()
        for j in range(32):
            reVector[0,32*i+j]=int(fileline[j])
    
    return reVector

"""
函数说明:手写数字分类测试

Parameters:
	无
Returns:
	无
"""
def handwritingClassTest():
    #Test set label create:
    hw_Label=[]
    #get testSet file dic
    trainingFileList=listdir('trainingDigits')
    #get the amount of file in trainingDigits directory
    m=len(trainingFileList)

    trainingMat=np.zeros((m,1024))
    for i in range(m):
        #依次把每个文件输入进去。
        trainingFileName=trainingFileList[i]
        #获取正确的答案，从文件名获得，第一位。
        hw_Label.append(trainingFileName.split('_')[0])
        #create training Matrix
        trainingMat[i,:]=img2vector('trainingDigits/%s'%trainingFileName)
    
    neigh=kNN(n_neighbors=3,algorithm='auto')
    neigh.fit(trainingMat,hw_Label)

    #input the test data
    testList=listdir('testDigits')
    #count the error
    err=0.0
    mTest=len(testList)
    #一个个做
    for i in range(mTest):
        testFileName=testList[i]
        #get the true value, to test the model' s judgement is right or wrong
        testTrueValue=int(testFileName.split('_')[0])
        #turn testfile to a vector
        testVector=img2vector(f'testDigits/{testFileName}')
        testPredictValue=int(neigh.predict(testVector)[0])

        print(f"true number is{testTrueValue}, while the predict number is{testPredictValue}")
        if testTrueValue!=testPredictValue:
            err+=1
    
    print("Err amount is: ",err)
    print("Err rate is:",100*err/mTest,'%')
        

        

    



if __name__=='__main__':
    filename='trainingDigits/0_0.txt'
    reVector=img2vector(filename)
    handwritingClassTest()
    print(reVector) 