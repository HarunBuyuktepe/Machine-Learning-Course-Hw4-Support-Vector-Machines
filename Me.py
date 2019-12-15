import numpy as np
from data import data
from sklearn import svm

trainSet = "features.train.txt"
testSet  = "features.test.txt"

trainX, trainY = data.load_file(trainSet)
testX, testY = data.load_file(testSet)

data =data(trainX, trainY,testX, testY)

#defining learning model
SVM_MODEL = svm.SVC(C = 0.01, kernel = 'poly',degree = 2, coef0 = 1.0, gamma = 1.0)

OddSVM_num = np.array([])
EvenSVM_num = np.array([])

eIns = []

for cur_num in range(10):
    data.set_filter([cur_num])
    cur_X = data.get_X("train")
    cur_Y = data.get_Y("train")

    #feed model with train data
    SVM_MODEL.fit(cur_X, cur_Y)
    cur_score = SVM_MODEL.score(cur_X, cur_Y)
    cur_numSvm = SVM_MODEL.n_support_
    cur_svmsum = np.array(cur_numSvm).sum()

    eIns.append([cur_num,(1-cur_score)])
    if cur_num % 2 == 0:
        EvenSVM_num = np.concatenate((EvenSVM_num, [cur_svmsum]))
    else:
        OddSVM_num = np.concatenate((OddSVM_num, [cur_svmsum]))

aodd_sum = np.sum(OddSVM_num)
aeven_sum = np.sum(EvenSVM_num)

print("\nQuestion 2")
for cur_num in range(0,10,2):
    print(eIns[cur_num][0], " versus all in-sample error: ", eIns[cur_num][1])
print("\nQuestion 3")
for cur_num in range(1,10,2):
    print(eIns[cur_num][0], " versus all in-sample error: ", eIns[cur_num][1])
print("\nQuestion 4")
a_diff = abs(aodd_sum - aeven_sum)
print("The difference between the number of supportvectors is" , a_diff)


data.set_filter([1,5])
TrainXlv5 = data.get_X("train")
TrainYlv5 = data.get_Y("train")
TestXlv5  = data.get_X("test")
TestYlv5  = data.get_Y( "test")

pk_Q = [2,5]
pk_C = [pow(10, -x) for x in reversed(range(5))]
print("\nQuestion 5 and Question 6")
for Q in pk_Q:
    SVM_MODEL.degree = Q
    print("For Q is ",Q)
    for C in pk_C:
        SVM_MODEL.C = C
        SVM_MODEL.fit(TrainXlv5, TrainYlv5)
        cur_ein = 1.0 - SVM_MODEL.score(TrainXlv5, TrainYlv5)
        cur_eout = 1.0 - SVM_MODEL.score(TestXlv5, TestYlv5)
        cur_numSvm = SVM_MODEL.n_support_
        cur_svmsum = np.array(cur_numSvm).sum()
        print("For C is ",C ,"\nE_in = ",cur_ein, " E_out = ",cur_eout,"\nNumber of support vector = " ,cur_svmsum)
