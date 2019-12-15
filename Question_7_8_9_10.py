import numpy as np
from data import data
from sklearn.model_selection import KFold
from sklearn import svm

trainFile = "features.train.txt"
testFile = "features.test.txt"

trainX, trainY = data.load_file(trainFile)
testX, testY = data.load_file(testFile)

data =data(trainX, trainY,testX, testY)

SVM_MODEL = svm.SVC(C = 0.01, kernel = 'poly',degree = 2, coef0 = 1.0, gamma = 1.0)

data.set_filter([1,5])
TrainXlv5 = data.get_X("train")
TrainYlv5 = data.get_Y("train")
TestXlv5  = data.get_X("test")
TestYlv5  = data.get_Y( "test")

print("Question 7 and Question 8")
Cs = [pow(10, -x) for x in reversed(range(5))]
cvRunTime = 100 #run
cvSplitNum = 10 #splits

#10-fold cross validation setup
cv_kf = KFold(n_splits=cvSplitNum)

e_cvs = np.ndarray((0, len(Cs)))
cv_minimum = np.zeros(len(Cs)) #record of the winning C each run
#iterate over runs
for i in range(cvRunTime):
    #iterate over possible c values
    cur_ecvs = np.array([])
    for C in Cs:
        SVM_MODEL.C = C
        e_vals = np.array([]) #array of validation errors
        #iterate over each fold
        for train_idx, test_idx in cv_kf.split(TrainXlv5):
            cv_xtrain, cv_xtest = TrainXlv5[train_idx], TrainXlv5[test_idx]
            cv_ytrain, cv_ytest = TrainYlv5[train_idx], TrainYlv5[test_idx]
            SVM_MODEL.fit(cv_xtrain, cv_ytrain)
            cur_err = 1.0 - SVM_MODEL.score(cv_xtest, cv_ytest)
            e_vals = np.concatenate((e_vals, [cur_err]))
        cur_ecv = np.average(e_vals) #i'th cv error
        cur_ecvs = np.concatenate((cur_ecvs, [cur_ecv]))
    win_idx = np.argmin(cur_ecvs) #index of the winning C
    #mark winner in our records
    cv_minimum[win_idx] = cv_minimum[win_idx] + 1
    #add cv errors to our records
    e_cvs = np.vstack((e_cvs, cur_ecvs))

#find average e_cvs for each C
ecv_avg = np.average(e_cvs, axis=0)
bestInAll = np.argmax(cv_minimum)
ecv_win = ecv_avg[bestInAll]

print("C is ",Cs[bestInAll],"is selected most often with average E_cv is " ,ecv_win )

print("\n\n\n\n\n")


print("Question 9 and Question 10")
rbf_C = [pow(10,x) for x in range(-2,7,2)]

SVM_MODEL.kernel = 'rbf'
SVM_MODEL.gamma = 1

for C in rbf_C:
    SVM_MODEL.C = C
    SVM_MODEL.fit(TrainXlv5, TrainYlv5)
    cur_ein = 1.0 - SVM_MODEL.score(TrainXlv5, TrainYlv5)
    cur_eout = 1.0 - SVM_MODEL.score(TestXlv5, TestYlv5)
    print("C is ",C," E_in = " ,cur_ein,"E_out = ", cur_eout)
