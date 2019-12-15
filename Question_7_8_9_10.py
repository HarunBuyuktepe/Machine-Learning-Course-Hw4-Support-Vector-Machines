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
x_1v5_train = data.get_X("train")
y_1v5_train= data.get_Y("train")
x_1v5_test = data.get_X("test")
y_1v5_test= data.get_Y( "test")

print("Question 7 and Question 8")
cv_Q = 2
cv_C = [pow(10, -x) for x in reversed(range(5))]
cv_runs = 100 #number of runs
cv_splits = 10 #number of splits

#note that k-fold validation is not in sklearn 0.17
cv_kf = KFold(n_splits=cv_splits)


e_cvs = np.ndarray((0, len(cv_C)))
cv_winner = np.zeros(len(cv_C)) #record of the winning C each run
#iterate over runs
for cur_run in range(cv_runs):
    #iterate over possible c values
    cur_ecvs = np.array([])
    for C in cv_C:
        SVM_MODEL.C = C
        e_vals = np.array([]) #array of validation errors
        #iterate over each fold
        for train_idx, test_idx in cv_kf.split(x_1v5_train):
            cv_xtrain, cv_xtest = x_1v5_train[train_idx], x_1v5_train[test_idx]
            cv_ytrain, cv_ytest = y_1v5_train[train_idx], y_1v5_train[test_idx]
            SVM_MODEL.fit(cv_xtrain, cv_ytrain)
            cur_err = 1.0 - SVM_MODEL.score(cv_xtest, cv_ytest)
            e_vals = np.concatenate((e_vals, [cur_err]))
        cur_ecv = np.average(e_vals) #current cv error
        cur_ecvs = np.concatenate((cur_ecvs, [cur_ecv]))
    win_idx = np.argmin(cur_ecvs) #index of the winning C
    #mark winner in our records
    cv_winner[win_idx] = cv_winner[win_idx] + 1
    #add cv errors to our records
    e_cvs = np.vstack((e_cvs, cur_ecvs))

#find average e_cvs for each C
ecv_avg = np.average(e_cvs, axis=0)
overall_winner = np.argmax(cv_winner)
ecv_win = ecv_avg[overall_winner]

print("C is ",cv_C[overall_winner],"is selected most often with average E_cv is " ,ecv_win )

print("\n\n\n\n\n")


print("Question 9 and Question 10")
rbf_C = [pow(10,x) for x in range(-2,7,2)]

SVM_MODEL.kernel = 'rbf'
SVM_MODEL.gamma = 1

for C in rbf_C:
    SVM_MODEL.C = C
    SVM_MODEL.fit(x_1v5_train, y_1v5_train)
    cur_ein = 1.0 - SVM_MODEL.score(x_1v5_train, y_1v5_train)
    cur_eout = 1.0 - SVM_MODEL.score(x_1v5_test, y_1v5_test)
    print("C is ",C," E_in = " ,cur_ein,"E_out = ", cur_eout)
