#Bismillah

import numpy as np
import hw4dataload as LFD_Data2
from sklearn import svm

hw8_train = "features.train.txt"
hw8_test = "features.test.txt"
hw8_C = 0.01
hw8_Q = 2


def load_file(self, filename):
    ret_X = np.array([])
    ret_Y = np.array([])
    num_ex = 0  # number of examples
    X_dim = 0  # dimension of data
    with open(filename) as f:
        data = f.readlines()
        num_ex = len(data)
        X_dim = len(data[0].split()) - 1
        for line in data:
            cur_XY = [float(x) for x in line.split()]
            ret_X = np.concatenate((ret_X, cur_XY[1:]))  # everything but first elt
            ret_Y = np.concatenate((ret_Y, [cur_XY[0]]))  # first elt
    ret_X = ret_X.reshape((num_ex, X_dim))
    self.dim = X_dim
    return ret_X, ret_Y


trainX, trainY = load_file(hw8_train)
testX, testY = load_file(hw8_test)

print(trainX)



""""
my_svm = svm.SVC(C = 0.01, kernel = 'poly',degree = 2, coef0 = 1.0, gamma = 1.0)

alphas_odd = np.array([])
alphas_even = np.array([])

for cur_num in range(10):
    # cur_num-vs-all
    hw8_data.set_filter([cur_num])
    cur_X = hw8_data.get_X("train")
    cur_Y = hw8_data.get_Y("train")
    my_svm.fit(cur_X, cur_Y)
    cur_score = my_svm.score(cur_X, cur_Y)
    cur_numalphas = my_svm.n_support_
    cur_asum = np.array(cur_numalphas).sum()
    print("%d-vs-all binary classifier in-sample error: %f" % (cur_num, (1.0 - cur_score)))
    if cur_num % 2 == 0:
        alphas_even = np.concatenate((alphas_even, [cur_asum]))
    else:
        alphas_odd = np.concatenate((alphas_odd, [cur_asum]))

aodd_sum = np.sum(alphas_odd)
aeven_sum = np.sum(alphas_even)
a_diff = abs(aodd_sum - aeven_sum)
print("Diff in number of sv's between odd and even: %d" % a_diff)

#loading 1-vs-5 data

hw8_data.set_filter([1,5])
x_1v5_train = hw8_data.get_X("train")
y_1v5_train= hw8_data.get_Y("train")
x_1v5_test = hw8_data.get_X("test")
y_1v5_test= hw8_data.get_Y("test")

print(x_1v5_train.shape, y_1v5_train.shape, x_1v5_test.shape, y_1v5_test.shape)


pk_Q = [2,5]
pk_C = [pow(10, -x) for x in reversed(range(5))]

for Q in pk_Q:
    my_svm.degree = Q
    print("~~~ For polynomial kernels of degree Q = %d ~~~" % Q)
    for C in pk_C:
        my_svm.C = C
        my_svm.fit(x_1v5_train, y_1v5_train)
        cur_ein = 1.0 - my_svm.score(x_1v5_train, y_1v5_train)
        cur_eout = 1.0 - my_svm.score(x_1v5_test, y_1v5_test)
        cur_numalphas = my_svm.n_support_
        cur_asum = np.array(cur_numalphas).sum()
        print("C = %f | E_in = %f, E_out = %f, num_sv = %d" % (C, cur_ein, cur_eout, cur_asum))
    print("")

from sklearn.model_selection import KFold

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
        my_svm.C = C
        e_vals = np.array([]) #array of validation errors
        #iterate over each fold
        for train_idx, test_idx in cv_kf.split(x_1v5_train):
            cv_xtrain, cv_xtest = x_1v5_train[train_idx], x_1v5_train[test_idx]
            cv_ytrain, cv_ytest = y_1v5_train[train_idx], y_1v5_train[test_idx]
            my_svm.fit(cv_xtrain, cv_ytrain)
            cur_err = 1.0 - my_svm.score(cv_xtest, cv_ytest)
            e_vals = np.concatenate((e_vals, [cur_err]))
        cur_ecv = np.average(e_vals) #current cv error
        cur_ecvs = np.concatenate((cur_ecvs, [cur_ecv]))
    win_idx = np.argmin(cur_ecvs) #index of the winning C
    #mark winner in our records
    cv_winner[win_idx] = cv_winner[win_idx] + 1
    #add cv errors to our records
    e_cvs = n