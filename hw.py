# Emin Ahmet Yazıcı - 150115064 - Homework4

import numpy as np
from sklearn import svm


# (x,y) = h.readdata()
def readdata():
    d = np.loadtxt('features.train.txt')
    return np.apply_along_axis(lambda x: x[1:3], 1, d), np.apply_along_axis(lambda x: x[0], 1, d)


def getbinary(y, choice=0):
    z = np.ones(len(y))
    z[y != choice] = -1
    return z


def runsvm(x, y, C=0.01, Q=2):
    # linear kernel would look like this
    # clf = svm.SVC(kernel='linear', C=C)
    # clf.fit(x, y2)
    # also test the custom kernel to verify the logic of kernel trick
    # clf = svm.SVC(kernel=poly_kernel, C=C, degree=Q, gamma=1, coef0=1)
    clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1, coef0=1)
    clf.fit(x, y)
    yhat = clf.predict(x)
    Ein = np.sum(y * yhat < 0) / (1. * y.size)
    return {'Ein': Ein, 'n_support': clf.support_vectors_.shape[0], 'clf': clf}
    # return {'Ein': Ein, 'b': clf.intercept_, 'clf': clf, 'n_support': clf.support_vectors_.shape[0]}


def q2():
    (x, y) = readdata()
    r0 = runsvm(x, getbinary(y, choice=0), C=0.01, Q=2)
    r2 = runsvm(x, getbinary(y, choice=2), C=0.01, Q=2)
    r4 = runsvm(x, getbinary(y, choice=4), C=0.01, Q=2)
    r6 = runsvm(x, getbinary(y, choice=6), C=0.01, Q=2)
    r8 = runsvm(x, getbinary(y, choice=8), C=0.01, Q=2)
    print('r0 -> ', r0)
    print('-----------------')
    print('r2 -> ', r2)
    print('-----------------')
    print('r4 ->', r4)
    print('-----------------')
    print('r6 ->', r6)
    print('-----------------')
    print('r8 ->', r8)
    # return {'Ein0': r0['Ein'], 'Ein2': r2['Ein'], 'Ein4': r4['Ein'], 'Ein6': r6['Ein'], 'Ein8': r8['Ein']}


def q3():
    (x, y) = readdata()
    r1 = runsvm(x, getbinary(y, choice=1), C=0.01, Q=2)
    r3 = runsvm(x, getbinary(y, choice=3), C=0.01, Q=2)
    r5 = runsvm(x, getbinary(y, choice=5), C=0.01, Q=2)
    r7 = runsvm(x, getbinary(y, choice=7), C=0.01, Q=2)
    r9 = runsvm(x, getbinary(y, choice=9), C=0.01, Q=2)
    print('r1 ->', r1)
    print('-----------------')
    print('r3 -> ', r3)
    print('-----------------')
    print('r5 ->', r5)
    print('-----------------')
    print('r7 ->', r7)
    print('-----------------')
    print('r9 ->', r9)
    # return {'Ein1': r1['Ein'], 'Ein3': r3['Ein'], 'Ein5': r5['Ein'], 'Ein7': r7['Ein'], 'Ein9': r9['Ein']}


def q4():
    (x, y) = readdata()
    print(runsvm(x, getbinary(y, choice=0))['n_support'] - runsvm(x, getbinary(y, choice=1))['n_support'])
    # return runsvm(x, getbinary(y, choice=0))['n_support'] - runsvm(x, getbinary(y, choice=1))['n_support']


def q5(Cs=[0.001, 0.01, 0.1, 1.0], Q=2):
    (x, y) = readdata()
    idx = np.logical_or(y == 1, y == 5)
    x = x[idx, :]
    y = getbinary(y[idx], 1)
    (xout, yout) = readdata()
    idx = np.logical_or(yout == 1, yout == 5)
    xout = xout[idx, :]
    yout = getbinary(yout[idx], 1)
    Ein = []
    Eout = []
    nsv = []
    for C in Cs:
        r = runsvm(x, y, C=C, Q=Q)
        Ein.append(r['Ein'])
        nsv.append(r['n_support'])
        clf = r['clf']
        yhat = clf.predict(xout)
        Eout.append(np.sum(yout * yhat < 0) / (1. * yout.size))

    print('Ein:  ', Ein)
    print('Eout: ', Eout)
    print('nsv:  ', nsv)
    return {'Ein': Ein, 'Eout': Eout, 'nsv': nsv}


def q6():
    r2 = q5(Cs=[0.0001, 0.001, 0.01, 0.1, 1.0], Q=2)
    r5 = q5(Cs=[0.0001, 0.001, 0.01, 0.1, 1.0], Q=5)
    return {'Q2': r2, 'Q5': r5}


# # k-fold cross validation version of SVM
def runsvm_cv(x, y, C=0.0001, Q=2, folds=10):
    kf = cross_validation.KFold(len(y), n_folds=folds, shuffle=True)
    Ecv = np.array([])
    Ein = np.array([])
    nsv = np.array([])
    i = 0
    for train, test in kf:
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        # print('fold %d: train_n %d, test_n %d' % (i, len(train), len(test)))
        r = runsvm(x_train, y_train, C=C, Q=Q)
        Ein = np.append(Ein, r['Ein'])
        nsv = np.append(nsv, r['n_support'])
        clf = r['clf']
        y_pred = clf.predict(x_test)
        Ecv = np.append(Ecv, np.sum(y_test * y_pred < 0) / (1. * y_pred.size))
        i += 1
    return {'Ecv': np.mean(Ecv), 'Ein': np.mean(Ein), 'nsv': np.mean(nsv)}


# use this for q8 as well
def q7(Cs=[0.0001, 0.001, 0.01, 0.1, 1.0], runs=100):
    (x, y) = readdata()
    idx = np.logical_or(y == 1, y == 5)
    x = x[idx, :]
    y = getbinary(y[idx], choice=1)
    wins = [0 for i in range(len(Cs))]
    Ecv = np.empty((runs, len(Cs)))
    for i in range(runs):
        print('iter: %d' % i)
        for j in range(len(Cs)):
            C = Cs[j]
            r = runsvm_cv(x, y, C=C, Q=2, folds=10)
            Ecv[i, j] = r['Ecv']
            # Ecv = np.append(Ecv, r['Ecv'])
        idx = np.argmin(Ecv[i, :])
        wins[idx] += 1
    return {'wins': zip(Cs, wins), 'Ecv': zip(Cs, np.mean(Ecv, 1).tolist())}


# SVC implements several kernels:
#  linear:     (x,x')
#  polynomial: (gamma*(x,x') + coef0)^degree
#  rbf:        exp(-gamma*||x,x'||^2)
#  sigmoid:    tanh(-gamma*(x,x') + coef0)
def runsvm_rbf(x, y, C=0.01):
    clf = svm.SVC(kernel='rbf', C=C, gamma=1.)
    clf.fit(x, y)
    yhat = clf.predict(x)
    Ein = np.sum(y * yhat < 0) / (1. * y.size)
    return {'Ein': Ein, 'b': clf.intercept_, 'clf': clf, 'n_support': clf.support_vectors_.shape[0]}


# use this for q10 as well
def q9(Cs=[0.01, 1, 100, 10 * 4, 10 * 6]):
    (x, y) = readdata()
    idx = np.logical_or(y == 1, y == 5)
    x = x[idx, :]
    y = getbinary(y[idx], 1)
    (xout, yout) = readdata()
    idx = np.logical_or(yout == 1, yout == 5)
    xout = xout[idx, :]
    yout = getbinary(yout[idx], 1)
    Ein = []
    Eout = []
    nsv = []
    for C in Cs:
        r = runsvm_rbf(x, y, C=C)
        Ein.append(r['Ein'])
        nsv.append(r['n_support'])
        clf = r['clf']
        yhat = clf.predict(xout)
        Eout.append(np.sum(yout * yhat < 0) / (1. * yout.size))

    print('Ein:  ', Ein)
    print('Eout: ', Eout)
    print('nsv:  ', nsv)
    # return {'Ein': Ein, 'Eout': Eout, 'nsv': nsv}



if __name__ == '__main__':
    q2()
    print('*****\n3')
    q3()
    print('*****\n4')
    q4()
    print('*****\n5')
    q5()
    print('*****\n6')
    q6()
    print('*****\n')
    q7()
    print('*****\n9-10')
    q9()

