import numpy as np

class data:

    def load_file(filename):
        retX = np.array([])
        retY = np.array([])
        with open(filename) as file:
            data = file.readlines()
            num_ex = len(data)
            X_dim = len(data[0].split()) - 1
            for line in data:
                cur_XY = [float(x) for x in line.split()]
                retX = np.concatenate((retX, cur_XY[1:]))   #split data
                retY = np.concatenate((retY, [cur_XY[0]]))  #take first element
        retX = retX.reshape((num_ex, X_dim)) #delete unused column
        return retX, retY

    def __init__(self, trainX, trainY,testX, testY):
        self.train_X, self.train_Y = trainX, trainY
        self.test_X, self.test_Y = testX, testY
        self.filt_argc = 0


    def set_filter(self, params=[]):
        self.filt_argc = min(2, len(params))
        self.filt_argv = params
        if len(params) == 2:
            self.my_filt = np.vectorize(lambda x: int(x) == params[0] or int(x) == params[1])
            print(self.my_filt)
    
    def get_X(self, req_set="train"):
        if req_set.lower() == "train".lower():
            if self.filt_argc == 0 or self.filt_argc == 1:
                return self.train_X
            elif self.filt_argc == 2:
                #defined filter applied
                filtered = np.where(self.my_filt(self.train_Y))[0]
                return self.train_X[filtered]
        else:
            if self.filt_argc == 0 or self.filt_argc == 1:
                return self.test_X
            elif self.filt_argc == 2:
                #defined filter applied
                filtered =  np.where(self.my_filt(self.test_Y))[0]
                return self.test_X[filtered]

    def get_Y(self, req_set="train"):
        if req_set.lower() == "train".lower():
            if self.filt_argc == 0:
                return self.train_Y
            elif self.filt_argc == 1:
                # one-liner for mapping given param as 1 else -1
                return np.subtract(
                    np.multiply(2, np.equal(self.train_Y.astype(int), int(self.filt_argv[0])).astype(int)), 1)
            elif self.filt_argc == 2:
                #defined filter applied
                filtered = np.where(self.my_filt(self.train_Y))[0]
                return np.subtract(
                    np.multiply(2, np.equal(self.train_Y[filtered].astype(int), int(self.filt_argv[0])).astype(int)), 1)
        else:
            if self.filt_argc == 0:
                return self.test_Y
            elif self.filt_argc == 1:
                # one-liner for mapping given param as 1 else -1
                return np.subtract(
                    np.multiply(2, np.equal(self.test_Y.astype(int), int(self.filt_argv[0])).astype(int)), 1)
            elif self.filt_argc == 2:
                #defined filter applied
                filtered = np.where(self.my_filt(self.test_Y))[0]
                return np.subtract(
                    np.multiply(2, np.equal(self.test_Y[filtered].astype(int), int(self.filt_argv[0])).astype(int)), 1)
