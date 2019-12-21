import numpy as np

class Knn:
    def __init__(self, k = 1, mode = False):
        self.train_datas = []
        self.train_labels = []
        self.distance = np.abs if mode else np.square
        self.k = k

    def train(self, datas, labels):
        self.train_datas = datas
        self.train_labels = labels

    def predict(self, test_datas):
        debug_i = 0
        predictions = []
        number_of_classes = np.amax(self.train_labels) + 1
        for test_data in test_datas:
            distances = np.sum(self.distance(self.train_datas - test_data), axis = 1)
            votes = np.zeros(number_of_classes, dtype = np.int)
            for i in np.argsort(distances)[:self.k]:
                votes[self.train_labels[i]] += 1
            debug_pre = np.argmax(votes)
            predictions.append(np.argmax(votes))
            print('[debug]', debug_i, debug_pre)
            debug_i += 1
        return predictions


# Handle MNIST dataset
print('Handling MNIST dataset:')
    # train data pre-process
print('\tStart preprocess train data...')
dataf = open(r'mnist/train-images.idx3-ubyte', 'rb')
labelf = open(r'mnist/train-labels.idx1-ubyte', 'rb')
magic = np.fromfile(dataf, dtype = np.uint32, count = 1)
magic = np.fromfile(labelf, dtype = np.uint32, count = 1)
num_of_data = np.fromfile(dataf, dtype = np.uint32, count = 1)
num_of_data = np.fromfile(labelf, dtype = np.uint32, count = 1)
num_of_data = 60000
num_of_row = np.fromfile(dataf, dtype = np.uint32, count = 1)
num_of_row = 28
num_of_col = np.fromfile(dataf, dtype = np.uint32, count = 1)
num_of_col = 28
traindatas = np.fromfile(dataf, dtype = np.uint8).astype(int).reshape((num_of_data, -1))
trainlabels = np.fromfile(labelf, dtype = np.uint8)
dataf.close()
labelf.close()
    # END traindata pre-process

    # testing data pre-process
print('\tStart preprocess test data...')
dataf = open(r'mnist/t10k-images.idx3-ubyte', 'rb')
labelf = open(r'mnist/t10k-labels.idx1-ubyte', 'rb')
magic = np.fromfile(dataf, dtype = np.uint32, count = 1)
magic = np.fromfile(labelf, dtype = np.uint32, count = 1)
num_of_data = np.fromfile(dataf, dtype = np.uint32, count = 1)
np.fromfile(labelf, dtype = np.uint32, count = 1)
num_of_data = 10000
num_of_row = np.fromfile(dataf, dtype = np.uint32, count = 1)
num_of_row = 28
num_of_col = np.fromfile(dataf, dtype = np.uint32, count = 1)
num_of_col = 28
testdatas = np.fromfile(dataf, dtype = np.uint8).astype(int).reshape((num_of_data, -1))
testlabels = np.fromfile(labelf, dtype = np.uint8)
    # END test data pre-process

    # KNN predictor with hyperparameter: k = 3, distance = L2
print('\tKNN predictor with hyperparameter: k = 3, distance = L2')
specifier = Knn(3, False)
print('\t\tTraining...')
specifier.train(traindatas, trainlabels)
print('\t\tPredicting...')
predictlabels = specifier.predict(testdatas)
result_of = open('MNIST-KNNpredictions-L2K3.csv', 'w')
result_of.write('id,label\n')
misrate = 0
for i in range(int(num_of_data)):
    result_of.write("{},{}\n".format(i + 1, predictlabels[i]))
    if predictlabels[i] != testlabels[i]:
        misrate += 1
result_of.write('TEST ERROR RATE,{}'.format(misrate/num_of_data))
result_of.close()
print('\t\tTEST ERROR RATE = {}'.format(misrate/num_of_data))
print('\t\tResult written in MNIST-KNNpredictions-L2K3.csv')

    # KNN predictor with hyperparameter: k = 3, distance = L1
print('\tKNN predictor with hyperparameter: k = 3, distance = L1')
specifier = Knn(3, True)
print('\t\tTraining...')
specifier.train(traindatas, trainlabels)
print('\t\tPredicting...')
predictlabels = specifier.predict(testdatas)
result_of = open('MNIST-KNNpredictions-L1K3.csv', 'w')
result_of.write('id,label\n')
misrate = 0
for i in range(int(num_of_data)):
    result_of.write("{},{}\n".format(i + 1, predictlabels[i]))
    if predictlabels[i] != testlabels[i]:
        misrate += 1
result_of.write('TEST ERROR RATE,{}'.format(misrate/num_of_data))
result_of.close()
print('\t\tTEST ERROR RATE = {}'.format(misrate/num_of_data))
print('\t\tResult written in MNIST-KNNpredictions-L1K3.csv')

# Handle CIFAR10 dataset
print('Handling CIFAR10 dataset:')
print('PNGs have been pre-processed already, using them;')
print(r'Please visit LINK[https://rec.ustc.edu.cn/share/f26b56e0-240a-11ea-a360-c7fdce35faa8] to get the pre-processed data')
    # train data pre-process
print('\tStart preprocess train data...')
dataf = open(r'cifar10/train_data.dat', 'rb')
labelf = open(r'cifar10/train_label.dat', 'rb')
num_of_data = 50000
num_of_row = 32
num_of_col = 32
num_of_dep = 3
traindatas = np.fromfile(dataf, dtype = np.uint8).astype(int).reshape((num_of_data, -1))
trainlabels = np.fromfile(labelf, dtype = np.uint8)
dataf.close()
labelf.close()
    # END traindata pre-process

    # testing data pre-process
print('\tStart preprocess test data...')
dataf = open(r'cifar10/test_data.dat', 'rb')
num_of_data = 300000
num_of_row = 32
num_of_col = 32
num_of_dep = 3
testdatas = np.fromfile(dataf, dtype = np.uint8).astype(int).reshape((num_of_data, -1))
    # END test data pre-process

    # KNN predictor with hyperparameter: k = 1, distance = L2
print('\tKNN predictor with hyperparameter: k = 1, distance = L2')
specifier = Knn(1, False)
print('\t\tTraining...')
specifier.train(traindatas, trainlabels)
print('\t\tPredicting...')
predictlabels = specifier.predict(testdatas)
result_of = open('CIFAR10-KNNpredictions-L2K1.csv', 'w')
result_of.write('id,label\n')
for i in range(int(num_of_data)):
    result_of.write("{},{}\n".format(i + 1, predictlabels[i]))
result_of.close()
print('\t\tResult written in CIFAR10-KNNpredictions-L2K1.csv')

    # KNN predictor with hyperparameter: k = 3, distance = L1
print('\tKNN predictor with hyperparameter: k = 1, distance = L1')
specifier = Knn(1, True)
print('\t\tTraining...')
specifier.train(traindatas, trainlabels)
print('\t\tPredicting...')
predictlabels = specifier.predict(testdatas)
result_of = open('CIFAR10-KNNpredictions-L1K1.csv', 'w')
result_of.write('id,label\n')
for i in range(int(num_of_data)):
    result_of.write("{},{}\n".format(i + 1, predictlabels[i]))
result_of.close()
print('\t\tTEST ERROR RATE = {}'.format(misrate/num_of_data))
print('\t\tResult written in CIFAR10-KNNpredictions-L1K1.csv')