# K Nearest Neighbor 图像识别

## knn.py 代码说明

### 库依赖

`numpy`

### 输入输出文件

#### MNIST 数据集

[数据集下载地址](http://yann.lecun.com/exdb/mnist/)

[额外地址](https://rec.ustc.edu.cn/share/c4547e20-240a-11ea-8817-612b3b1eb75f)

##### 输入路径

- `./mnist/train-images.idx3-ubyte`
- `./mnist/train-labels.idx1-ubyte`
- `./mnist/t10k-images.idx3-ubyte`
- `./mnist/t10k-labels.idx1-ubyte`

##### 输出文件

- `MNIST-KNNpredictions-L2K3.csv`：使用 K = 3，L2 距离的 KNN 类获得的预测结果，其中文件的最后一行为对照标准答案的错误率。输出的标签为 0 ~ 9。

- `MNIST-KNNpredictions-L1K3.csv`：使用 K = 3，L1 距离的 KNN 类获得的预测结果，其中文件的最后一行为对照标准答案的错误率。输出的标签为 0 ~ 9。

  已有的一份输出结果可以在[此处](https://rec.ustc.edu.cn/share/fd45bc80-240e-11ea-94cc-d70f05087a77)下载。

#### CIFAR-10 数据集

[源数据集下载地址](https://www.kaggle.com/c/cifar-10/data)

本人将数据预处理为：

- 50000\*32\*32\*3 Byte 的输入图像矩阵，类型 `np.uint8`
- 50000\*1 Byte 的图像对应的类型向量，类型 `np.uint8`，类型对应编码分别为：
  - airplane 0
  - automobile 1
  - bird 2
  - cat 3
  - deer 4
  - dog 5
  - frog 6
  - horse 7
  - ship 8
  - truck 9
  
  预处理过程得益于[此 discussion](https://www.kaggle.com/c/cifar-10/discussion/110706)。

完整的本人所使用的预处理数据在[此链接](https://rec.ustc.edu.cn/share/f26b56e0-240a-11ea-a360-c7fdce35faa8)下载。

##### 输入路径

- `./cifar10/train_data.dat`
- `./cifar10/train_label.dat`
- `./cifar10/test_data.dat`

##### 输出文件

- `CIFAR10-KNNpredictions-L2K1.csv`：使用 K = 1，L2 距离的 KNN 类获得的预测结果。输出的标签为 “airplane” 等。由于没有标准答案，没有输出错误率。

- `CIFAR10-KNNpredictions-L1K1.csv`：使用 K = 1，L1 距离的 KNN 类获得的预测结果。输出的标签为 “airplane” 等。由于没有标准答案，没有输出错误率。

  已有的一份预测结果可以从[此处](https://rec.ustc.edu.cn/share/fd45bc80-240e-11ea-94cc-d70f05087a77)下载。

## supporting codes 说明

使用 C 编写的，对于两个数据集，均通过在训练集内随机选取 10000 个数据作为验证集，计算最佳的 K 的程序。

## 其他文件

`./docs` 包含本项目的相关文档。

`./reports` 包含本项目报告。