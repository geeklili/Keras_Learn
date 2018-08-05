from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.datasets import mnist

# 下载数据集
# 训练集的大小为60000个，每个训练集特征值都是28*28的图片，标签值为0-9的数字
# 测试集的大小为10000个，每个测试集特征值都是28*28的图片，标签值为0-9的数字
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 对图片的数据进行归一化处理
# X_train: (60000, 28, 28)---->(60000, 784)
X_train = X_train.reshape(X_train.shape[0], -1) / 255
# X_test: (10000, 28, 28)---->(10000, 784)
X_test = X_test.reshape(X_test.shape[0], -1) / 255
# 把0-9的数字转化成了one_hot列表： [1. 0. 0. ... 0. 0. 0.]
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

# 构建神经网络
model = Sequential([
    Dense(units=32, input_dim=784),
    Activation('relu'),
    Dense(units=10),
    Activation('softmax')
])

# 定义自己的优化器:optimizer
rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

# 定义模型的损失函数，优化器
model.compile(
    optimizer=rmsprop,  # 此处也可以将：optimizer='rmsprop', 如果是字符串的rmsprop就是默认的没有改动的优化器
    loss='categorical_crossentropy',  # 交叉熵损失
    metrics=['accuracy']
)

# 训练数据集: epochs: 训练几个轮回， batch_size: 一次训练多少数据
model.fit(X_train, Y_train, epochs=2, batch_size=32)

# 测试数据
loss, accuracy = model.evaluate(X_test, Y_test)
print('loss:', loss)
print('accuracy:', accuracy)
