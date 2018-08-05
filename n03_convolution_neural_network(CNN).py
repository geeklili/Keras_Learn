from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 数据集预处理
# X: (60000, 28, 28)----> (60000, 1, 28, 28), 相当于在外面加了一个[], 形式模拟3维的彩色图片
X_train = X_train.reshape(-1, 1, 28, 28) / 255
X_test = X_test.reshape(-1, 1, 28, 28) / 255
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

# 创建模型
model = Sequential()

# 以下为构建卷积神经网络
# input_shape(1, 28, 28)
# 1. 卷积层[1]: output shape (32, 28, 28)
model.add(Convolution2D(
    filters=32,  # 卷积核的个数
    kernel_size=5,  # 卷积核的大小
    strides=1,  # 卷积核每一次走几步
    padding='same',  # padding method
    input_shape=(1, 28, 28),  # channels, height, width
    data_format='channels_first',
))

# 2. 激活层
model.add(Activation('relu'))

# 3. 池化层[1]: output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,  # 池化核大小
    strides=2,  # 步长
    padding='same',
    data_format='channels_first',
))

# 4. 卷积层[2]: output shape (64, 14, 14)
model.add(Convolution2D(
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first'
))

# 5. 激活层
model.add(Activation('relu'))

# 6. 池化层[2]: output shape (64, 7, 7)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))

# 7. 全连接层[1]
# 将64维图片压缩成一维
model.add(Flatten())
# input_shape: (64 * 7 * 7), output_shape: 设置成1024
model.add(Dense(1024))
# 激活
model.add(Activation('relu'))

# 8. 全连接层[2]
model.add(Dense(10))
model.add(Activation('softmax'))

# 创建优化器
adam = Adam(lr=1e-4)

# 定义模型的损失函数，优化器
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
print('training...')
# epochs：一共训练几个轮回, batch_size：一次训练多少数据
model.fit(X_train, Y_train, epochs=1, batch_size=64)

# 测试
print('testing...')
loss, accuracy = model.evaluate(X_test, Y_test)

print('loss: ', loss)
print('accuracy: ', accuracy)
