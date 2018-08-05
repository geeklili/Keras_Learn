from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.optimizers import Adam

TIME_STEPS = 28  # same as the height of the image/单次时间读取的步数，应该是一次读取多少行
INPUT_SIZE = 28  # same as the width of the image/视野的宽度
BATCH_SIZE = 50  # 一批数据包含几张图片
BATCH_INDEX = 0
OUTPUT_SIZE = 10  # 最终的输出结果有几个
CELL_SIZE = 50  # 图片变成了几维
LR = 0.001

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 数据预处理
X_train = X_train.reshape(-1, 28, 28) / 255  # normalize
X_test = X_test.reshape(-1, 28, 28) / 255  # normalize
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

# 创建rnn模型
model = Sequential()

# 进入网络
model.add(SimpleRNN(
    batch_size=(TIME_STEPS, INPUT_SIZE),
    output_dim=CELL_SIZE,
    unroll=True
))

# 输出层
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))  # 默认是tanh

# 优化器和损失函数
adam = Adam(LR)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# training
print('training...')
for step in range(5001):
    X_batch = X_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = Y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE, :]

    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    print(BATCH_INDEX)
    if step % 100 == 0:
        cost, accuracy = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test_accracy: ', accuracy)























