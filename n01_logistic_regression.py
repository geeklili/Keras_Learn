import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# 每次运行这个文件的时候，产生的随机数列都一样
np.random.seed(5)

# 创建数据集, linespace:在指定的间隔内返回均匀间隔的数字
X = np.linspace(-1, 1, 200)
# print(X)
# 打乱这些数据
np.random.shuffle(X)
# print(X)
# loc: 均值， scale: 标准差， size: 数据集合大小
Y = 0.5 * X + 2 + np.random.normal(loc=0, scale=0.05, size=200)
# print(Y)
# 查看样本数据, 画出散点图
plt.scatter(X, Y)
plt.show()

# 分割数据集
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# 创建模型，Sequential(): 多个网络层的线性堆叠模型
model = Sequential()
# 添加神经网络层，并指定，input_dim：输入维度的个数，units：神经元的个数
model.add(Dense(input_dim=1, units=1))
# 选择损失函数，优化器
model.compile(loss='mse', optimizer='sgd')

# 训练模型,300步
print('training...')
for step in range(3001):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('第%d步' % step, 'train_cost:', cost)

print('testing...')
# 评估测试集
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost is: ', cost)

# 获取网络的权重w与b
w, b = model.layers[0].get_weights()
print('w: ', w, 'b: ', b)

# 展示预测集
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()




