# Coding:utf-8

import tensorflow as tf
import numpy as np

# 每次喂入数据的个数
BATCH_SIZE = 8
# 随机种子
seed = 23455
# 利用随机种子产生随机数
rng = np.random.RandomState(seed)

# 随机产生已知数据X，
X = rng.rand(32, 2)
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print('X:\n', X)
print('Y:\n', Y)

# 设置输入x和输出y_的占位
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

# 随机产生权值矩阵
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 搭建前向传播网络
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 损失函数计算
# 均方误差
loss = tf.reduce_mean(tf.square(y-y_))
# 训练 学习率
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 开启会话计算结果
with tf.Session() as sess:
    # 所有变量初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("w1:\n", w1)
    print("w2:\n", w2)
    print("\n")

    # 计算轮数
    STEPS = 5000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start+BATCH_SIZE
        # 喂入数据训练
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            # 喂入数据 计算总损失 每500次计算输出一次
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training step(s), loss on all data is %g" % (i, total_loss))

    print('\n')
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))
