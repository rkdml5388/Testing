import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_epoch = 20001

x1 = np.array([3., 1., 6., 1., 3., 3., 2., 5., 8., 4., 5., 2., 2., 1., 2.], dtype=np.float32)
x2 = np.array([86., 60., 97., 86., 73., 77., 92., 75., 66., 74., 98., 98., 74., 78., 84.], dtype=np.float32)
x3 = np.array([8., 2., 2., 6., 3., 2., 2., 4., 2., 0., 5., 1., 8., 5., 5.], dtype=np.float32)
y = np.array([3.38, 2.55, 4.3, 3.43, 3.02, 3.29, 3.85, 3.04, 2.8, 3.09, 4.12, 4.3, 2.79, 3.21,
              3.43], dtype=np.float32)

x1_train = x1[:10]/10
x1_train_ord2=x1_train**2
x1_train_ord3=x1_train**3
x2_train = x2[:10]/100
x2_train_ord2=x2_train**2
x2_train_ord3=x2_train**3
x3_train = x3[:10]/10
x3_train_ord2=x3_train**2
x3_train_ord3=x1_train**3
y_train = y[:10][np.newaxis]

x1_test = x1[10:]/10
x1_test_ord2 = x1_test**2
x1_test_ord3 = x1_test**3
x2_test = x2[10:]/100
x2_test_ord2 = x2_test**2
x2_test_ord3 = x2_test**3
x3_test = x3[10:]/10
x3_test_ord2 = x3_test**2
x3_test_ord3 = x3_test**3
y_test = y[10:][np.newaxis]

y_test = y_test.transpose()
x_train = np.vstack((x1_train,x1_train_ord2,x1_train_ord3, x2_train,x2_train_ord2,x2_train_ord3, x3_train,x3_train_ord2,x3_train_ord3))
x_test = np.vstack((x1_test,x1_test_ord2,x1_test_ord3, x2_test,x2_test_ord2,x2_test_ord3, x3_test,x3_test_ord2,x3_test_ord3)).transpose()
x_train = x_train.transpose()
y_train = y_train.transpose()
print(x_test)

X_train = tf.placeholder(tf.float32, shape=[None, 9])
Y_train = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([9, 1]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X_train, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(200001):
    cist_val, w, _b, _ = sess.run([cost, W, b, train], feed_dict={X_train: x_train, Y_train: y_train})
    if step % 2000 == 0:
        print(sess.run(tf.matmul(x_test, w) + _b),sess.run(tf.matmul(x_train,w)+_b))

print(y_train)