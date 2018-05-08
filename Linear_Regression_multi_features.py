import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_epoch=200001

x1= np.array([3., 1., 6., 1., 3., 3., 2., 5., 8., 4., 5., 2., 2., 1., 2.],dtype=np.float32)
x2= np.array([86., 60., 97., 86., 73., 77., 92., 75., 66., 74., 98., 98., 74., 78., 84.],dtype=np.float32)
x3= np.array([8., 2., 2., 6., 3., 2., 2., 4., 2., 0., 5., 1., 8., 5., 5.],dtype=np.float32)
y= np.array([3.38, 2.55, 4.3,  3.43, 3.02, 3.29, 3.85, 3.04, 2.8,  3.09, 4.12, 4.3,  2.79, 3.21,
 3.43],dtype=np.float32)

x1_train=x1[:10]/10
x2_train=x2[:10]/100
x3_train=x3[:10]/10
y_train=y[:10][np.newaxis]

x1_test=x1[10:]/10
x2_test=x2[10:]/100
x3_test=x3[10:]/10
y_test=y[10:][np.newaxis]

y_test=y_test.transpose()
x_train=np.vstack((x1_train,x2_train,x3_train))
x_test=np.vstack((x1_test,x2_test,x3_test)).transpose()
x_train=x_train.transpose()
y_train=y_train.transpose()
print(x_train)

X_train=tf.placeholder(tf.float32,shape=[None,3])
Y_train=tf.placeholder(tf.float32,shape=[None,1])

W=tf.Variable(tf.random_normal([3,1]),name='weight1')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=tf.matmul(X_train,W)+b
cost=tf.reduce_mean(tf.square(hypothesis-Y_train))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(num_epoch):
    cost_val,w,_b,_=sess.run([cost,W,b,train],feed_dict={X_train:x_train,Y_train:y_train})

    if step%2000==0:
        hyp_test = sess.run(tf.matmul(x_test, w) + _b)
        print(hyp_test,"\n\n",sess.run(tf.reduce_mean(tf.square(hyp_test-y_test))),"\n")
        