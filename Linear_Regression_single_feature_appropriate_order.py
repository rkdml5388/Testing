import numpy as np
import tensorflow as tf
Num_epoch=20001

x=np.arange(0,10,1)

y=x**2
n=len(y)

e=np.random.normal(0,0.1,n)
y_ob=y+e

x_train=x
y_train=y_ob

W1=tf.Variable(tf.random_normal([1]),name='weight1')
W2=tf.Variable(tf.random_normal([1]),name='weight2')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=b+W1*x_train+W2*x_train**2
cost=tf.reduce_mean(tf.square(hypothesis-y_train))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(Num_epoch):
    cost_val,_W1,_W2,_b,_=sess.run([cost,W1,W2,b,train])
    if step%10==0:
        print("weight1=",_W1,"weight2=",_W2,"bias=",_b)