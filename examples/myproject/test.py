import tensorflow as tf
import cmath
import numpy as np
import gym
# sess=tf.Session()
# log_path="log"
# writer=tf.summary.FileWriter(log_path,sess.graph)
#
# tf.summary.scalar("name",variable_name)#标量信息存储
#
# #张量转换为标量输出，从均值，标准差，最大值等多个角度转化标量
#
# def variable_summaries(var):
#     with tf.name_scope("summaries"):
#         mean=tf.reduce_mean(var)
#         tf.summary.scalar("mean",mean)
#         with tf.name_scope("stddev"):
#             stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
#         tf.summary.scalar("stddev",stddev)
#         tf.summary.scalar("max",tf.reduce_max)
#         tf.summary.scalar("min",tf.reduce_min)
#
# #张量转换为直方图输出
#
# with tf.name_scope("Nerual_Node"):
#     W=tf.Variable(tf.random_normal([inputDim,1],name='weight'))
#     tf.summary.histogram('weight',W)
#     b=tf.Variable(tf.zeros([1],'b'))
#     tf.summary.histogram('bias',b)
#
#
# merged=tf.summary.merge_all()  #整理所有日志生成操作
#
# for i in range(train_steps):
#     summary,_=sess.run([merged,train_step],feed_dict={x:xs,y:ys})
#     writer.add_summary(summary,i)
# # distance,theta=cmath.polar(complex(-2.5525988750998185 ,3.482649557006688))
# # print(distance,np.degrees(theta))
# # b=np.arcsin(3.482649557006688/distance)
# print(b)
# a=np.array([1,2,3,4,np.inf,np.inf,8])
# b=[3]
# data=np.hstack([a,b])
#
# # a[np.where(np.isinf(np.array(a)))[0]]=0
# # data=data.reshape(2,4)
# data=np.asarray(data)
# print(type(data))
# import numpy as np
# import tensorflow as tf
#
# phi=np.array([1,2,3,9,8,7])
# index=np.arange(2,5)
# a=np.empty((4,2))
# a[0:3]=phi.take(index,axis=0,mode="wrap")
# a[-1]=4
# a=[[1,2,3],
#    [4,5,6]]
# b=[[1,2,3],
#    [4,5,6]]
# c=[[1,2,3],
#    [4,5,6]]
# print("a=",a)
# print("b=",b)
# print("c=",c)

# a=np.array([1,2,3])
# b=np.array([2,3,4])
# c=np.vstack([a,b])
# # d=np.stack((a,b,c),axis=2)
# a=a.reshape(len(a),1)
# a=np.zeros((3,4))
# b=np.ones(7)
# a=np.array([[1,2,3],[1,2,4],[2,3,4],[2,3,5]])
# # memory=np.hstack((a[1],[1,2]))
# data=a[:,0:2].reshape((1,8))
# # b=data.reshape((1,8))
# # print(data)
# # print([b[1],)
# data2=np.array([[b[1]],[b[3]]])
# print(data2)
# print(data2[1,:])

# for i in range(a.shape[0]):
#     a[i:]=i*2+1
b=np.zeros((10,12))
for i in range(np.shape(b)[0]):
    b[i]=i*np.ones((1,12))
print(b)
# b=np.array([a[0],a[2]])
c=np.array([[1,2,3,2,3,4]])
print(type(b))
print(np.shape(b))
# b[0]=np.array([1,2,3])
# b[1,0]=[1,2,3,4]
# b[1]=np.array([1,1,1])
#
# b[2]=np.array([1,2,2])
# b[3]=np.array([1,2,2])
# c[2]=np.array([1,2,3])
# print(type(b))
# print(b)
# # print(c)
# data=np.array([b[1],b[2]])
# data2=np.vstack((b[1],b[2]))
# data3=np.vstack(([b[1],b[2]]))
# data=[b[1],b[2]]
# print(data)
# print(data2)
# print(data3)
# print([b[1][2]])
# print(c)
# print(b.reshape((1,16)))
phi=np.empty((4,12))
index=np.arange(5-4+1,5)
phi[0:3]=b.take(index,axis=0)
phi[-1]=b[4]
phi=phi.reshape(1,12*4)
print(phi)
print(phi[0][3])
print(np.shape(phi))