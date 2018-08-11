import numpy as np
a = np.array([1,2,3,4])
b = np.array([1,2,8,9])
c = np.array([2,7,5,4])
data = np.zeros(3,dtype=object)
list=[]
data[0] = a
data[1] = b
data[2] = c
# print(data)
# print(data[0])
# print(data[0][0:2])
# # print(np.hstack(data[0:2]))
# for i in range(3):
#     temp=data[i]
#     print(temp)
#     list.extend(temp[:2])
temp=np.vstack((a,b,c))
# print(temp)
# print(temp[0:2].reshape(1,8))
data=(a,b,c)
a1,a2,a3=data
# a=[1,2,3]
# b=[1,2,3]
print(np.multiply(a,b))