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
print(data[0][0:2])
# print(np.hstack(data[0:2]))
for i in range(3):
    temp=data[i]
    print(temp)
    list.extend(temp[:2])
print(temp)
print(list)
