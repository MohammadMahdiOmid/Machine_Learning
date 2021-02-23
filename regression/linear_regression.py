import numpy as np
import  matplotlib.pyplot as plt

#get data

data=np.genfromtxt('data/house_price.txt',delimiter=',')
print(data)

#column 0 from matrix of data
x=data[:,0]
print(x.shape)
print(x)


#column 1 from matrix of data
y=data[:,1]
print(y.shape)
print(y)


#show data with matplot
fig=plt.figure(figsize=(10,6))
plt.scatter(x,y,s=90,c='b',marker='*')
plt.show()
fig.savefig('data.png')