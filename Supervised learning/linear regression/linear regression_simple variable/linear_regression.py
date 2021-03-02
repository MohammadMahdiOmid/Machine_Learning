import numpy as np
import  matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='html5')

#get data

data=np.genfromtxt('data/house_price.txt',delimiter=',')
print(data)

#column 0 from matrix of data
real_data_x= data[:, 0]
print(real_data_x.shape)
print(real_data_x)


#column 1 from matrix of data
real_data_y= data[:, 1]
print(real_data_y.shape)
print(real_data_y)


#show data with matplot
fig=plt.figure(figsize=(10,6))
plt.scatter(real_data_x, real_data_y, s=90, c='b', marker='*')
plt.xlabel('size(q.feet)')
plt.ylabel('price(x100$)')
plt.title('Houses Dataset')
plt.show()
fig.savefig('data.png')

# Hypothesis

theta = np.array([[1000.],[0]])
print(theta)

random_x=np.arange(500, 3000, 100)
print(random_x)

#convert to 2DX
new2dx= random_x[:, None]
print(new2dx.shape)

# Horizental Concatination
new2dx=np.concatenate((np.ones_like(new2dx),new2dx),axis=1)
print(new2dx)

h=new2dx @ theta
print('h(x) is :\n',h)

#show Hypothesis
fig=plt.figure(figsize=(10,6))
plt.scatter(real_data_x, real_data_y, s=90, c='b', marker='*')
plt.plot(new2dx,h,'k')
plt.xlabel('size (Sq. feet)')
plt.ylabel('price (x100 $)')
plt.title('House Dataset')
plt.xlim(500, 3000)
plt.ylim(300, 5200)
plt.show()
fig.savefig('hypothesis without GD.png')

##normalization
real_data_x = data[:,0]
real_data_y = data[:,1]

avg = real_data_x.mean()
sigma = real_data_x.std()
x_normal = (real_data_x-avg) / sigma
print('data normal :\n',x_normal)

##Learning rate
alpha=100e-3
print('alpha is :\n',alpha)

##INITIALIZE PARAMETERS RANDOMLY
theta_0=np.random.randn()
theta_1=np.random.randn()
print("The value of theta_0 is {0} and The value of theta_1 is {1}".format(theta_0,theta_1))


costs = []

def hypothesis(theta_0,theta_1,x_normal):
    return theta_0 + theta_1 *x_normal


def cost_function(y_predict,real_data_y_true):
    return 0.5 * ((y_predict-real_data_y_true) **2).mean()


def update_step():
    global theta_0,theta_1,costs

    #predict
    predict_y = hypothesis(theta_0,theta_1,x_normal)
    costs.append(cost_function(predict_y,real_data_y))

    #compute gradient descent
    dtheta_0 = predict_y-real_data_y
    dtheta_1 = x_normal*dtheta_0

    #update oarameters
    theta_0 -= alpha * dtheta_0.mean()
    theta_1 -= alpha * dtheta_1.mean()


for i in range(100):
    update_step()


#result plot
fig= plt.figure(figsize=(10,6))
plt.scatter(real_data_x,real_data_y,c='b',s=90 , marker='*')
plt.plot(real_data_x,hypothesis(theta_0,theta_1,x_normal) , 'k')
plt.title("Houses Dataset")
plt.xlabel('size (Sq. feet)')
plt.ylabel('price (x100 $)')
plt.xlim(500, 3000)
plt.ylim(300, 5200)
plt.show()
fig.savefig('hypothesis with GD')


#loss function
fig=plt.figure(figsize=(12, 6))
plt.plot(costs,'k')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('loss fonction')
plt.show()
fig.savefig('loss function')


##test

x_test=1700

x_test_normal=(x_test-avg) / sigma

y_test = hypothesis(theta_0,theta_1,x_test_normal)

print("for size {} the predicted price is : {:.2f} $".format(x_test,y_test))

