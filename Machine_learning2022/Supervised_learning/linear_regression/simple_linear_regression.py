import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,data):

        self.X=data[:,0]
        self.y=data[:,1]


    def change_dimention(self,x,y):
        print(x.shape)
        print(y.shape)

        x=x[:,None]
        y=y[:,None]
        print(x.shape)
        print(y.shape)

        self.horizontal_concatination(x,y)

    def horizontal_concatination(self,x,y):
        x=np.concatenate((np.ones_like(x),x),axis=1)
        y=np.concatenate((np.ones_like(y),y),axis=1)

        print(x)
        print(y)


    def hypothesis(self):
        theta0=0.5
        theta1=0.5
        h=theta0+(theta1*self.X)

        return h


    def cost_function(self):
        Jtheta=0.5*(sum(self.hypothesis()-self.y)**2)
        return Jtheta
    def demonstrate(self):
        plt.figure(figsize=(10,6))
        plt.title("Home Prediction",color="b")
        plt.xlabel("Size Of Home",color="b")
        plt.ylabel("Price Of Home",color="b")
        plt.scatter(self.X,self.y,marker='*')

        plt.plot(self.X,self.hypothesis(),'r')
        plt.show()

    def gradient_descent(self):
        pass


if __name__ == '__main__':
    data = np.genfromtxt('data/house_price.txt', delimiter=',')
    print(data)
    obj1=LinearRegression(data)
    obj1.demonstrate()