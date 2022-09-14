import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        pass

    # create random number
    def create_data(self):
        x=np.random.randint(100,size=50)
        print("x is:",x)

        y=np.random.randint(100,size=50)
        print("y is:",y)


        self.change_dimention(x,y)

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
        pass


    def demonstrate(self,x,y):
        pass

    def gradient_descent(self):
        pass


if __name__ == '__main__':
    obj1=LinearRegression()
    obj1.create_data()