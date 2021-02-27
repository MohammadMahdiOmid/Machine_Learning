#imports
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self):
        #keep loss
        self.losses=[]

        self.x=self.load_data()
        self.y=self.load_data()


    #load data
    def load_data(self):

        data=np.genfromtxt("data/house_price.txt",delimiter=',')

        x=data[1:,0]
        y=data[1:,1]

        plt.scatter(x,y,c='r')
        plt.show()

        return x , y

    #plot_hypothesis
    def plot_hypothesis(self,predicions):
        #plot the predicted line
        plt.plot(self.x,predicions,c='r')
        #plot data
        plt.scatter(self.x,self.y)
        plt.show()


    #gradient descent
    def compute_gradient(self,predictions):
        temp=predictions-self.y
        total_loss=0.5 * np.sum(temp**2)

        #compute derivatives
        theta_0=np.sum(temp)
        theta_1=np.sum(temp * self.x)

        return total_loss , theta_0 , theta_1

    def fit(self , learning_rate,epochs=30):
        #initialize the parameters
        a=0.0
        bias=0.0

        for i in range(epochs):
            predictions = (a * self.x) + bias

            loss , theta_0 , theta_1 = self.compute_gradient(predictions)

            #add this epoch loss to the overall loss
            self.losses.append(loss)

            bias += -learning_rate * theta_0
            a += -learning_rate * theta_1

            self.plot_hypothesis(predictions)


            print(f'Epoch {i}, loss {loss}')

        #plot losses over the training
        plt.plot(np.arange(0 ,epochs),self.losses)
        plt.title('loss')
        plt.xlabel("Epochs")
        plt.ylabel('Loss')
        plt.show()

if __name__ == '__main__':
    lr=LinearRegression()
    lr.fit(learning_rate=0.000000005 , epochs=30)

