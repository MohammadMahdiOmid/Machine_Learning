import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MultiplrRegression:
    def __init__(self, data):
        # extracting data
        self.data = data
        my_x = data[['Volume', 'Weight']]
        self.y = data['CO2']
        # to scaling data
        scale = StandardScaler()
        self.x = scale.fit_transform(my_x)
        self.print(self.x)
        self.gradient_descent()

    def print(self, x):
        print(x[:,0])
        # print(x['Weight'])

    # Hypothesis Function With 2 variable
    def hypothesis(self, theta_0, theta_1, theta_2):
        return theta_0 + (theta_1 * self.x[:,0]) + (theta_2 * self.x[:,1])

    # Cost Function
    def cost_function(self, theta_0, theta_1, theta_2):
        # To Demonstrate Loss Function
        cost = []
        # To finding best parameters
        j_theta = 0.5 * ((self.hypothesis(theta_0, theta_1, theta_2) - self.y) ** 2).mean()
        return j_theta

    def gradient_descent(self):
        cost = []
        # Hyperparameter
        alpha = 100e-3
        # Random theta to start
        theta_0 = np.random.randn()
        theta_1 = np.random.randn()
        theta_2 = np.random.randn()
        print("Begining theta_0 is:", theta_0)
        print("Begining theta_1 is:", theta_1)
        print("Begining theta_2 is:", theta_2)

        for i in range(100):
            # # For derivation
            dtheta0 = (self.hypothesis(theta_0, theta_1, theta_2) - self.y)
            dtheta1 = ((self.hypothesis(theta_0, theta_1, theta_2) - self.y) * self.x[:,0])
            dtheta2 = ((self.hypothesis(theta_0, theta_1, theta_2) - self.y) * self.x[:,1])
            cost.append(self.cost_function(theta_0, theta_1, theta_2))


            # Simultaneous update
            theta_0 -= alpha * dtheta0.mean()
            theta_1 -= alpha * dtheta1.mean()
            theta_2 -= alpha * dtheta2.mean()

        print("Cost theta_0 is:\n", theta_0)
        print("Cost theta_1 is:\n", theta_1)
        print("Cost theta_2 is:\n", theta_2)

        y_predict = self.hypothesis(theta_0, theta_1, theta_2)
        self.plotting(self.x, y_predict, cost)

    def plotting(self, normal_x, y_prediction, cost):
        # plotting real data
        # plt.figure(figsize=(8, 5))
        # plt.scatter(normal_x[:,0],normal_x[:,1], self.y, c='g', marker='*')
        # # plt.xlabel("Home Size", c='g')
        # # plt.ylabel("Home Price", c='g')
        # plt.title("Prediction Home Price", c='g')
        # # plotting hypothesis function
        # plt.plot(normal_x[:,0],normal_x[:,1], y_prediction, c='r')
        # plt.show()
        # plot cost function
        print("Cost values are :", cost)
        plt.figure(figsize=(8, 6))
        plt.title("Cost Function", c='r')
        plt.plot(cost, c='r')

        plt.show()


if __name__ == '__main__':
    data = pd.read_csv("data/data.csv")
    new_object = MultiplrRegression(data)
