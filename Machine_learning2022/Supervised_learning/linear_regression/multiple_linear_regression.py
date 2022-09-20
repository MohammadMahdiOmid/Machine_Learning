import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MultiplrRegression:
    def __init__(self, data, theta_0, theta_1, theta_2):
        # extracting data
        self.data = data
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        my_x = data[['Volume', 'Weight']]
        self.y = data['CO2']
        # to scaling data
        # average = my_x.mean()
        # sigma = my_x.std()
        # self.x = (my_x - average) / sigma

        scale = StandardScaler()
        self.x = scale.fit_transform(my_x)
        self.gradient_descent()

    # Hypothesis Function With 2 variable
    def hypothesis(self):
        return theta_0 + (self.theta_1 * self.x[:, 0]) + (self.theta_2 * self.x[:, 1])

    # Cost Function
    def cost_function(self):
        # To finding best parameters
        j_theta = 0.5 * ((self.hypothesis() - self.y) ** 2).mean()
        return j_theta

    def gradient_descent(self):
        cost = []
        # Hyperparameter
        alpha = 100e-3
        # Random theta to start
        print("Begining theta_0 is:", self.theta_0)
        print("Begining theta_1 is:", self.theta_1)
        print("Begining theta_2 is:", self.theta_2)

        for i in range(100):
            # # For derivation
            dtheta0 = (self.hypothesis() - self.y)
            dtheta1 = ((self.hypothesis() - self.y) * self.x[:, 0])
            dtheta2 = ((self.hypothesis() - self.y) * self.x[:, 1])
            cost.append(self.cost_function())

            # Simultaneous update
            self.theta_0 -= alpha * dtheta0.mean()
            self.theta_1 -= alpha * dtheta1.mean()
            self.theta_2 -= alpha * dtheta2.mean()

        print("Cost theta_0 is:\n", self.theta_0)
        print("Cost theta_1 is:\n", self.theta_1)
        print("Cost theta_2 is:\n", self.theta_2)

        y_predict = self.hypothesis()
        self.plotting(self.x, y_predict, cost)
        # self.test_model(self.theta_0, self.theta_1, self.theta_2)

    def test_model(self, theta_0, theta_1, theta_2):
        # average = self.x.mean()
        # sigma = self.x.std()
        # normal_x = (self.x - average) / sigma
        # xtest_normal=([[1.6,1390]]-average)/sigma
        scale = StandardScaler()
        scaled = scale.fit_transform([[2400, 1.3]])
        y_predict = self.hypothesis(theta_0, theta_1, theta_2, scaled)
        print("The prediction of {0} is : {1}".format(scaled, y_predict))

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
    theta_0 = np.random.randn()
    theta_1 = np.random.randn()
    theta_2 = np.random.randn()
    data = pd.read_csv("data/data.csv")
    new_object = MultiplrRegression(data, theta_0, theta_1, theta_2)
