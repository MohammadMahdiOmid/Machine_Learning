import numpy as np
import matplotlib.pyplot as plt
class LinearRegression:
    def __init__(self, data):
        # initialize data
        self.x = data[:, 0]
        self.y = data[:, 1]

    def hypothesis(self, theta0, theta1, x):
        # creating hypothesis function for prediction
        h = theta0 + theta1 * x
        return h

    def cost_function(self, prediction_y, y):
        # create cost for save value of cost function result to demonstrate it
        cost = []
        # cost function
        Jtheta = 0.5 * ((prediction_y - y) ** 2).mean()
        cost.append(Jtheta)
        print("Cost is:", cost)
        return cost

    def gradient_descent(self):
        costs = []
        # Normalization
        average = self.x.mean()
        sigma = self.x.std()
        normal_x = (self.x - average) / sigma
        print("x normal is:", normal_x)
        # HyperParameter
        alpha =100e-3
        # create random theta_0 and theta_1 for begin
        theta0 = np.random.randn()
        theta1 = np.random.randn()
        print("Initial theta_0 is: {0} , theta_1 is: {1}, alpha is: {2}".format(theta0, theta1, alpha))

        # To train algorithem
        for i in range(100):
            prediction_y = self.hypothesis(theta0, theta1, normal_x)
            costs.append(self.cost_function(prediction_y, self.y))
            # For derivation
            dtheta0 = (prediction_y - self.y)
            dtheta1 = dtheta0 * normal_x
            # Simultaneous update
            theta0 -= alpha * dtheta0.mean()
            theta1 -= alpha * dtheta1.mean()

        print("Theta_0 after training is :",theta0)
        print("Theta_1 after training is :",theta1)

        # To plotting result
        y_pre = self.hypothesis(theta0, theta1, normal_x)
        self.plotting(normal_x, y_pre, costs)

    def plotting(self, normal_x, y_prediction, cost):
        # plotting real data
        plt.figure(figsize=(8, 5))
        plt.scatter(normal_x, self.y, c='g', marker='*',s=100)
        plt.xlabel("Home Size", c='g')
        plt.ylabel("Home Price", c='g')
        plt.title("Prediction Home Price", c='g')
        # plotting hypothesis function
        plt.plot(normal_x, y_prediction, c='r')
        plt.show()
        # plot cost function
        print("Cost values are :", cost)
        plt.figure(figsize=(8, 6))
        plt.title("Cost Function",c='r')
        plt.plot(cost, c='r')

        plt.show()


if __name__ == '__main__':
    data = np.genfromtxt('data/house_price.txt', delimiter=',')
    print(data)
    new_object = LinearRegression(data)
    new_object.gradient_descent()
