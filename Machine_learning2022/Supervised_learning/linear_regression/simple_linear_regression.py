import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, data):
        # initialize data
        self.X = data[:, 0]
        self.y = data[:, 1]

    def hypothesis(self, theta0, theta1, x):
        # creating hypothesis function for prediction
        h = theta0 + theta1 * x
        return h

    def cost_function(self, prediction_y, y):
        # create cost for save value of cost function result to demonstrate it
        cost = []
        Jtheta = 0.5 * ((prediction_y - y) ** 2).mean()
        cost.append(Jtheta)
        print("Cost is:", cost)
        return cost

    def gradient_descent(self):
        global theta0, theta1
        costs = []
        # Normalization
        mu = self.X.mean()
        sigma = self.X.std()
        xn = (self.X - mu) / sigma
        print("x normal is:", xn)

        alpha = 5e-2
        theta0 = np.random.randn()
        theta1 = np.random.randn()
        print("Initial theta_0 is: {0} , theta_1 is: {1}, alpha is: {2}".format(theta0, theta1, alpha))

        for i in range(180):
            prediction_y = self.hypothesis(theta0, theta1, xn)
            costs.append(self.cost_function(prediction_y, self.y))

            dtheta0 = (prediction_y - self.y)
            dtheta1 = dtheta0 * xn

            theta0 -= alpha * dtheta0.mean()
            theta1 -= alpha * dtheta1.mean()

        print(theta0)
        print(theta1)

        # To plotting result
        y_pre = self.hypothesis(theta0, theta1, xn)
        self.plotting(xn, y_pre, costs)

    def plotting(self, normal_x, y_prediction, cost):
        # plotting real data
        plt.figure(figsize=(10, 6))
        plt.scatter(normal_x, self.y, c='g', marker='*')
        plt.xlabel("Home Size", c='b')
        plt.ylabel("Home Price", c='b')
        plt.title("Prediction Hom", c='b')
        # plotting hypothesis function
        plt.plot(normal_x, y_prediction, c='r')
        plt.show()
        # plot cost function
        print("Cost values are :", cost)
        plt.figure(figsize=(10, 8))
        plt.plot(cost, c='r')
        plt.show()


if __name__ == '__main__':
    data = np.genfromtxt('data/house_price.txt', delimiter=',')
    print(data)
    new_object = LinearRegression(data)
    new_object.gradient_descent()
