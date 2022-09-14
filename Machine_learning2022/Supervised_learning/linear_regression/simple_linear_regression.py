import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, data):
        self.X = data[:, 0]
        self.y = data[:, 1]

    def change_dimention(self, x, y):
        print(x.shape)
        print(y.shape)

        x = x[:, None]
        y = y[:, None]
        print(x.shape)
        print(y.shape)

        self.horizontal_concatination(x, y)

    def horizontal_concatination(self, x, y):
        x = np.concatenate((np.ones_like(x), x), axis=1)
        y = np.concatenate((np.ones_like(y), y), axis=1)

        print(x)
        print(y)

    def hypothesis(self, theta0, theta1, x):
        h = theta0 + (theta1 * x)
        return h

    def cost_function(self, prediction_y):
        cost = []
        Jtheta = 0.5 * (sum(prediction_y - self.y) ** 2)
        cost.append(Jtheta)
        return Jtheta

    def demonstrate(self):
        plt.figure(figsize=(10, 6))
        plt.title("Home Prediction", color="b")
        plt.xlabel("Size Of Home", color="b")
        plt.ylabel("Price Of Home", color="b")
        plt.scatter(self.X, self.y, marker='*')

        # plt.plot(self.,'r')
        plt.show()

        self.gradient_descent()

    def gradient_descent(self):
        # Normalization
        mu = self.X.mean()
        sigma = self.X.std()
        xn = (self.X - mu) / sigma
        print("x normal is:", xn)

        alpha = 5e-3
        theta0 = np.random.randn()
        theta1 = np.random.randn()

        prediction_y = self.hypothesis(theta0, theta1, xn)
        self.cost_function(prediction_y)

        dtheta0 = (prediction_y - self.y)
        dtheta1 = dtheta0 * xn

        theta0 -= alpha * dtheta0.mean()
        theta1 -= alpha * dtheta1.mean()

        print(theta0)
        print(theta1)


if __name__ == '__main__':
    data = np.genfromtxt('data/house_price.txt', delimiter=',')
    print(data)
    obj1 = LinearRegression(data)
    obj1.demonstrate()
