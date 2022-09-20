import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MultiplrRegression:
    def __init__(self, data):
        # extracting data
        self.data = data
        self.x = data[['Volume', 'Weight']]
        self.y = data['CO2']
        self.print(self.x)
        self.gradient_descent()

    def print(self, x):
        print(x['Volume'])
        print(x['Weight'])

    # Hypothesis Function With 2 variable
    def hypothesis(self, theta_0, theta_1, theta_2):
        return theta_0 + theta_1 * self.x['Volume'] + theta_2 * self.x['Weight']

    # Cost Function
    def cost_function(self, theta_0, theta_1, theta_2):
        # To Demonstrate Loss Function
        cost = []
        # To finding best parameters
        j_theta = 0.5 * ((self.hypothesis(theta_0, theta_1, theta_2)-self.y) ** 2).mean()
        return j_theta

    def gradient_descent(self):
        # Hyperparameter
        alpha=5e-3
        # Random theta to start
        theta_0=np.random.randn()
        theta_1=np.random.randn()
        theta_2=np.random.randn()
        print("Begining theta_0 is:",theta_0)
        print("Begining theta_1 is:",theta_1)
        print("Begining theta_2 is:",theta_2)



if __name__ == '__main__':
    data = pd.read_csv("data/data.csv")
    new_object = MultiplrRegression(data)
