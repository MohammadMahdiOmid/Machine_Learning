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
    def print(self,x):
        print(x['Volume'])
        print(x['Weight'])
    def hypothesis(self, theta_0, theta_1, theta_2):
        return theta_0 + theta_1 * self.x['Volume'] + theta_2 * self.x['Weight']


if __name__ == '__main__':
    data = pd.read_csv("data/data.csv")
    new_object = MultiplrRegression(data)
