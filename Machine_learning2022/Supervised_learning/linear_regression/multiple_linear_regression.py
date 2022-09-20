import numpy as np
import matplotlib.pyplot as plt

class MultiplrRegression:
    def __init__(self,data):
        # extracting data
        self.x=data[:,0]
        self.y=data[:,1]
