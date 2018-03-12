import numpy as np


class bandit():

    def __init__(self, m):

        self.m = m
        self.mean = 0
        self.N = 0

    def pull(self):

        
