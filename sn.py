#!/usr/bin/python3

import random

a = 0.0001

def instances(n, f):
    def instance():
        xs = [random.randrange(2) for _ in range(2)]
        return [2 * f(*xs) - 1, [2 * x - 1 for x in xs]]
    return [instance() for _ in range(n)]

def lxor(x1, x2):
    return x1 ^ x2

def land(x1, x2):
    return x1 and x2

class Perceptron(object):

    def __init__(self):
        self.ws = [2 * random.random() - 1 for _ in range(2)]
        self.bias = 0

    def h(self, xs):
        t = 0
        for i in range(2):
            t += self.ws[i] * xs[i]
        t += self.bias
        return t

    def y(self, xs):
        h0 = self.h(xs)
        return 2 * int(h0 > 0) -1

    def train(self, c, xs):
        c0 = self.y(xs)
        e = c - c0
        for i in range(2):
            self.ws[i] += a * xs[i] * e
        self.bias += a * e

pct = Perceptron()

train = instances(1000, land)
for _ in range(1000):
    for c, xs in train:
        pct.train(c, xs)

clas = instances(10, land)
for c, xs in clas:
    print(xs, c, pct.y(xs))
