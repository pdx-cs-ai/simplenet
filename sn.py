#!/usr/bin/python3

import random

a = 0.0001

def lxor(x1, x2):
    return x1 ^ x2

def land(x1, x2):
    return x1 and x2

bfun = land

def instances(n, f):
    def instance():
        xs = [random.randrange(2) for _ in range(2)]
        return [2 * f(*xs) - 1, [2 * x - 1 for x in xs]]
    return [instance() for _ in range(n)]

def squash(h):
    return max(-1, h)

def dsquash(h):
    if h >= -1:
        return 1
    return 0

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
        return squash(self.h(xs))

    def train(self, c, xs):
        c0 = self.y(xs)
        e = c - c0
        for i in range(2):
            if xs[i] >= -1:
                self.ws[i] += a * xs[i] * e
        if self.bias >= -1:
            self.bias += a * e

    def backprop(self, e):
        xxs = []
        for i in range(2):
            xe = dsquash(self.ws[i] * e)
            xxs.append(self.ws[i] + xe)
        return xxs

class Net(object):
    def __init__(self):
        self.l1 = [Perceptron() for _ in range(2)]
        self.l2 = Perceptron()
    
    def y(self, xs):
        ys = [p.y(xs) for p in self.l1]
        return self.l2.y(ys)
        
    def train(self, c, xs):
        y = self.y(xs)
        e = c - y
        xxs = self.l2.backprop(a * e)
        ys = []
        for i in range(2):
            p = self.l1[i]
            p.train(xxs[i], xs)
            ys.append(p.y(xs))
        self.l2.train(c, ys)

net = Net()

train = instances(1000, bfun)
for _ in range(1000):
    for c, xs in train:
        net.train(c, xs)

clas = instances(10, bfun)
for c, xs in clas:
    print(xs, c, net.y(xs))
