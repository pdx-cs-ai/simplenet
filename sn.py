#!/usr/bin/python3

# https://medium.com/typeme/
#  lets-code-a-neural-network-from-scratch-part-3-87e23adbe4b6

import random

a = 0.01

def lxor(x1, x2):
    return x1 ^ x2

def land(x1, x2):
    return x1 and x2

bfun = lxor

def instances(n, f):
    def instance():
        xs = [random.randrange(2) for _ in range(2)]
        return [f(*xs), xs]
    return [instance() for _ in range(n)]

def squash(h):
    if h >= 0:
        return h
    return 0.1 * h

def dsquash(h):
    if h >= 0:
        return 1
    return 0.1

class Perceptron(object):

    def __init__(self):
        self.ws = [0.1 + 0.9 * random.random() for _ in range(2)]
        self.bias = 0

    def h(self, xs):
        t = 0
        for i in range(2):
            t += self.ws[i] * xs[i]
        t += self.bias
        return t

    def y(self, xs):
        h = self.h(xs)
        self.yy = squash(h)
        return self.yy

    def train(self, c, xs):
        c0 = self.y(xs)
        e = c - c0
        for i in range(2):
            self.ws[i] += a * xs[i] * e
        self.bias += a * e

    def backprop(self, xs=None, preds=None):
        delta = a * self.ee * dsquash(self.yy)
        if preds != None:
            for i in range(2):
                preds[i].ee = 0
        for i in range(2):
            if preds == None:
                self.ws[i] += xs[i] * delta
            else:
                preds[i].ee += self.ws[i] * self.ee
                self.ws[i] += preds[i].yy * delta
            self.bias += delta

    def print_ws(self):
        print(self.ws, self.bias)

class Net(object):
    def __init__(self):
        self.l1 = [Perceptron() for _ in range(2)]
        self.l2 = Perceptron()
    
    def y(self, xs):
        ys = [p.y(xs) for p in self.l1]
        return self.l2.y(ys)
        
    def train(self, c, xs):
        y = self.y(xs)
        self.l2.ee = c - y
        self.l2.backprop(preds=self.l1)
        for p in self.l1:
            p.backprop(xs=xs)

    def print_ws(self):
        for p in self.l1:
            p.print_ws()
        self.l2.print_ws()

net = Net()

train = instances(1000, bfun)
for _ in range(1000):
    for c, xs in train:
        net.train(c, xs)

net.print_ws()
print()

ttab = [[0, 0], [0, 1], [1, 0], [1, 1]]
clas = [(bfun(*xs), xs) for xs in ttab]
for c, xs in clas:
    y = net.y(xs)
    print(xs, c, int(y > 0.5), y)
