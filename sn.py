#!/usr/bin/python3

# https://medium.com/typeme/
#  lets-code-a-neural-network-from-scratch-part-3-87e23adbe4b6

import random, sys

# Learning rate.
a = 0.01

# Boolean xor.
def lxor(x1, x2):
    return x1 ^ x2

# Boolean and.
def land(x1, x2):
    return x1 and x2

# Boolean or.
def lor(x1, x2):
    return x1 or x2

bfuns = {
    "xor": lxor,
    "and": land,
    "or": lor,
}

# Function to learn.
bfun = bfuns[sys.argv[1]]

# Noise in output.
noise = 0
if len(sys.argv) > 3:
    noise = float(sys.argv[3])

# Generate n instances of the function f.
def instances(n, f):
    def instance():
        xs = [random.randrange(2) for _ in range(2)]
        y = int(f(*xs))
        if random.random() < noise:
            y = int(not y)
        return [y, xs]
    return [instance() for _ in range(n)]

# "Leaky ReLU" activation function. 
def squash(h):
    if h >= 0:
        return h
    return 0.1 * h

# Derivative of activation function.
def dsquash(h):
    if h >= 0:
        return 1
    return 0.1

# Single Perceptron.
class Perceptron(object):

    # Initialize with random positive weights and fixed
    # bias.
    def __init__(self):
        self.ws = [0.1 + 0.9 * random.random() for _ in range(2)]
        self.bias = 0.5

    # Compute weighted sum of inputs xs and bias.
    def h(self, xs):
        t = 0
        for i in range(2):
            t += self.ws[i] * xs[i]
        t += self.bias
        return t

    # Compute output for input xs. Remember output
    # for possible neural net learner.
    def y(self, xs):
        h = self.h(xs)
        self.yy = squash(h)
        return self.yy

    # Train single perceptron with instance with features xs
    # and class c.
    def train(self, c, xs):
        c0 = self.y(xs)
        e = c - c0
        for i in range(2):
            self.ws[i] += a * xs[i] * e
        self.bias += a * e

    # Train a net perceptron with input xs or predecessor
    # layer preds. In the latter case, assume that y() has
    # been called on each perceptron in preds to establish
    # its output.
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

    # Print weights including bias.
    def print_ws(self):
        print(self.ws, self.bias)

# Neural net consisting of a 2-perceptron
# hidden layer and a one-perceptron output layer.
class Net(object):

    # Set up the net.
    def __init__(self):
        self.l1 = [Perceptron() for _ in range(2)]
        self.l2 = Perceptron()
    
    # Compute the net output from its inputs.
    def y(self, xs):
        ys = [p.y(xs) for p in self.l1]
        return self.l2.y(ys)
        
    # Train the net via backpropagation.
    def train(self, c, xs):
        y = self.y(xs)
        self.l2.ee = c - y
        self.l2.backprop(preds=self.l1)
        for p in self.l1:
            p.backprop(xs=xs)

    # Print weights of all perceptrons in net.
    def print_ws(self):
        for p in self.l1:
            p.print_ws()
        self.l2.print_ws()

# Learner to use for experiments.
nets = {
    "perceptron": Perceptron,
    "net": Net,
}
net = nets[sys.argv[2]]()

# Train the net on 1000 random instances
# of the target function.
train = instances(1000, bfun)
for _ in range(1000):
    for c, xs in train:
        net.train(c, xs)

# Show the weights of the resulting net.
net.print_ws()
print()

# Show the learned output for each possible
# input, both booleanized and in raw form.
ttab = [[0, 0], [0, 1], [1, 0], [1, 1]]
clas = [(bfun(*xs), xs) for xs in ttab]
for c, xs in clas:
    y = net.y(xs)
    print(xs, c, int(y > 0.5), y)
