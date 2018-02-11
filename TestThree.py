import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 1],
              [1, 1, 0, 1, 0, 0],
              [1, 1, 1, 1, 1, 1]])

Y = np.array([-1, 1, 1, -1])

W = (np.random.random(6) - 0.5) * 2

lr = 0.11

n = 0

O = 0


def update():
    global X, Y, W, lr, n
    n += 1
    O = np.dot(X, W.T)
    # print("O: ", O)
    # print("Y-O.T : ", Y - O.T)
    W_C = lr * ((Y - O.T).dot(X)) / X.shape[0]
    # print("W_C * Y - O.T: ", (Y - O.T).dot(X))
    # print("W_C * Y - O.T: ", X.dot(Y - O.T))
    W = W + W_C


for _ in range(1000):
    update()
    # print("W: ", W)
    # print("n: ", n)

x1 = [0, 1]
y1 = [1, 0]

x2 = [0, 1]
y2 = [0, 1]


def calculate(x, root):
    a = W[5]
    b = W[2] + x * W[4]
    c = W[0] + x * W[1] + x * x * W[3]
    if root == 1:
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    elif root == 2:
        return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


xdata = np.linspace(-1, 2)
print(xdata)
plt.figure()
plt.plot(xdata, calculate(xdata, 1), 'r')
plt.plot(xdata, calculate(xdata, 2), 'r')
plt.plot(x1, y1, 'bo')
plt.plot(x2, y2, 'yo')
plt.show()
