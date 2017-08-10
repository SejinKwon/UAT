# opensource file을 기반으로 나의 code를 만들어보자.
# Mission : graph가 있는 image를 read하여 함수를 읽고, 이 함수를 나타내는 NN을 제작한다.(Universal Approximation Theorem)
# Author : Sejin Kwon
# Date : 1st draft : 2017.08.08

# Model : 1-hidden layer NN

# 1.Required functions

# Reading Image and extracting points
# feature(x) scaling
# function value(y) scaling
# Weights(theta) allocation randomly
# Feed-forward algorithm
# back propagation
# learning part
# plot the learning curve, graph
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import sys


# Reading Image and extracting points

def read_image(img_location):
    # aim : 2d image를 읽어 [0~numX]에 대한 y값을 반환한다.
    # input : img_location(img 저장경로)
    # output : x:그림의 x값(row length)vector(numX*1) /y:그림에 있는 함수의 y값 vector (numX*1)

    # Load image
    img = color.rgb2gray(io.imread(img_location))
    (numY, numX) = np.shape(img)

    # find black points
    dots = np.where(img < 0.2)

    # temp_f[0][i] : sum of y values at x=i/temp_f[1][i] : num of points at x=i
    temp_f = np.zeros((2, numX))
    for i in range(len(dots[1])):
        temp_f[0][dots[1][i]] += dots[0][i]
        temp_f[1][dots[1][i]] += 1

    # f[0] : x값(column index), f[1] : y값(row index)
    x = np.zeros((numX, 1))
    f = np.copy(x)
    # x값 할당
    for i in range(numX):
        x[i][0] = i

    #f값 할당
    for i in range(len(temp_f[0])):
        if (temp_f[1][i] != 0):
            f[i][0] = temp_f[0][i] / temp_f[1][i]
         # f값은 기준이 행렬이므로, 증가 방향으로 반대로 바꾸어야함
        f[i][0] = numY - f[i][0]

    return (x, f, numY)

# f scaling
def f_scaling(f, numY, maxY):
    f = maxY*f/numY
    return f

# Extract the custom set
def ext_custom_set(x,f,n): #n개의 dataset을 추출한다.(n<numX)
    x_custom = np.zeros([n,1])
    f_custom = np.copy(x_custom)

    m = x[len(x)-1][0]

    for i in range(n):
        x_custom[i][0] = int(i*m/(n-1))
        f_custom[i][0] = f[int(x_custom[i][0])][0]

    return (x_custom, f_custom)



# feature(x) scaling
def feature_scaling(x):
    mean_x = np.mean(x)
    delta_x = np.max(x) - np.min(x)

    x =  10*(x - mean_x) / delta_x
    return x


# Unroll parameter
def unroll_parameter(theta1, theta2):
    return np.append(theta1, theta2)


def roll_parameter(theta, num_inputlayer, num_hiddenlayer, num_outputlayer):

    theta1 = theta[0:num_hiddenlayer * (num_inputlayer + 1)]
    theta2 = theta[num_hiddenlayer * (num_inputlayer + 1):]
    theta1 = np.reshape(theta1, (num_hiddenlayer, num_inputlayer+1))
    theta2 = np.reshape(theta2, (num_outputlayer, num_hiddenlayer+1))
    return (theta1, theta2)


# Weights(theta) allocation randomly
def random_init(theta):
    eps = 10**-2
    m = len(theta)
    theta = eps * (2.0 * np.random.rand(m)-1)
    return theta


# sigmoid function
Z_MAX = 100
def sigmoid(z):
    z = np.clip(z, -Z_MAX, Z_MAX)
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(z):

    return z * (1.0 - z)


# Feed-forward
def feed_forward(x, theta, num_inputlayer, num_hiddenlayer, num_outputlayer):
    (theta1, theta2) = roll_parameter(theta,num_inputlayer, num_hiddenlayer, num_outputlayer)

    a1 = np.vstack(([1.0], x))  # bias 노드 추가
    z2 = np.dot(theta1, a1)
    a2 = sigmoid(z2)
    a2 = np.vstack(([1.0], a2))  # bias 노드 추가
    a3 = np.dot(theta2, a2)

    return (a1, z2, a2, a3)

def cost(a3, y):
    return np.sum((a3 - y) ** 2 / 2)

def back_propagation(a1, z2, a2, a3, Del, theta,num_inputlayer, num_hiddenlayer, num_outputlayer, y):
    (new_theta1, new_theta2) = roll_parameter(theta,num_inputlayer, num_hiddenlayer, num_outputlayer)

    # 마지막 레이어에 activation 함수 있을때
    # delta3 = (a3 - y)*d_sigmoid(a3)
    # 없을때
    delta3 = (a3 - y)
    delta2 = np.dot(new_theta2.T, delta3) * d_sigmoid(a2)

    '''
    Del1 = np.zeros((num_hiddenlayer,num_inputlayer+1))
    Del2 = np.zeros((num_outputlayer,num_hiddenlayer+1))
    '''
    (Del1, Del2) = roll_parameter(Del,num_inputlayer, num_hiddenlayer, num_outputlayer)
    Del2 += np.dot(delta3, a2.T)
    Del1 += np.dot(delta2[1:, :], a1.T)


    return np.append(Del1.flatten(), Del2.flatten())

# Gradient Checking
def grad_checking(x, theta, Del, y,  num_inputlayer, num_hiddenlayer, num_outputlayer):
    #Numerical Gradient
    eps = 10 ** (-4)
    num_grad = np.copy(Del)
    sample_x = x
    for i in range(len(theta)):
        theta[i] += eps
        (a1, z2, a2, a3) = feed_forward(sample_x, theta, num_inputlayer, num_hiddenlayer, num_outputlayer)
        J_plus  = cost(a3, y)
        theta[i] -= 2 * eps
        (a1, z2, a2, a3) = feed_forward(sample_x, theta, num_inputlayer, num_hiddenlayer, num_outputlayer)
        J_min = cost(a3, y)
        theta[i] += eps
        num_grad[i] = (J_plus - J_min) / (2 * eps)

    if (sum((num_grad-Del)**2/len(Del)) <= 10**-2):
        return True
    else:
        return False

# Main
img_location = "sample_1.PNG"
# for printing all data on the console
np.set_printoptions(threshold=np.nan)

# f[0] : x값(column index), f[1] : y값(row index)
(x, f, numY) = read_image(img_location)

# f Scaling
maxY = 100
f = f_scaling(f,numY,maxY)



#Prepare training set, test set
n_train=40
(x_train, f_train) = ext_custom_set(x,f,n_train)

n_test=200
(x_test, f_test) = ext_custom_set(x,f,n_test)

# Feature Scaling
x = feature_scaling(x)
x_train = feature_scaling(x_train)
x_test = feature_scaling(x_test)

# Define the number of nodes of each layer
num_inputlayer = 1
num_outputlayer = 1
num_hiddenlayer = 100

# Define Weights(theta)
theta = np.zeros(num_hiddenlayer * (num_inputlayer + 1) + num_outputlayer * (num_hiddenlayer + 1))

# Random initialization
theta = random_init(theta)

#Gradient Checking for i=1
(a1, z2, a2, a3) = feed_forward(x_train[0][0], theta, num_inputlayer, num_hiddenlayer, num_outputlayer)
Del = np.zeros(num_hiddenlayer * (num_inputlayer + 1) + num_outputlayer * (num_hiddenlayer + 1))
Del = back_propagation(a1, z2, a2, a3, Del, theta,num_inputlayer, num_hiddenlayer, num_outputlayer, f_train[0][0])

print("Gradient Checking.....")
if grad_checking(x_train[0][0], theta, Del, f_train[0][0],  num_inputlayer, num_hiddenlayer, num_outputlayer)==False:
    sys.exit("ㅋㅋ 다시만들게나..")
else:
    print("OK")

#Learning
print("Learning Start.....")
#cost, Delta, Learning Rate
LR = 0.03
n_iter=0
PLT_PAUSETIME = 0.1
plt.ion()
plt.figure(figsize=(160, 120))
while n_iter<=40000:
    n_iter += 1
    cost_train=0.0
    Del = np.zeros(num_hiddenlayer * (num_inputlayer + 1) + num_outputlayer * (num_hiddenlayer + 1))


    #training
    for i in range(n_train):
        (a1, z2, a2, a3) = feed_forward(x_train[i][0], theta, num_inputlayer, num_hiddenlayer, num_outputlayer)
        cost_train += cost(a3, f_train[i][0])
        Del = back_propagation(a1, z2, a2, a3, Del, theta, num_inputlayer, num_hiddenlayer, num_outputlayer,
                               f_train[i][0])

    #Del 계산
    Del = Del/n_train
    #cost 계산
    cost_train = cost_train/n_train
    error_train = cost
    #theta update
    theta -= LR*Del
    print(n_iter)
    print(cost_train)



print("------------------------------Learning over----------------------------------------")


while True:
    A3 = np.zeros((n_test, 1))
    for i in range(n_test):
        (a1, z2, a2, a3) = feed_forward(x_test[i][0], theta, num_inputlayer, num_hiddenlayer, num_outputlayer)
        A3[i][0] = a3

    plt.clf()
    ax = plt.subplot(111)
    ax.axis([x_train[0][0], x_train[n_train - 1][0], 0, maxY * 1.2])
    ax.text(0.5, 0.9, "t(X)=" + img_location,
            fontsize=50, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.8, "Iteration:" + str(n_iter),
            fontsize=30, ha='center', transform=ax.transAxes)

    ax.plot(x, f,
            c="green", marker="o", markersize=5.0, linewidth=5.0)
    ax.plot(x_train, f_train,
            c="blue", marker="d", markersize=20.0, linestyle='None')

    ax.plot(x_test, A3,
            c="red", marker="o", markersize=10.0, linestyle='None')

    plt.draw()
    plt.pause(PLT_PAUSETIME)