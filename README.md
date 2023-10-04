# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required. 
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: THRIKESWAR P
RegisterNumber:212222230162
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter = ',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J,grad
    
X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    grad = np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta  = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min, x_max = X[:, 0].min() - 1,X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max, 0.1),np.arange(y_min,y_max, 0.1))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot = np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)),X))
    prob = sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X) == y)
```

## Output:
![1](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/04447526-9230-4d43-a37b-153034f4dc14)
![2](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/0b9a727c-ec6d-4e70-866c-d3f55049c042)
![3](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/8ad779ea-5490-4839-9aac-92fe0e0853de)
![4](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/09505d40-a04c-462b-9f3d-e5abf89c1177)
![5](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/429cf876-9731-41fc-83e6-18f471705eb4)
![6](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/30e2985c-ab4d-4597-84ec-5e0f690dcc96)
![7](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/9f14bfc9-008d-44d9-b3ae-1eb8e418328e)
![8](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/542173b5-07cf-43ab-a9cd-21416db5f273)
![9](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/25abce13-7b43-424d-b9ca-61c5320e62a0)
![10](https://github.com/Naveensrinivasan07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119475891/6d585b69-e29d-425c-878d-1e08b5c63a37)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
