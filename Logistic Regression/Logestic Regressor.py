# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:11:09 2019

@author: Adak
"""

import numpy as np
import matplotlib.pyplot as plt
class LogisticRegressor:
    def __init__(self,X,y,alpha=0.01,lambdaa=100):
        self.X=X
        self.y=y
        self.alpha=alpha
        self.Theta=np.random.rand(self.X.shape[0],1)
        self.lambdaa=lambdaa
    def plotTrainingSet(self):
        plt.figure()
        for i in range(self.y.shape[1]):
            if self.y[:,i]==0:
                plt.plot(self.X[1,i],self.X[2,i],'bo')
            else:
                plt.plot(self.X[1,i],self.X[2,i],'rx')
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def h(self,x):
        return self.sigmoid(self.Theta.T*x)
    def J(self):
        m=self.y.shape[1]
        cost=0
        for i in range(m):
            cost+=(-self.y[:,i]*(np.log(self.h(self.X[:,i])))-(1-self.y[:,i])*(np.log(1-self.h(self.X[:,i]))))
        cost/=m
        return float(cost)
    def regularizedJ(self):
        m=self.y.shape[1]
        self.Theta=np.matrix(self.Theta)
        cost=0
        for i in range(m):
            cost+=(-self.y[:,i]*(np.log(self.h(self.X[:,i])))-(1-self.y[:,i])*(np.log(1-self.h(self.X[:,i]))))
        cost/=m
        cost+=(self.lambdaa*self.Theta.T*self.Theta)/(2*m)
        return float(cost)
    def regularizedGradientDescent(self):
        m=self.y.shape[1]
        flag=10
        iterations=0
        iterations_list=[0]
        cost_list=[self.regularizedJ()]
        while flag>0.00001:
            iterations+=1
            iterations_list.append(iterations)
            old_cost=self.regularizedJ()
            error=self.h(self.X)-self.y
            error=error.T
            temp=self.X*error
            temp*=(self.alpha/m)
            self.Theta=self.Theta*(1-self.alpha*self.lambdaa/m)-temp
            new_cost=self.regularizedJ()
            cost_list.append(new_cost)
            flag=abs(old_cost-new_cost)
        print('Gradient Descent done in {} iterations'.format(iterations))
        print(self.Theta)
        plt.figure()
        plt.plot(iterations_list,cost_list)
        
    def gradientDescent(self):
        m=self.y.shape[1]
        flag=10
        iterations=0
        iterations_list=[0]
        cost_list=[self.J()]
        while flag>0.00001:
            iterations+=1
            iterations_list.append(iterations)
            old_cost=self.J()
            error=self.h(self.X)-self.y
            error=error.T
            temp=self.X*error
            temp*=(self.alpha/m)
            self.Theta-=temp
            new_cost=self.J()
            cost_list.append(new_cost)
            flag=abs(old_cost-new_cost)
        print('Gradient Descent done in {} iterations'.format(iterations))
        print(self.Theta)
        plt.figure()
        plt.plot(iterations_list,cost_list)
    def plotDesicionBoundary(self):
         plt.figure()
         for i in range(self.y.shape[1]):
             if self.y[:,i]==0:
                 plt.plot(self.X[1,i],self.X[2,i],'bo')
             else:
                plt.plot(self.X[1,i],self.X[2,i],'rx')
         x=np.linspace(-0.5,1.5,100)
         plt.plot(x,-self.Theta[0,0]/self.Theta[2,0]-self.Theta[1,0]*x/self.Theta[2,0])
              
X=np.matrix('0,0;0,1;1,0;1,1') 
X=X.T
ones=np.ones((1,X.shape[1]))
X=np.concatenate((ones,X))
#y=np.matrix('0,0,0,1') #this is for gate --And--
y=np.matrix('0,1,1,1') #this is for gate --Or--
#y=np.matrix('0,1,1,0') #this is for gate --XOR-- **
#** XOR is not correct for this linear logestic Desicion bondary
regressor=LogisticRegressor(X,y,0.01,0)#lambdaa can be 100 for regularization and big datas
x=np.matrix('1;2;4')
#regressor.gradientDescent()
#regressor.plotDesicionBoundary()
#print(regressor.regularizedJ())
#regressor.plotTrainingSet()
regressor.regularizedGradientDescent()
regressor.plotDesicionBoundary()
