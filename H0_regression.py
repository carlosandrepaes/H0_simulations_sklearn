#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from random import seed
from random import gauss
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import cross_val_score

def hz_model(x, a, b):
    return b*np.sqrt(a*(1.+x)**3. + (1.-a))

#load cosmic chronometres data
#z,hz,errhz,hzid = np.loadtxt('hz_nobao.dat', unpack='true')
#dataset=pd.read_csv('hz_sim_P18_mc#0004.dat',delim_whitespace=True)
dataset=pd.read_csv('hz_sim_sigh0p02_30pts_zmax1p5_P18_mc#0001.dat',delim_whitespace=True)
#print(dataset)

#associating z and hz with the first and second columns of the dataset, respectively
z = dataset.iloc[:,0]
hz = dataset.iloc[:,1]
errhz = dataset.iloc[:,2]

#drawing hz value from a normal distribution centred at hz, with errhz as standard deviation. 
#hz1 = hz + errhz*np.random.randn()
#hz1 = gauss(hz,errhz)
#print(z,hz,hz1)

#reshaping both arrays to be processed by the regression algorithm later on
z=z.values.reshape((len(z),1))
hz=hz.values.reshape((len(hz),1))
#hz1=hz1.values.reshape((len(hz1),1))

#for i in range(len(z)):
    #print(z[i],hz[i],errhz[i],hz1[i],hz1[i]/hz[i])
    
#splitting the hz dataset into training and testing set using sklearn
z_train, z_test, hz_train, hz_test = train_test_split(z, hz, test_size=0.25, random_state=42)
#z_train, z_test, hz_train, hz_test = train_test_split(z, hz, test_size=0.25)
#print(z_train, hz_train)
#for i in range(len(z_train)):
    #print(z_train[i],hz_train[i])

#regression algorithm - linear model
#reg = linear_model.LinearRegression()
#reg = linear_model.Ridge(alpha=0.01)
#reg = linear_model.BayesianRidge()
#reg = linear_model.Lasso(alpha=0.01)

#regression algorithm - SVM
#reg = SVR(kernel='linear', C=100, gamma='auto', degree=1.5, epsilon=.1, coef0=1)
#reg = SVR(kernel='poly', C=100, gamma='auto', degree=3.0, epsilon=.1, coef0=1)
#reg = SVR(kernel='rbf', C=100, gamma='auto', degree=3.0, epsilon=.1, coef0=1)

#regression algorithm - Nearest Neighbour
#reg = KNeighborsRegressor(n_neighbors=3)
#reg = RadiusNeighborsRegressor(radius=1.0, weights='distance')

#regression algorithm - Decision Tree
#reg = DecisionTreeRegressor(max_depth=5)
reg = ExtraTreesRegressor(n_estimators=1000,random_state=42)

#regression algorithm - Ensemble
#reg = AdaBoostRegressor(n_estimators=100)
#reg = AdaBoostRegressor(ExtraTreesRegressor(),n_estimators=100,random_state=42)
#reg = RandomForestRegressor(n_estimators=500, random_state=42)
#reg = GradientBoostingRegressor(n_estimators=1000)

#regression algorithm - neural neural_network
#reg = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', solver='lbfgs', learning_rate='adaptive', max_iter=200, random_state=42)

#training the algorithm with the training data
reg.fit(z_train, hz_train)   

#making predictions using the testing set
hz_pred = reg.predict(z_test)

#cross-validation score
scores = cross_val_score(reg, z_train, hz_train)

## The coefficients for linear regression
#print('Coefficients: \n', reg.coef_)
##The intercept
#print('intercept: \n', reg.intercept_)
## The mean squared error
#print("Mean absolute error: %.2f"
      #% mean_absolute_error(hz_test, hz_pred))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(hz_test, hz_pred))

#for i in range(len(z)):
    #print(z[i],hz[i],errhz[i],hz_pred[i])

#printing the results:
# Estimated H0
#print('H0:', reg.fit(z, hz).predict(0))
## The mean squared error
#print("Mean absolute error: %.2f" % mean_absolute_error(hz_test, hz_pred))
## Accuracy score
#print('Variance score: %.2f' % r2_score(hz_test, hz_pred))
## cross validation score
##print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#print("%0.2f %0.2f" % (scores.mean(), scores.std() * 2))

#printing the results (simplified version)
print(reg.fit(z_test, hz_test).predict(0.), mean_absolute_error(hz_test, hz_pred), r2_score(hz_test, hz_pred), scores.mean(), scores.std() * 2)
#print(reg.fit(z, hz1).predict(0.)*hz[0]/hz1[0])

#############################################################################################

## z-array
#zmin=0.00
#zmax=1.50
#nz=1000
#z_arr=zmin+(zmax-zmin)*np.arange(nz)/(nz-1.0)

#plotting hz data

##latex rendering text fonts
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

## Create figure size in inches
#fig, ax = plt.subplots(figsize = (8.5, 6.))

## Define axes
#ax.set_ylabel(r"$H(z) \; (\mathrm{km} \; \mathrm{s}^{-1} \; \mathrm{Mpc}^{-1})$", fontsize=22)
#ax.set_xlabel(r"$z$", fontsize=22)
##plt.yscale('log')
##plt.xscale('log')
##ax.set_xlim(0., 2.)
#for t in ax.get_xticklabels(): t.set_fontsize(20)
#for t in ax.get_yticklabels(): t.set_fontsize(20)

#plt.plot(z, reg.fit(z, hz).predict(z), '-', color='red')
##plt.plot(z, hz_model(z,0.3166,67.36), '--', color='blue')
#plt.errorbar(z, hz, yerr=errhz, fmt='.', color='black')
#plt.title((r"Hz fit, CC+BAO, SVM"), fontsize='20')
#plt.legend((r"predicted", "real"), loc='best', fontsize='22')
##plt.legend((r"predicted", "$\Lambda$CDM", "real"), loc='best', fontsize='22')
#plt.show()

##saving the plot
#fig.savefig('hz_svm_pred_P18_mc#0001.png')