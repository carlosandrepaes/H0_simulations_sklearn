#!/usr/bin/env python
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

nsim = 100
sig = 0.01
h = 67.36
#h = 74.03
nz = 30

for n in range(nsim):

    #load cosmic chronometres data
    #dataset=pd.read_csv('hz_sim_sigh0p02_30pts_zmax1p5_P18_mc#.dat',delim_whitespace=True)
    dataset=pd.read_csv('input/hz_sim_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_mc#'+str(n+1)+'.dat', delim_whitespace=True)

    #associating z and hz with the first and second columns of the dataset, respectively
    z = dataset.iloc[:,0]
    hz = dataset.iloc[:,1]
    errhz = dataset.iloc[:,2]

    #reshaping both arrays to be processed by the regression algorithm later on
    z=z.values.reshape((len(z),1))
    hz=hz.values.reshape((len(hz),1))

    #splitting the hz dataset into training and testing set using sklearn
    z_train, z_test, hz_train, hz_test = train_test_split(z, hz, test_size=0.25, random_state=42)
    #z_train, z_test, hz_train, hz_test = train_test_split(z, hz, test_size=0.25)
    #for i in range(len(z_train)):
        #print(z_train[i],hz_train[i])
    
    #rescaling the training set - NOT NECESSARY
    #sc = StandardScaler()    
    #sc.fit(z_train)

    #z_train_std = sc.transform(z_train)
    #z_test_std = sc.transform(z_test)
    
    #for i in range(len(z_train)):
        #print(z_train[i],z_train_std[i])

    #regression algorithm - linear model
    #reg = linear_model.LinearRegression()
    #reg = linear_model.Ridge(alpha=0.01)
    #reg = linear_model.BayesianRidge()
    #reg = linear_model.Lasso(alpha=0.01)

    #regression algorithm - SVM
    #reg = SVR(kernel='linear', C=100, gamma='auto', degree=1.5, epsilon=.1, coef0=1)
    reg = SVR(kernel='poly', C=100, gamma='auto', degree=3.0, epsilon=.1, coef0=1)
    #reg = SVR(kernel='rbf', C=100, gamma='auto', degree=3.0, epsilon=.1, coef0=1)

    #regression algorithm - Nearest Neighbour
    #reg = KNeighborsRegressor(n_neighbors=3)
    #reg = RadiusNeighborsRegressor(radius=1.0, weights='distance')

    #regression algorithm - Decision Tree
    #reg = DecisionTreeRegressor(max_depth=5)
    #reg = ExtraTreesRegressor(n_estimators=100,random_state=42)

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
    scores = cross_val_score(reg, z_train, hz_train, cv=3)
    
    #printing the results (simplified version)
    #print( n, reg.predict([[0.]]), mean_absolute_error(hz_test, hz_pred), r2_score(hz_test, hz_pred), reg.score(z_train, hz_train), reg.score(z_test, hz_test), scores.mean(), scores.std() * 2 )
    print(n, reg.predict([[0.]]), mean_absolute_error(hz_test, hz_pred), reg.score(z_train, hz_train), reg.score(z_test, hz_test))
        
    np.savetxt('output/H0_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_SVM_mc#'+str(n+1)+'.dat', ([reg.predict([[0.]])]))

    np.savetxt('output/mae_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_SVM_mc#'+str(n+1)+'.dat', ([mean_absolute_error(hz_test, hz_pred)]))
    
    np.savetxt('output/scoretrain_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_SVM_mc#'+str(n+1)+'.dat', ([reg.score(z_train, hz_train))]))
    
    np.savetxt('output/scoretest_sklearn_sigh0p0'+str(int(sig*100))+'_'+str(nz)+'pts_h0p'+str(int(h*100))+'_SVM_mc#'+str(n+1)+'.dat', ([reg.score(z_test, hz_test)]))

            
#############################################################################################

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