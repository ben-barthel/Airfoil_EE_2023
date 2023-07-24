# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 05:49:22 2022

@author: bhhba
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import scipy.io as sp
from scipy import stats
tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers
from BB_Funs import *
import time
import sys
print("TensorFlow version:", tf.__version__)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%% User Inputs
###########################################################################
tau = 7 # Forecasting lead time 
N_sens = 3 # Number of sensors
lossType = 'OWMSE' # loss function: `MSE', 'OWMSE', 'MAE', 'OWMAE'
AQtype = 'IU' # acquisition function: `IU', 'PW'


# Training Inputs
###########################################################################
Ntest_active_search = 10 # Ensemble size for active search
Ntest_final = 10 # Ensemble size for final model
Nepochs_active_search = 70 # Number of epochs for active search
Nepochs_final = 200 # Number of epochs for final model
batch_size_active_search = 0.9 # fraction of training data to include in each ensemble for active search
batch_size_final = 0.75 # fraction of training data to include in each ensemble for active search

# Model Parameters
###########################################################################
layerSize = np.int64((8,16,16,8)) # NN layer sizes
inputData = 'wavelet' # Input basis type
val_split = 0 # validation split
reg_const_a = 0.05 # activation regularizer
reg_const_k = 0 # kernel regularizer

###########################################################################
#%% Load Data
waveletDB = sp.loadmat('waveletDB.mat')
waveletDB = waveletDB['waveletDB']
sensors = np.arange(1,98,5)
sensors_full = np.arange(0,100,1)
Ltrain = 40000
Lt0 = 40000
g = np.zeros((50000,2,100))
P = np.zeros((50000,100))
Ps = np.zeros((50000,100))
q = waveletDB[0,1]['q'][0,0]
t = waveletDB[0,1]['t'][0,0]
for j in range(100):
    g[:,0,j] = waveletDB[0,j]['gm'][0,0]
    g[:,1,j] = waveletDB[0,j]['dgm'][0,0]
    if Nin > 2:
        g[:,2,j] = waveletDB[0,j]['d2gm'][0,0]
    Ps[:,j] = waveletDB[0,j]['P_smooth'][0,0]
    P[:,j] = waveletDB[0,j]['P'][0,0]
#g = g[:,:,:]
q = np.atleast_2d(np.transpose(q))
p_q = stats.gaussian_kde(q.reshape((1,len(q))))
t = np.atleast_2d(np.transpose(t))
dt = t[1,0]-t[0,0]

DT = int(np.floor(tau/dt))
TH = 50000 - (DT)
t_test = t[Lt0+DT:TH+DT]
t_train = t[:Ltrain]
q_test = q[Lt0+DT:TH+DT]


#%% Format Data and Constract Model
# Data Dimensions
numIn, outDim = dataDims(inputData,lossType)
# Format Data
_, _, y, yt  = formatData(inputData,P,Ps,g,q,Ltrain,(1,2),DT,TH,Lt0)
# Loss Function Type
y, loss_fun = setLoss(lossType,y)
# Test set PDF
p_q = stats.gaussian_kde(q_test.reshape((1,len(q_test))))
pq_test = p_q.evaluate(yt.reshape((1,len(yt)))).reshape((len(yt),1))


#%% Active Search Loop
S0 = []
aqFun = np.zeros((N_sens,len(sensors)))


for i in range(N_sens):
    print('Placing Sensor  j = '+ str(i+1))
    
    # Model
    model = tfk.Sequential([
      tfkl.Dense(layerSize[0], input_dim=numIn*(i+1)),
      tfkl.Dense(layerSize[1],activation=tf.nn.swish,kernel_regularizer=tf.keras.regularizers.L1(reg_const_k),activity_regularizer=tf.keras.regularizers.L1(reg_const_a)),
      tfkl.Dense(layerSize[2],activation=tf.nn.swish,kernel_regularizer=tf.keras.regularizers.L1(reg_const_k),activity_regularizer=tf.keras.regularizers.L1(reg_const_a)),
      tfkl.Dense(layerSize[3],activation=tf.nn.swish,kernel_regularizer=tf.keras.regularizers.L1(reg_const_k),activity_regularizer=tf.keras.regularizers.L1(reg_const_a)),
      tfkl.Dense(outDim),
    
    ])
    if outDim == 2:
          print('This code is only for models with a single output')
    # Sweep over candidate sensors
    for j in range(len(sensors)):
        print('Placing Sensor  j = '+ str(i+1)+ '. Sweep Progress: '+ str(np.round(100*(j+1)/len(sensors), decimals=1)) + '%')

        # Only consider sensored not already placed
        if not any(np.isin(S0,sensors[j])):
            S = (np.append(S0,sensors[j]))
            S = S.astype(int)
            # Format training Data
            # Format Data
            x, xt, _, _  = formatData(inputData,P,Ps,g,q,Ltrain,S,DT,TH,Lt0)

            # Compile And Train Model
            q_hat = np.zeros((Ntests_active_search,len(t_test)))
            m = np.linspace(0,Ltrain)
            for jt in range(Ntests_active_search):
                start_time = time.time()
                print('NN test # ' + str(jt+1) + '/'+ str(Ntests_active_search) + ' started.')
                # randomly select 90% subset of training data
                m = np.int64(np.linspace(0,Ltrain-1,Ltrain))
                Ltrain_r = np.int64(np.floor(Ltrain*batch_size_active_search))
                m_r = np.random.choice(m,Ltrain_r,replace = False)
                # Train Model
                model.compile(optimizer='adam',loss=loss_fun)
                model.fit(x[m_r,:], y[m_r], epochs=Nepochs,validation_split = val_split,verbose = 0)
                q_hat[jt,:] = np.squeeze(model(xt).numpy())
                print('NN test # ' + str(jt+1) + '/'+ str(Ntests_active_search) + ' completed. Time taken: ' + str(np.round(time.time() - start_time)) +' seconds' )
            # Compute Mean and Standard Deviation
            q_mean =  np.mean(q_hat,0)
            if Ntests_active_search == 1:
                q_std = 0 *q_mean
            else:
                q_std = np.std(q_hat,0)
            # Acquisition Function
            aqFun[i,j] = AQ(q_std,q_mean,q_test,pq_test,AQtype)
                
        # else record NAN
        else:
            aqFun[i,j] = np.nan
    # evaluate acquisition function          
    ind = np.nanargmin(aqFun[i,:].squeeze() )
    S0 = np.append(np.int64(S0),sensors[ind])
       

# Final Optimal Locations
S = np.int64(S0) 
#%% Train Final Model
q_hat = np.zeros((Ntest_final,len(t_test)))
m = np.linspace(0,Ltrain)
for j in range(Ntest_final):
    start_time = time.time()
    print('NN test # ' + str(j+1) + '/'+ str(Ntest_final) + ' started.')
    # randomly select 90% subset of training data
    m = np.int64(np.linspace(0,Ltrain-1,Ltrain))
    Ltrain_r = np.int64(np.floor(Ltrain*batch_size_final))
    m_r = np.random.choice(m,Ltrain_r)
    # Train Model
    model.compile(optimizer='adam',loss=loss_fun)
    model.fit(x[m_r,:], y[m_r], epochs=Nepochs_final,validation_split = val_split,verbose = 0)
    q_hat[j,:] = np.squeeze(model(xt).numpy())
    print('NN test # ' + str(j+1) + '/'+ str(Ntest_final) + ' completed. Time taken: ' + str(np.round(time.time() - start_time)) +' seconds' )
    

#%% Compute Mean and Standard Deviation of Final Model
q_mean =  np.mean(q_hat,0)
q_std = np.std(q_hat,0)



#%% Output PDF
#p_q = stats.gaussian_kde(q_test.reshape((1,len(q_test))))
val = np.linspace(-3,6,500)
pq_t = p_q.evaluate(val.reshape((1,len(val)))).reshape((len(val),1))
pq_t = pq_t/np.trapz(pq_t.squeeze(),val)
p_q_hat = stats.gaussian_kde(q_mean.reshape((1,len(q_test))))
pq_m = p_q_hat.evaluate(val.reshape((1,len(val)))).reshape((len(val),1))
pq_m = pq_m/np.trapz(pq_m.squeeze(),val)
val = val.reshape(pq_t.shape)




#%% SaveOutputs
if saveOutputs == 1:
    filename1 = ('Ensemble_' + inputData + '_NN_tau' + str(tau) +  '_Nep' + str(Nepochs) + '_Ntest' + str(Ntest_final) + '_' + lossType +  '_' + AQtype + '_out.npy')
    filename2 = ('Ensemble_' + inputData + '_NN_tau' + str(tau) +  '_Nep' + str(Nepochs) + '_Ntest' + str(Ntest_final) + '_' + lossType +  '_' + AQtype + '_pdf.npy')
    filename3 = ('Ensemble_' + inputData + '_NN_tau' + str(tau) +  '_Nep' + str(Nepochs) + '_Ntest' + str(Ntest_final) + '_' + lossType +  '_' + AQtype + '_S.npy')
   
    testSet = np.zeros((len(q_mean),4))
    testPDF = np.zeros((len(pq_m),3))
    
    testSet[:,0] = t_test.squeeze()
    testSet[:,1] = q_mean.squeeze()
    testSet[:,2] = q_std.squeeze()
    testSet[:,3] = q_test.squeeze()
    
    testPDF[:,0] = val.squeeze()
    testPDF[:,1] = pq_m.squeeze()
    testPDF[:,2] = pq_t.squeeze()
    
    np.save(filename1,testSet)
    np.save(filename2,testPDF)
    np.save(filename3,S)
    
    print('Output Saved')


