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
from opt_funs import *
from ee_funs import *
import time
import sys
print("TensorFlow version:", tf.__version__)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%% User Inputs
###########################################################################
Ntest = 10 # ensemble size
computeAlphaStar = 0 # 1 for yes 0 for no
saveOutputs = 1 # 1 for yes 0 for no


lossType_vec =['OWMAE','MAE'] # loss function: `MSE', 'OWMSE', 'MAE', 'OWMAE'
inputData_vec = ['rawP','smoothP','wavelet'] # Input Data
tauv = (0,3,7,10) # lead times

# Model Parameters 
###########################################################################

layerSize = np.int64((8,16,16,8))
S = (5,35,65,95) # Sensor locations: 0 - 99
Nepochs = 201 # number of epochs
val_split = 0 # fraction of training data used for validation
batch_size = 0.75 # fraction of training data to include in each element of ensemble
reg_const_a = 0.05 # activation regularizer
reg_const_k = 0 # kernel regularizer

#########################################################################
#%% Load Data
waveletDB = sp.loadmat('../../waveletDB.mat')
waveletDB = waveletDB['waveletDB']
Nin = 2
Ltrain = 40000 # Size of traning data
Lt0 = 40000
g = np.zeros((50000,Nin,100))
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



#%%% Loop over tau and basis types

aStar = np.zeros((3,2,4))
for j_tau in range(len(tauv)):
    tau = tauv[j_tau]
    
    DT = int(np.floor(tau/dt))
    TH = 50000 - (DT)
    t_test = t[Lt0+DT:TH+DT]
    t_train = t[:Ltrain]
    q_test = q[Lt0+DT:TH+DT]
    
    
    
    for j_l in range(len(lossType_vec)):
        lossType = lossType_vec[j_l]
        print(lossType)
        for j_id in range(len(inputData_vec)):
            inputData = inputData_vec[j_id]
            print(inputData)
            
            #%% Format Data and Constract Model
            
            # Data Dimensions
            numIn, outDim = dataDims(inputData,lossType)
            
            # Format Data
            x, xt, y, yt, numIn  = formatData(inputData,P,Ps,g,q,Ltrain,S,DT,TH,Lt0)
            
            # Loss Function Type
            y, loss_fun = setLoss(lossType,y)
            
            # Test set PDF
            p_q = stats.gaussian_kde(q_test.reshape((1,len(q_test))))
            pyt = p_q.evaluate(yt.reshape((1,len(yt)))).reshape((len(yt),1))
            
            
            # Model
            model = tfk.Sequential([
              tfkl.Dense(layerSize[0], input_dim=numIn*len(S)),
              tfkl.Dense(layerSize[1],activation=tf.nn.swish,kernel_regularizer=tf.keras.regularizers.L1(reg_const_k),activity_regularizer=tf.keras.regularizers.L1(reg_const_a)),
              tfkl.Dense(layerSize[2],activation=tf.nn.swish,kernel_regularizer=tf.keras.regularizers.L1(reg_const_k),activity_regularizer=tf.keras.regularizers.L1(reg_const_a)),
              tfkl.Dense(layerSize[3],activation=tf.nn.swish,kernel_regularizer=tf.keras.regularizers.L1(reg_const_k),activity_regularizer=tf.keras.regularizers.L1(reg_const_a)),
              tfkl.Dense(outDim),
            
            ])
            if outDim == 2:
                  print('This code is only for models with a single output')
            
            
            #%% Compile And Train Model
            q_hat = np.zeros((Ntest,len(t_test)))
            m = np.linspace(0,Ltrain)
            for j in range(Ntest):
                start_time = time.time()
                print('NN test # ' + str(j+1) + '/'+ str(Ntest) + ' started.')
                # randomly select 90% subset of training data
                m = np.int64(np.linspace(0,Ltrain-1,Ltrain))
                Ltrain_r = np.int64(np.floor(Ltrain*batch_size))
                m_r = np.random.choice(m,Ltrain_r,replace = False)
                # Train Model
                model.compile(optimizer='adam',loss=loss_fun)
                model.fit(x[m_r,:], y[m_r], epochs=Nepochs,validation_split = val_split,verbose = 0)
                q_hat[j,:] = np.squeeze(model(xt).numpy())
                print('NN test # ' + str(j+1) + '/'+ str(Ntest) + ' completed. Time taken: ' + str(np.round(time.time() - start_time)) +' seconds' )
                
            
            #%% Compute Mean and Standard Deviation
            q_mean =  np.mean(q_hat,0)
            q_std = np.std(q_hat,0)
            
            
            
            
            
            #%% Compute Output PDF
            #p_q = stats.gaussian_kde(q_test.reshape((1,len(q_test))))
            val = np.linspace(-3,6,500)
            pq_t = p_q.evaluate(val.reshape((1,len(val)))).reshape((len(val),1))
            pq_t = pq_t/np.trapz(pq_t.squeeze(),val)
            
            p_q_hat = stats.gaussian_kde(q_mean.reshape((1,len(q_test))))
            
            pq_m = p_q_hat.evaluate(val.reshape((1,len(val)))).reshape((len(val),1))
            pq_m = pq_m/np.trapz(pq_m.squeeze(),val)
            
            val = val.reshape(pq_t.shape)
            
           
            
            
            #%% Save Outputs
            if saveOutputs == 1:
                filename1 = ('Basis_Test_Ensemble_' + inputData + '_NN_tau' + str(tau) +  '_Nep' + str(Nepochs) + '_Ntest' + str(Ntest) + '_' + lossType + '_S_' + str(S) + '_out.npy')
                filename2 = ('Basis_Test_Ensemble_' + inputData + '_NN_tau' + str(tau) +  '_Nep' + str(Nepochs) + '_Ntest' + str(Ntest) + '_' + lossType + '_S_' + str(S) + '_pdf.npy')
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
                
                print('Output Saved')
        
            #%% Compute Alpha Star
            if computeAlphaStar == 1:
                tr = np.linspace(t_test[0],t_test[len(t_test)-1],10000)
                a = np.interp(tr.squeeze(),t_test.squeeze(),q_test.squeeze())
                b = np.interp(tr.squeeze(),t_test.squeeze(),q_mean.squeeze())
                alpha_star, omega_opt, a_opt, b_opt = guth_criterion(a.reshape((len(tr),1)), b.reshape((len(tr),1)),return_thresholds=True,nq=51,q_min=0.0,q_max=0.5,nb=501)
                aStar[j_id,j_l,j_tau] = alpha_star
                print(inputData + ' alpha star = ' + str(alpha_star))

if computeAlphaStar == 1:
    filename3 = ('Basis_Test_Ensemble_NN_tau' + str(tau) +  '_Nep' + str(Nepochs) + '_Ntest' + str(Ntest) + str(S) + '_alpha_star.npy')
    np.save(filename3,aStar)