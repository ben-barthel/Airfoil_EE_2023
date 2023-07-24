# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 18:27:03 2022

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



#%% Set output dimension

def dataDims(inputData,lossType):
    # Input size
    if inputData == 'wavelet':
        numIn = 2
    else:
        numIn = 1
    # NN output size
    if lossType == 'NLL':
        
        outDim = 2
    elif lossType == 'OWNLL':
        
        outDim = 2
    elif lossType == 'MSE':
        
        outDim = 1
    elif lossType == 'OWMSE':
        
        outDim = 1
    elif lossType == 'OWMSE2':
        
        outDim = 1
    elif lossType == 'MAE':
        
        outDim = 1
    elif lossType == 'OWMAE':
        
        outDim = 1
    elif lossType == 'OWMAE2':
        
        outDim = 1
    elif lossType == 'KLD':
        
        outDim = 1
    elif lossType == 'RE':
        
        outDim = 1
    else:
        print('Please select valid loss type: MSE, OWMSE, NLL, OWNLL')
    return numIn, outDim
#%% Format Array Data

def formatArrayData(inputData,P,Ps,g,q,Ltrain,S,DT,TH,Lt0):
    st = 0
    (P_train,Ps_train, G_train, q_train) = (P[st:st+Ltrain,S],Ps[st:st+Ltrain,S],g[st:st+Ltrain,:,S],q[st+DT:st+Ltrain+DT]) 
    (P_test,Ps_test, G_test, q_test) = (P[Lt0:TH,S],Ps[Lt0:TH,S],g[Lt0:TH,:,S],q[Lt0+DT:TH+DT])
    y = q_train
    yt = q_test
    
    if inputData == 'wavelet':
        #numIn = 20
        numIn = np.shape(g)[1]
    else:
        numIn = 1

    # Input Data Type
    if inputData == 'wavelet':
        x = G_train
        x = np.asarray(x,dtype=np.float32).reshape((Ltrain,numIn,len(S)))
        x = x.reshape((Ltrain,numIn*len(S)))
        xt = G_test
        xt = np.asarray(xt,dtype=np.float32).reshape((len(q_test),numIn*len(S)))
        print('Running Model with Wavelet Array Basis')
    else:
        print('Select Valid Input Data ')

    return x, xt, y, yt, numIn

#%% Format Data

def formatData(inputData,P,Ps,g,q,Ltrain,S,DT,TH,Lt0):
    st = 0
    (P_train,Ps_train, G_train, q_train) = (P[st:st+Ltrain,S],Ps[st:st+Ltrain,S],g[st:st+Ltrain,:,S],q[st+DT:st+Ltrain+DT]) 
    (P_test,Ps_test, G_test, q_test) = (P[Lt0:TH,S],Ps[Lt0:TH,S],g[Lt0:TH,:,S],q[Lt0+DT:TH+DT])
    y = q_train
    yt = q_test
    
    if inputData == 'wavelet':
        #numIn = 2
        numIn = np.shape(g)[1]
    else:
        numIn = 1

    # Input Data Type
    if inputData == 'rawP':
        x = P_train
        x = np.asarray(x,dtype=np.float32).reshape((Ltrain,numIn,len(S)))
        x = x.reshape((Ltrain,numIn*len(S)))
        xt = P_test
        xt = np.asarray(xt,dtype=np.float32).reshape((len(q_test),numIn*len(S)))
        print('Running Model with Raw Pressure Basis')
    elif inputData == 'smoothP':
        x = Ps_train
        x = np.asarray(x,dtype=np.float32).reshape((Ltrain,numIn,len(S)))
        x = x.reshape((Ltrain,numIn*len(S)))
        xt = Ps_test
        xt = np.asarray(xt,dtype=np.float32).reshape((len(q_test),numIn*len(S)))
        print('Running Model with Smooth Pressure Basis')
    elif inputData == 'wavelet':
        x = G_train
        x = np.asarray(x,dtype=np.float32).reshape((Ltrain,numIn,len(S)))
        x = x.reshape((Ltrain,numIn*len(S)))
        xt = G_test
        xt = np.asarray(xt,dtype=np.float32).reshape((len(q_test),numIn*len(S)))
        print('Running Model with Wavelet Basis')
    else:
        print('Select Valid Input Data ')

    return x, xt, y, yt, numIn



#%% Set Losses

def setLoss(lossType,y):
    
    # output pdf for probability weight
    # Output Weight
    if lossType == 'OWMSE' or lossType == 'OWMAE' or lossType == 'OWMSE2' or lossType == 'OWMAE2':
        p_y = stats.gaussian_kde(y.reshape((1,len(y))))
        y = np.asarray(y,dtype=np.float32).reshape((len(y),1))
        py = p_y.evaluate(y.reshape((1,len(y)))).reshape((len(y),1))
        ys = np.hstack((y,py))

    # Loss Function Type
    if lossType == 'NLL':
        loss_fun = negloglik
        y = y
        outDim = 2
    elif lossType == 'OWNLL':
        loss_fun  = OWnegloglik
        y = ys
        outDim = 2
    elif lossType == 'MSE':
        loss_fun  = tf.keras.losses.MeanSquaredError()
        y = y
        outDim = 1
    elif lossType == 'OWMSE':
        loss_fun  = OWMSE
        y = ys
        outDim = 1
    elif lossType == 'OWMSE2':
        loss_fun  = OWMSE2
        y = ys
        outDim = 1
    elif lossType == 'MAE':
        loss_fun  = tf.keras.losses.MeanAbsoluteError()
        y = y
        outDim = 1
    elif lossType == 'OWMAE':
        loss_fun  = OWMAE
        y = ys
        outDim = 1
    elif lossType == 'OWMAE2':
        loss_fun  = OWMAE2
        y = ys
        outDim = 1
    elif lossType == 'KLD':
        loss_fun  = tf.keras.losses.KLDivergence()
        y = y
        outDim = 1
    elif lossType == 'RE':
        loss_fun  = RE
        y = y
        outDim = 1
    else:
        print('Please select valid loss type: MSE, OWMSE, NLL, OWNLL')
    return y, loss_fun

#%% Loss functions
def negloglik(y, dist):
    
    return -dist.log_prob(y)

def OWnegloglik(y, dist):
    py = y[:,1] + 1e-5
    y = y[:,0]
    y = y.reshape((len(y),1))
    py = y.reshape((len(y),1))
    mu = dist.mean().reshape((len(y),1))
    sig = dist.stddev().reshape((len(y),1))
    nll_ow = tf.reduce_mean(((y - mu)**2)/py ) + tf.math.log(sig)
    
    
    return nll_ow

def OWMSE(y, y_hat):
    py = y[:,1] + 1e-5
    y = y[:,0]
    y = y.reshape((len(y),1))
    py = py.reshape((len(y),1))
    y_hat = y_hat.reshape((len(y),1))
    mse_ow = tf.reduce_mean(((y - y_hat)**2)/py )
    #mse_ow = tf.reduce_mean(((y.reshape((len(y),1)) - y_hat)**2)/py )
    
    #mse_ow = tf.math.square(y_hat - y.reshape((len(y),1)))/py.reshape((len(y),1))
    return mse_ow

def OWMAE(y, y_hat):
    py = y[:,1] + 1e-5
    y = y[:,0]
    y = y.reshape((len(y),1))
    py = py.reshape((len(y),1))
    y_hat = y_hat.reshape((len(y),1))
    mae_ow = tf.reduce_mean((tf.abs(y - y_hat))/py )
    #mse_ow = tf.reduce_mean(((y.reshape((len(y),1)) - y_hat)**2)/py )
    
    #mse_ow = tf.math.square(y_hat - y.reshape((len(y),1)))/py.reshape((len(y),1))
    return mae_ow

def OWMSE2(y, y_hat):
    py = y[:,1] + 1e-5
    y = y[:,0]
    y = y.reshape((len(y),1))
    py = py.reshape((len(y),1))
    y_hat = y_hat.reshape((len(y),1))
    mse_ow = tf.reduce_mean(tf.abs(y)*((y - y_hat)**2)/py )
    #mse_ow = tf.reduce_mean(((y.reshape((len(y),1)) - y_hat)**2)/py )
    
    #mse_ow = tf.math.square(y_hat - y.reshape((len(y),1)))/py.reshape((len(y),1))
    return mse_ow

def OWMAE2(y, y_hat):
    py = y[:,1] + 1e-5
    y = y[:,0]
    y = y.reshape((len(y),1))
    py = py.reshape((len(y),1))
    y_hat = y_hat.reshape((len(y),1))
    mae_ow = tf.reduce_mean(tf.abs(y)*(tf.abs(y - y_hat))/py )
    #mse_ow = tf.reduce_mean(((y.reshape((len(y),1)) - y_hat)**2)/py )
    
    #mse_ow = tf.math.square(y_hat - y.reshape((len(y),1)))/py.reshape((len(y),1))
    return mae_ow



def RE(y, y_hat):
    lam = 0.1
    pos_loss = tf.reduce_mean(tf.math.exp(y_hat) - y_hat*tf.math.exp(y))
    neg_loss = tf.reduce_mean(tf.math.exp(-y_hat) + y_hat*tf.math.exp(-y))

    re = (1-lam)*pos_loss + lam*neg_loss
    return re


#%% Aquisition Functions

def AQ(sigma,ym,y,py,name):
    y = y.squeeze()
    ym = ym.squeeze()
    py = py.squeeze()
    T = len(y)
    ya = np.abs(y-np.mean(y))
    if name == 'IU':
        err = np.sum(sigma.squeeze())/T
    elif name == 'PW':
        err = np.sum(sigma.squeeze()/py.squeeze())/T
    elif name == 'AW':
        err = np.sum(y.squeeze()*sigma.squeeze())/T
    elif name == 'PAW':
        err = np.sum(y.squeeze()*sigma.squeeze()/py.squeeze())/T
    elif name == 'OWMAE':
        err = np.sum( np.abs(y - ym)/py.squeeze())/T  
    elif name == 'MAE':
        err = np.sum( np.abs(y - ym))/T
    elif name == 'OWMSE':
        err = np.sum( ((y - ym)**2)/py.squeeze())/T  
    elif name == 'MSE':
        err = np.sum( (y - ym)**2)/T
    return err