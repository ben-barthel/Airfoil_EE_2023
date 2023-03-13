# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:19:00 2023

@author: bhhba
"""


import numpy as np 
from tqdm import tqdm
import pymech.neksuite as nek
from scipy.interpolate import griddata
#from numpy and matplotlib
data_path = '../Re_17500/'
directory = data_path+'outfiles/'
snapshot_file = directory+'airfoil0.f{0:05d}'.format(1)
field = nek.readnek(snapshot_file)

t = field.time
nel = len(field.elem) # Number of spectral elements
nGLL = field.elem[0].vel.shape[3] # Order of the spectral mesh
n = nel*nGLL**2
def get_wing_boundary(alpha=5, n_points=50):
    
    # Parameters for naca 4412 airfoil
    m = 0.04
    p = 0.4
    t = 0.12
    c = 1
    x_nose = -0.25
    
    X = []
    Y = []
    
    for j in range(n_points):

        x = j/(n_points-1)

        # Airfoil thickness
        yt = 5*t*(0.2969*np.sqrt(x)-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)

        # Center coord height
        if x < p:
            yc = m/p**2*(2*p*(x/c)-(x/c)**2)
            dyc = 2*m/(1-p)**2*(p/c-x/c**2)
        else:
            yc = m/(1-p)**2*(1-2*p+2*p*(x/c)-(x/c)**2)
            dyc = 2*m/(1-p)**2*(p/c-x/c**2)

        theta = np.arctan(dyc)
        xu = x - yt*np.sin(theta) + x_nose
        yu = yc + yt*np.cos(theta)

        xj = np.round(xu*np.cos(-alpha*np.pi/180) + yu*np.sin(alpha*np.pi/180), 5)
        yj = np.round(-xu*np.sin(alpha*np.pi/180) + yu*np.cos(alpha*np.pi/180), 5)

        X.append(xj)
        Y.append(yj)

    for j in range(n_points):

        x = 1-(j+1)/n_points # Now going backwards

        # Airfoil thickness
        yt = 5*t*(0.2969*np.sqrt(x)-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)

        # Center coord height
        if x < p:
            yc = m/p**2*(2*p*(x/c)-(x/c)**2)
            dyc = 2*m/(1-p)**2*(p/c-x/c**2)
        else:
            yc = m/(1-p)**2*(1-2*p+2*p*(x/c)-(x/c)**2)
            dyc = 2*m/(1-p)**2*(p/c-x/c**2)

        theta = np.arctan(dyc)
        xb = x + yt*np.sin(theta) + x_nose
        yb = yc - yt*np.cos(theta)

        xj = np.round(xb*np.cos(-alpha*np.pi/180) + yb*np.sin(alpha*np.pi/180), 5)
        yj = np.round(-xb*np.sin(alpha*np.pi/180) + yb*np.cos(alpha*np.pi/180), 5)

        X.append(xj)
        Y.append(yj)
    
    return X,Y

def interp(field, Cx, Cy, XX, YY, method='linear', mask=None):
    """
    field - 1D array of cell values
    Cx, Cy - cell x-y values
    X, Y - meshgrid x-y values
    """
    ngrid = len(XX.flatten())
    grid_field = np.squeeze(np.reshape(griddata((Cx, Cy), field, (XX, YY), method=method), (ngrid, 1)))
    grid_field = grid_field.reshape(XX.shape)
    
    if mask is not None:
        for m in mask: grid_field[m[1],m[0]] = 0
        
    return grid_field

def load_file(file, return_xy=True):
    """
    Load velocity, pressure, and coorinates field from the file
    """

    field = nek.readnek(file)
    
    t = field.time
    nel = len(field.elem) # Number of spectral elements
    nGLL = field.elem[0].vel.shape[3] # Order of the spectral mesh
    n = nel*nGLL**2
    
    Cx = np.array([field.elem[i].pos[0, 0, j, k]
                   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    Cy = np.array([field.elem[i].pos[1, 0, j, k]
                   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])

    u = np.array([field.elem[i].vel[0, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    v = np.array([field.elem[i].vel[1, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    p = np.array([field.elem[i].pres[0, 0, j, k]
            for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    vort = np.array([field.elem[i].temp[0, 0, j, k]
            for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    
    if return_xy: return t,Cx,Cy,u,v,p,vort
    else: return t,u,v,p
    
def get_snapshot(num):
    
    snapshot_file = '../../Re_17500/outfiles/airfoil0.f{0:05d}'.format(num)
    t,Cx,Cy,u,v,p,vort = load_file(snapshot_file)

    return t,Cx,Cy,u,v,p,vort


####################################################################################################

wing_boundary_x, wing_boundary_y = get_wing_boundary(n_points=200)
nx = 800
ny = 400
nt= 1000
scale = 3
x = np.linspace(-2/scale,6.75/scale,nx)
y = np.linspace(-2.5/scale,2.5/scale,ny)
XX, YY = np.meshgrid(x, y)
t_ss = np.zeros((n))
u_ss = np.zeros((nx,ny,nt))
v_ss = np.zeros((nx,ny,nt))
w_ss = np.zeros((nx,ny,nt))
for j in range(nt):
    jj = j+3000
    t,Cx,Cy,u,v,p,W = get_snapshot(jj)
    u_ss[:,:,j] = interp(u, Cx, Cy, XX, YY, method='linear').reshape(ny,nx).transpose()
    v_ss[:,:,j] = interp(v, Cx, Cy, XX, YY, method='linear').reshape(ny,nx).transpose()
    w_ss[:,:,j] = interp(W, Cx, Cy, XX, YY, method='linear').reshape(ny,nx).transpose()
    t_ss[j] = t
    


np.save('u_snap_shot',u_ss)
np.save('v_snap_shot',v_ss)
np.save('vorticity_snap_shot',w_ss)
np.save('time_snap_shot',t_ss)

print('Snapshots Saved')