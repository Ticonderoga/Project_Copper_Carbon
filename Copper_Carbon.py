#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:17:02 2023

@author: phil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as scsp
import scipy.interpolate as scint
plt.close('all')

def contour(F,field) :
    newF=np.r_[F[:77],np.nan,F[77:83],np.nan,F[83:]]
    mask = np.zeros_like(newF, dtype=bool)
    mask[77] = True
    mask[84] = True
    newF = newF.reshape((13,7))
    mask = mask.reshape((13,7))
    newF = np.flipud(np.ma.array(newF, mask=mask))
    x = np.linspace(0,12,7)
    y = np.linspace(0,24,13)
    
    if field == 'Ther' :    
        plt.contourf(x,y,newF, 100, corner_mask=False)
        plt.colorbar()
        plt.contour(x,y,newF, 10, colors='k',corner_mask=False)
        plt.title('Temperature [°C]')

    elif field == "Elec" :
        plt.contourf(x,y,newF, 100, corner_mask=False)
        plt.colorbar()
        plt.contour(x,y,newF, 10, colors='k',corner_mask=False)
        plt.title('Potential [V]')
    else :
        plt.contourf(x,y,newF, 100, corner_mask=False)
        plt.colorbar()
        plt.contour(x,y,newF, 10, colors='k',corner_mask=False)
        plt.title(field)

    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.axis('tight')

if __name__ == '__main__' :

    # Excel imports
    mat_Th = pd.read_excel('TP_Outils_Maths.xlsx',sheet_name='Thermal',
                           usecols=np.arange(1,90))
    mat_Th.fillna(0,inplace=True)
    
    rhs_Th = pd.read_excel('TP_Outils_Maths.xlsx',sheet_name='Thermal',
                           usecols=[91])
    
    mat_El = pd.read_excel('TP_Outils_Maths.xlsx',sheet_name='Electrical',
                           usecols=np.arange(1,90))
    mat_El.fillna(0,inplace=True)
    
    rhs_El = pd.read_excel('TP_Outils_Maths.xlsx',sheet_name='Electrical',
                           usecols=[91])
    
    
    M_grad_Vx = pd.read_excel('TP_Outils_Maths.xlsx',sheet_name='Grad_Vx',
                           usecols=np.arange(1,90))
    M_grad_Vx.fillna(0,inplace=True)
    MgVx = M_grad_Vx.to_numpy(dtype=float)
    
    
    M_grad_Vy = pd.read_excel('TP_Outils_Maths.xlsx',sheet_name='Grad_Vy',
                           usecols=np.arange(1,90))
    M_grad_Vy.fillna(0,inplace=True)
    MgVy = M_grad_Vy.to_numpy(dtype=float)
    
    
    # Constants
    dx= 2e-3            # spatial step
    k = 15              # thermal conductivity
    rho = 2200          # density
    Cp = 712            # heat capacity
    tf = 10*60          # final time
    h = 20              # heat transfer coefficient
    Tinf = 293.15       # ambient temperature
    Tinit = 293.15      # Initial temperature
    nbpt = mat_Th.shape[0]  # 89 points from 0 to 88
    
    d = 5e-3            # depth
    t_I =     [0, tf/4, tf/4, tf/2, tf] # time for current profile
    I_val =   [1,    1,   50,   10, 2]  # current
    I = scint.interp1d(t_I, I_val)      # I is a function of t
    rho_El = 300e-8
    
    
    # Initial vector of temperature
    T = Tinit * np.ones((nbpt,))
    
    # Initial vector of Potential
    V = np.zeros_like(T)
    
    
    n = 12000 # Nb of points on time
    time, dt  = np.linspace(0,tf,n+1,retstep=True)
    savet = 15 # time interval bewteen saves [s]
    print('Time step [s] : ',dt)
    nb_save = tf // savet +1
    
    
    save_T = np.zeros((nbpt,nb_save))
    save_V = np.zeros_like(save_T)
    save_time = np.zeros(nb_save)

    
    # Fourier and Biot numbers
    Fo = k/rho/Cp*dt/dx**2
    Bi = h*dx/k
    
    # Electrical Resistance at the contact in ohm / m2
    Re_c = 10e-3/ 1e-4 # 10 mOhm / cm2 from experiments
    Re_top = Re_c 
    
    
    # On remplace les chaines de caractères trouvées dans l'Excel
    mat_Th.replace(to_replace='MFO',value=-Fo,inplace=True)
    mat_Th.replace(to_replace='1P4FO',value=1+4*Fo,inplace=True)
    mat_Th.replace(to_replace='M2FO',value=-2*Fo,inplace=True)
    mat_Th.replace(to_replace='1P4FOP2BIFO',value=1+4*Fo+2*Bi*Fo,inplace=True)
    mat_Th.replace(to_replace='1P4FOPBIFO',value=1+4*Fo+Bi*Fo,inplace=True)
    mat_Th.replace(to_replace='1P4FOP4BIFO',value=1+4*Fo+4*Bi*Fo,inplace=True)
    mat_Th.replace(to_replace='M43FO',value=-4/3*Fo,inplace=True)
    mat_Th.replace(to_replace='M23FO',value=-2/3*Fo,inplace=True)
    mat_Th.replace(to_replace='1P4FOP43BIFO',value=1+4*Fo+4/3*Bi*Fo,inplace=True)
    
    # Transform into Numpy array and sparse matrix
    # Thermal
    MT=mat_Th.to_numpy(dtype=float)
    MT_sparse = scsp.csc_matrix(MT)
    # LU decomposition for MT
    LU_T = scsp.linalg.splu(MT_sparse)
    
    # Electrical
    ME=mat_El.to_numpy(dtype=float)
    ME_sparse = scsp.csc_matrix(ME)
    # LU decomposition for ME
    LU_E = scsp.linalg.splu(ME_sparse)
    
    # Thermal RHS
    # indices where we should inject specific values
    ind_QO = rhs_Th.index[rhs_Th['RHS'] =='QO']
    ind_QT = rhs_Th.index[rhs_Th['RHS'] =='QT']
    ind_CVJE = rhs_Th.index[rhs_Th['RHS'] =='CVJE']
    ind_QB = rhs_Th.index[rhs_Th['RHS'] =='QB']
    ind_CVBL = rhs_Th.index[rhs_Th['RHS'] =='CVBL']
    ind_BC42 = rhs_Th.index[rhs_Th['RHS'] =='BC42']
    ind_BC70 = rhs_Th.index[rhs_Th['RHS'] =='BC70']
    ind_BC71 = rhs_Th.index[rhs_Th['RHS'] =='BC71']
    
    rhsT_init = np.zeros_like(T)
    
    
    # Electrical RHS
    # indices where IB is present
    ind_IB = rhs_El.index[rhs_El['RHS']=='IB']
    rhsEl = np.zeros_like(V)
    
    
    i=0
    # On efftectue la boucle temporelle
    for t in time :
        if np.isclose(t%savet,0) :
            print('time : ',t)
            save_time[i] = t 
            save_T[:,i] = T-273.15
            save_V[:,i] = V
            i = i+1

        # Electrical Simulation
        j = I(t) / (5.5*dx*d)  # current density at the contact
        IB = -2 * rho_El * j *dx
        rhsEl[ind_IB] = IB 

        V = LU_E.solve(rhsEl)
        gVx = MgVx @ V / dx
        gVy = MgVy @ V / dx
        QJ = (gVx**2+gVy**2)/rho_El # Vector of Joule Effect sources
        
        # rhs Thermal
        rhsT = rhsT_init + QJ*dt/rho/Cp
        
        rhsT[ind_QO] = rhsT[ind_QO] + 4*Re_top*j**2*(dx*d)**2 * dt/rho/Cp/dx
        rhsT[ind_QT] = rhsT[ind_QT] + 2*Re_top*j**2*(dx*d)**2 * dt/rho/Cp/dx
        rhsT[ind_CVJE] = rhsT[ind_CVJE] + 2*Bi*Fo*Tinf
        
        rhsT[ind_QB] = rhsT[ind_QB] + 2*Re_c*j**2*(dx*d)**2 * dt/rho/Cp/dx
        rhsT[ind_CVBL] = rhsT[ind_CVBL] + 2*Re_c*j**2*(dx*d)**2 * dt/rho/Cp/dx + 2*Bi*Fo*Tinf
        rhsT[ind_BC42] = rhsT[ind_BC42] + Re_top*j**2*(dx*d)**2* dt/rho/Cp/dx+Fo*Bi*Tinf
        rhsT[ind_BC70] = rhsT[ind_BC70] + 4*Bi*Fo*Tinf
        rhsT[ind_BC71] = rhsT[ind_BC71] + 4/3*Bi*Fo*Tinf
        
        # Thermal Simulation
        T = LU_T.solve(T+rhsT)
            
            
        
        
    # contour at a specific time
    idx_t_watch = 8 # about 2 minutes because 8 x 15 = 120 s 
    plt.figure(1)
    contour(save_T[:,idx_t_watch],'Ther')
    plt.figure(2)
    contour(save_V[:,idx_t_watch],'Elec')
    
    # T profile at specific node 
    plt.figure(3)    
    node_number = 81
    plt.plot(save_time,save_T[node_number,:])
    plt.grid(True)
    plt.xlabel('time [s]')
    plt.ylabel('temperature [°C]') 
    plt.title ('T profile at node :'+str(node_number))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
