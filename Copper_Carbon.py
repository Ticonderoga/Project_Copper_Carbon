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
        plt.contourf(x,y,newF-273.15, 100, corner_mask=False)
        plt.colorbar()
        plt.contour(x,y,newF-273.15, 10, colors='k',corner_mask=False)
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
    
    d = 5e-3            # depth 
    I = 10              # current
    j = I / (5.5*dx*d)  # current density
    rho_El = 300e-8
    
    IB = -2 * rho_El * j *dx
    rhs_El.replace(to_replace='IB',value=IB,inplace=True)
    rhsEl = rhs_El.to_numpy(dtype=float).flatten()

    # Initial vector of temperature
    T = Tinit * np.ones((89,))
    
    n = 12000 # Nbre de points en temps
    time, dt  = np.linspace(0,tf,n+1,retstep=True)
    savet = 15 # interval de temps pour les sauvegardes [s]
    print('Time step [s] : ',dt)
    nb_save = tf // savet +1
    n = mat_Th.shape[0]  # 89 points from 0 to 88
    
    save_T = np.zeros((n,nb_save))
    save_V = np.zeros_like(save_T)

    
    # Nbre de Fourier et Biot
    Fo = k/rho/Cp*dt/dx**2
    Bi = h*dx/k
    
    # Sources électriques
    # Effet Joule en W.m-3
    QJ = 0
    # QJ = 0
    # Source au contact W.m-2
    qc = 10e-3/ 1e-4 # 10 mOhm / cm2
    # qc=0
    qtop = qc 
    
    
    # Calcul des indices
    # indices_top = np.arange(0,7)
    # indices_left = np.r_[np.arange(0,71,7) , [77, 83]]
    # indices_right = np.r_[np.arange(6,77,7) ,[82, 88]]
    # indices_bottom = np.arange(83,89)
    
    # corner_top_left = np.intersect1d(indices_top,indices_left)
    # corner_top_right = np.intersect1d(indices_top,indices_right)
    # corner_bottom_left = np.intersect1d(indices_bottom,indices_left)
    # corner_bottom_right = np.intersect1d(indices_bottom,indices_right)
    
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
    # LU decomposition
    LU_T = scsp.linalg.splu(MT_sparse)
    
    # Electrical
    ME=mat_El.to_numpy(dtype=float)
    ME_sparse = scsp.csc_matrix(ME)
    # LU decomposition
    LU_E = scsp.linalg.splu(ME_sparse)
    
    # Constante dans le 2nd membre
    JE = QJ*dt/rho/Cp
    QO = JE+4*qtop*dt/rho/Cp/dx
    QT = JE+2*qtop*dt/rho/Cp/dx
    CVJE = JE+2*Bi*Fo*Tinf
    QB = JE+2*qc*dt/rho/Cp/dx
    CVBL = QB + CVJE
    BC42 = JE+qtop*dt/rho/Cp/dx+Fo*Bi*Tinf
    BC70 = JE+4*Bi*Fo*Tinf
    BC71 = JE+4/3*Bi*Fo*Tinf
    
    # On remplace les chaines de caractères trouvées dans l'Excel
    rhs_Th.replace(to_replace='JE',value=JE,inplace=True)
    rhs_Th.replace(to_replace='QO',value=QO,inplace=True)
    rhs_Th.replace(to_replace='QT',value=QT,inplace=True)
    rhs_Th.replace(to_replace='CVJE',value=CVJE,inplace=True)
    rhs_Th.replace(to_replace='QB',value=QB,inplace=True)
    rhs_Th.replace(to_replace='CVBL',value=CVBL,inplace=True)
    rhs_Th.replace(to_replace='BC42',value=BC42,inplace=True)
    rhs_Th.replace(to_replace='BC70',value=BC70,inplace=True)
    rhs_Th.replace(to_replace='BC71',value=BC71,inplace=True)
    
    # on transforme rhs en Numpy
    rhsT_init = rhs_Th.to_numpy(dtype=float).flatten()
    
    save_T[:,0] = T
    save_V[:,0] = 0
    i=0
    # On efftectue la boucle temporelle
    for t in time :
        # Electrical Simulation
        V = LU_E.solve(rhsEl)
        gVx = MgVx @ V / dx
        gVy = MgVy @ V / dx
        QJ = (gVx**2+gVy**2)/rho_El
        
        rhsT = rhsT_init + QJ*dt/rho/Cp
        
        T = LU_T.solve(T+rhsT)
        if np.isclose(t%savet,0) :
            print('time : ',t)
            if i!=0 :
                save_T[:,i] = T
                save_V[:,i] = V
            i = i+1
            
            
        
        
    # T = LU_T.solve(T+rhsT)
    plt.figure(1)
    contour(T,'Ther')
    plt.figure(2)
    contour(V,'Elec')
    
    plt.figure(3)
    
    contour(QJ,'Heat sources [W.m-3]')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
