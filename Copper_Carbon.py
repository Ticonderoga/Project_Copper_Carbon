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

def contour_Ther(T) :
    newT=np.r_[T[:77],np.nan,T[77:83],np.nan,T[83:]]
    mask = np.zeros_like(newT, dtype=bool)
    mask[77] = True
    mask[84] = True
    newT = newT.reshape((13,7))
    mask = mask.reshape((13,7))
    newT = np.flipud(np.ma.array(newT, mask=mask))
    x = np.linspace(0,12,7)
    y = np.linspace(0,24,13)
    
    plt.contourf(x,y,newT-273.15, 100, corner_mask=False)
    plt.colorbar()
    
    plt.contour(x,y,newT-273.15, 10, colors='k',corner_mask=False)
    
    plt.title('Champ de température [°C]')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.axis('tight')

if __name__ == '__main__' :

    # Importation du fichier Excel
    mat_Th = pd.read_excel('TP_Outils_Maths.ods',sheet_name='Thermal',
                           usecols=np.arange(1,90))
    mat_Th.fillna(0,inplace=True)
    
    rhs_Th = pd.read_excel('TP_Outils_Maths.ods',sheet_name='Thermal',
                           usecols=[91])
    
    
    # Constantes
    dx= 2e-3
    k = 15
    rho = 2200
    Cp = 712
    tf = 30*60
    h = 20
    Tinf = 293.15
    Tinit = 293.15 # Température initiale
    
    T = Tinit * np.ones((89,))
    
    n = 12000 # Nbre de points en temps
    time, dt  = np.linspace(0,tf,n+1,retstep=True)
    savet = 15 # interval de temps pour les sauvegardes [s]
    print('Time step [s] : ',dt)
    
    # Nbre de Fourier et Biot
    Fo = k/rho/Cp*dt/dx**2
    Bi = h*dx/k
    
    # Sources électriques
    # Effet Joule en W.m-3
    QJ = 1e-3
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
    
    # On transforme en Numpy puis en matrice creuse
    MT=mat_Th.to_numpy(dtype=float)
    MT_sparse = scsp.csc_matrix(MT)
    # On effectue la décomposition LU
    LU_T = scsp.linalg.splu(MT_sparse)
    
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
    rhsT = rhs_Th.to_numpy(dtype=float).flatten()
    
    # On efftectue la boucle temporelle
    for t in time :
        if np.isclose(t%savet,0):
            print('time : ',t)
        T = LU_T.solve(T+rhsT)
    # T = LU_T.solve(T+rhsT)
    contour_Ther(T)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
