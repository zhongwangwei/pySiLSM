# -*- coding: utf-8 -*-

"""
A libray with Python functions for calculations of
micrometeorological parameters and some miscellaneous
utilities.
functions:
    pc_bias : percentage bias
    apb :     absolute percent bias
    rmse :    root mean square error
    mae :     mean absolute error
    bias :    bias
    NS :      Nash-Sutcliffe Coefficient
    L:        likelihood estimation
    correlation: correlation
"""
__author__ = "Zhongwang Wei / zhongwang007@gmail.com"
__version__ = "0.1"
__release__ = "0.1"
__date__ = "Jan 2022"

# import required modules
import numpy as np
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from pylab import rcParams
import matplotlib
import scipy
import xarray as xr
import subprocess


### Plot settings
font = {'family' : 'DejaVu Sans'}
#font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 12,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)

def filter_nan(s= np.array([]),o= np.array([])):
    """
    this functions removed the data from simulated and observed data
    whereever the observed data contains nan

    this is used by all other functions, otherwise they will produce nan as
    output
    """
    data = np.array([s.flatten(),o.flatten()])
    data = np.transpose(data)
    data = data[~np.isnan(data).any(1)]
    
    return data[:,0],data[:,1]

def pc_bias(s,o):
    """
    Percent Bias
    input:
        s: simulated
        o: observed
    output:
        pc_bias: percent bias
    """
    s,o = filter_nan(s,o)
    return 100.0*sum(s-o)/sum(o)

def apb(s,o):
    """
    Absolute Percent Bias
    input:
        s: simulated
        o: observed
    output:
        apb_bias: absolute percent bias
    """
    s,o = filter_nan(s,o)
    return 100.0*sum(abs(s-o))/sum(o)

def rmse(s,o):
    """
    Root Mean Squared Error
    input:
        s: simulated
        o: observed
    output:
        rmses: root mean squared error
    """
    s,o = filter_nan(s,o)
    return np.sqrt(np.mean((s-o)**2))

def mae(s,o):
    """
    Mean Absolute Error
    input:
        s: simulated
        o: observed
    output:
        maes: mean absolute error
    """
    s,o = filter_nan(s,o)
    return np.mean(abs(s-o))

def bias(s,o):
    """
    Bias
    input:
        s: simulated
        o: observed
    output:
        bias: bias
    """
    s,o = filter_nan(s,o)
    return np.mean(s-o)

def NS(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
    s,o = filter_nan(s,o)
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)

def L(s,o, N=5):
    """
    Likelihood
    input:
        s: simulated
        o: observed
    output:
        L: likelihood
    """
    s,o = filter_nan(s,o)
    return np.exp(-N*sum((s-o)**2)/sum((o-np.mean(o))**2))

def correlation(s,o):
    """
    correlation coefficient
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """
    s,o = filter_nan(s,o)
    if s.size == 0:
        corr = np.NaN
    else:
        corr = np.corrcoef(o, s)[0,1]

    return corr

def KGE(s, o):
    """
    Kling-Gupta Efficiency
    input:
        s: simulated
        o: observed
    output:
        kge: Kling-Gupta Efficiency
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    s,o = filter_nan(s,o)
    cc = correlation(s,o)
    alpha = np.std(s)/np.std(o)
    beta = np.sum(s)/np.sum(o)
    kge = 1- np.sqrt( (cc-1)**2 + (alpha-1)**2 + (beta-1)**2 )
    return kge   #, cc, alpha, beta

def index_agreement(s,o):
    """
	index of agreement
	input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    s,o = filter_nan(s,o)
    ia = 1 -(np.sum((o-s)**2))/(np.sum(
    			(np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
    return ia


class KAPPA:

    def __init__(self,s,o):
        s = s.flatten()
        o = o.flatten()
        #check if the length of the vectors are same or not
        if len(s) != len(o):
            raise Exception("Length of both the vectors must be same")

        self.s = s.astype(int)
        self.o = o.astype(int)

    def kappa_coeff(self):
        s = self.s
        o = self.o
        n = len(s)

        foo1 = np.unique(s)
        foo2 = np.unique(o)
        unique_data = np.unique(np.hstack([foo1,foo2]).flatten())
        self.unique_data = unique_data
        kappa_mat = np.zeros((len(unique_data),len(unique_data)))

        ind1 = np.empty(n, dtype=int)
        ind2 = np.empty(n, dtype=int)
        for i in range(len(unique_data)):
            ind1[s==unique_data[i]] = i
            ind2[o==unique_data[i]] = i

        for i in range(n):
            kappa_mat[ind1[i],ind2[i]] += 1

        self.kappa_mat = kappa_mat

        # compute kappa coefficient
        # formula for kappa coefficient taken from
        # http://adorio-research.org/wordpress/?p=2301
        tot = np.sum(kappa_mat)
        Pa = np.sum(np.diag(kappa_mat))/tot
        PA = np.sum(kappa_mat,axis=0)/tot
        PB = np.sum(kappa_mat,axis=1)/tot
        Pe = np.sum(PA*PB)
        kappa_coeff = (Pa-Pe)/(1-Pe)

        return kappa_mat, kappa_coeff

    def kappa_figure(self,fname,data,data_name):
        data = np.array(data)
        data = data.astype(int)

        try:
            self.kappa_mat
        except:
            self.kappa_coeff()

        kappa_mat = self.kappa_coeff()
        unique_data = self.unique_data

        tick_labels = []
        for i in range(len(unique_data)):
            unique_data[i] == data
            tick_labels.append(data_name[find(data==unique_data[i])])

        plt.subplots_adjust(left=0.3, top=0.8)
        plt.imshow(kappa_mat, interpolation='nearest',origin='upper')
        #plt.gca().tick_top()
        plt.xticks(range(len(unique_data)),tick_labels, rotation='vertical')
        plt.yticks(range(len(unique_data)),tick_labels)
        #yticks(range(0,25),np.linspace(0,3,13))
        plt.colorbar(shrink = 0.8)
        #plt.title(vi_name[j])
        plt.savefig(fname)
        plt.close()

if __name__=='__main__':
    argv                   = sys.argv
    Model                  = str(argv[1])
    Vdata                  = str(argv[2])

    Sim_tim_res            = str(argv[3])
    Sim_spatial_res        = str(argv[4])
    Sim_data_groupby       = str(argv[5])
    Sim_data_dir           = str(argv[6])

    Obs_tim_res            = str(argv[7])
    Obs_spatial_res        = str(argv[8])
    Obs_data_groupby       = str(argv[9])
    Obs_data_dir           = str(argv[10])

    Stationlist_dir        = str(argv[11])
    obsvarname             = str(argv[12])
    simvarname             = str(argv[13])
    casename               = str(argv[14])
    compar_tim_res         = str(argv[15])
    sim_sdate              = int(argv[16])
    sim_edate              = int(argv[17])
    
    stnlist                     =f"cases/{casename}_{compar_tim_res}/selected_list_{Model}_{Vdata}.csv"
    station_list                = pd.read_csv(stnlist,header=0)

    station_list['PBIAS']   =  [-9999.0] * len(station_list['lon'])
    station_list['APB']     =  [-9999.0] * len(station_list['lon'])
    station_list['RMSE']    =  [-9999.0] * len(station_list['lon'])
    station_list['MAE']     =  [-9999.0] * len(station_list['lon'])
    station_list['BIAS']    =  [-9999.0] * len(station_list['lon'])
    station_list['NSE']     =  [-9999.0] * len(station_list['lon'])
    station_list['L']       =  [-9999.0] * len(station_list['lon'])
    station_list['R']       =  [-9999.0] * len(station_list['lon'])
    station_list['KGE']     =  [-9999.0] * len(station_list['lon'])
    station_num             =  len(station_list['use_Sdate'])
    print(station_num)
    os.makedirs(f'cases/{casename}_{compar_tim_res}/plot_stn', exist_ok=True)
    os.makedirs(f'cases/{casename}_{compar_tim_res}/metrics', exist_ok=True)
    for j in range(station_num):
        simnc=(f'cases/{casename}_{compar_tim_res}/tmp/sim/'+f"sim_{station_list['ID'][j]}"+f"_{station_list['use_Sdate'][j]}"+f"_{station_list['use_Edate'][j]}.nc")
        obsnc=(f'cases/{casename}_{compar_tim_res}/tmp/obs/'+f"obs_{station_list['ID'][j]}"+f"_{station_list['use_Sdate'][j]}"+f"_{station_list['use_Edate'][j]}.nc")

        with xr.open_dataset(obsnc) as dobs:
            obs=dobs['%s'%(obsvarname)].squeeze()#("lon","lat")
            #obs=obs.set_index((f'{obsvarname}')=("time"))
        del dobs
        with xr.open_dataset(simnc) as dsim:
            #sim=dsim.outflw  #[:,0,0]  #.outflw.reduce #squeeze("lon_cama","lat_cama")
            sim=dsim['%s'%(simvarname)].squeeze() #("lon","lat")
        del dsim
        print(station_list['ID'][j])

        try:    
            station_list['PBIAS'].values[j]=pc_bias(sim.values[:],obs.values[:])
            station_list['APB'].values[j]=apb(sim.values[:],obs.values[:])
            station_list['RMSE'].values[j]=rmse(sim.values[:],obs.values[:])
            station_list['MAE'].values[j]=mae(sim.values[:],obs.values[:])
            station_list['BIAS'].values[j]=bias(sim.values[:],obs.values[:])
            station_list['NSE'].values[j]=NS(sim.values[:],obs.values[:])
            station_list['L'].values[j]=L(sim.values[:],obs.values[:])
            station_list['R'].values[j]=correlation(sim.values[:],obs.values[:])
            station_list['KGE'].values[j]=KGE(sim.values[:],obs.values[:])
        except:
            station_list['PBIAS'].values[j]=-9999.0
            station_list['APB'].values[j]=-9999.0
            station_list['RMSE'].values[j]=-9999.0
            station_list['MAE'].values[j]=-9999.0
            station_list['BIAS'].values[j]=-9999.0
            station_list['NSE'].values[j]=-9999.0
            station_list['L'].values[j]=-9999.0
            station_list['R'].values[j]=-9999.0
            station_list['KGE'].values[j]=-9999.0

        legs =['obs','sim']
        lines=[1.5, 1.5]
        alphas=[1.,1.]
        linestyles=['solid','dotted']
        colors=['g',"purple"]
        fig, ax = plt.subplots(1,1,figsize=(10,5))
        obs[:].plot.line (x='time', label='obs', linewidth=lines[0], linestyle=linestyles[0], alpha=alphas[0],color=colors[0]) 
        sim[:].plot.line(x='time', label='sim', linewidth=lines[0], linestyle=linestyles[1], alpha=alphas[1],color=colors[1],add_legend=True) 
        ax.set_ylabel(f'{obsvarname}', fontsize=18)
        ax.set_xlabel('Date', fontsize=18)
        #addtxt='R=%s PBIAS=%s KGE=%s'%(station_list['R'].values[j],station_list['PBIAS'].values[j],station_list['KGE'].values[j])
        #ax.text(3, 4, addtxt, style='italic', fontsize=12, 
        #bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        ax.tick_params(axis='both', top='off', labelsize=16)
        ax.legend(loc='upper right', shadow=False, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'cases/{casename}_{compar_tim_res}/plot_stn/plot_'+f"{station_list['ID'][j]}"+'_timeseries.png')
        plt.close(fig)
        del obs,sim,obsnc,simnc
    
    station_list.to_csv(f'cases/{casename}_{compar_tim_res}/metrics/'+f"{casename}_{compar_tim_res}_metrics_{Model}_{Vdata}_{obsvarname}_vs_{simvarname}.csv",index=False)


