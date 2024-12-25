# -*- coding: utf-8 -*-
__author__ = "Zhongwang Wei / zhongwang007@gmail.com"
__version__ = "0.1"
__release__ = "0.1"
__date__ = "May 2022"

import xarray as xr
import numpy as np
import datetime
import pandas as pd
import glob, os, shutil,sys
import subprocess
from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from pylab import rcParams
import matplotlib
from mpl_toolkits.basemap import Basemap
import numpy as np

class runvalidation:
    def __init__(self,casename,obsvarname,simvarname,compar_tim_res):
        self.name    = 'runvalidation'
        self.version = '0.1'
        self.release = '0.1'
        self.date    = 'Mar 2023'
        self.author  = "Zhongwang Wei / zhongwang007@gmail.com"
        self.casename       = casename
        self.obsvarname     = obsvarname
        self.simvarname     = simvarname
        self.compar_tim_res = compar_tim_res
        print(self.compar_tim_res)
  
    def filter_nan(self,s= np.array([]),o= np.array([])):
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

    def pc_bias(self,s,o):
        """
        Percent Bias
        input:
            s: simulated
            o: observed
        output:
            pc_bias: percent bias
        """
        s,o = self.filter_nan(s,o)
        return 100.0*sum(s-o)/sum(o)

    def apb(self,s,o):
        """
        Absolute Percent Bias
        input:
            s: simulated
            o: observed
        output:
            apb_bias: absolute percent bias
        """
        s,o = self.filter_nan(s,o)
        return 100.0*sum(abs(s-o))/sum(o)

    def rmse(self,s,o):
        """
        Root Mean Squared Error
        input:
            s: simulated
            o: observed
        output:
            rmses: root mean squared error
        """
        s,o = self.filter_nan(s,o)
        return np.sqrt(np.mean((s-o)**2))

    def mae(self,s,o):
        """
        Mean Absolute Error
        input:
            s: simulated
            o: observed
        output:
            maes: mean absolute error
        """
        s,o = self.filter_nan(s,o)
        return np.mean(abs(s-o))

    def bias(self,s,o):
        """
        Bias
        input:
            s: simulated
            o: observed
        output:
            bias: bias
        """
        s,o = self.filter_nan(s,o)
        return np.mean(s-o)

    def NS(self,s,o):
        """
        Nash Sutcliffe efficiency coefficient
        input:
            s: simulated
            o: observed
        output:
            ns: Nash Sutcliffe efficient coefficient
        """
        s,o = self.filter_nan(s,o)
        return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)

    def L(self,s,o, N=5):
        """
        Likelihood
        input:
            s: simulated
            o: observed
        output:
            L: likelihood
        """
        s,o = self.filter_nan(s,o)
        return np.exp(-N*sum((s-o)**2)/sum((o-np.mean(o))**2))

    def correlation(self,s,o):
        """
        correlation coefficient
        input:
            s: simulated
            o: observed
        output:
            correlation: correlation coefficient
        """
        s,o = self.filter_nan(s,o)
        if s.size == 0:
            corr = np.NaN
        else:
            corr = np.corrcoef(o, s)[0,1]

        return corr

    def KGE(self,s, o):
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
        s,o = self.filter_nan(s,o)
        cc = self.correlation(s,o)
        alpha = np.std(s)/np.std(o)
        beta = np.sum(s)/np.sum(o)
        kge = 1- np.sqrt( (cc-1)**2 + (alpha-1)**2 + (beta-1)**2 )
        return kge   #, cc, alpha, beta

    def index_agreement(self,s,o):
        """
	    ndex of agreement
	    input:
            s: simulated
            o: observed
        output:
            ia: index of agreement
        """
        s,o = self.filter_nan(s,o)
        ia = 1 -(np.sum((o-s)**2))/(np.sum(
    			(np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
        return ia

    def validation(self):
        casename               = self.casename
        obsvarname             = self.obsvarname
        simvarname             = self.simvarname
        compar_tim_res         = self.compar_tim_res

        stnlist                 =f"../cases/{casename}/list/selected_list.txt"
        print(stnlist)
        station_list = pd.read_csv(stnlist,header=0)
        station_list['PBIAS']   =  [-9999.0] * len(station_list['SiteName'])
        station_list['APB']     =  [-9999.0] * len(station_list['SiteName'])
        station_list['RMSE']    =  [-9999.0] * len(station_list['SiteName'])
        station_list['MAE']     =  [-9999.0] * len(station_list['SiteName'])
        station_list['BIAS']    =  [-9999.0] * len(station_list['SiteName'])
        station_list['NSE']     =  [-9999.0] * len(station_list['SiteName'])
        station_list['L']       =  [-9999.0] * len(station_list['SiteName'])
        station_list['R']       =  [-9999.0] * len(station_list['SiteName'])
        station_list['KGE']     =  [-9999.0] * len(station_list['SiteName'])
        station_num             =  len(station_list['SiteName'])
        print(station_num)
        for i in np.arange(len(station_list['SiteName'])):
            simdata=f'../cases/{casename}/sim/'+f"{station_list['SiteName'][i]}.nc"
            obsdata=f"/tera06/zhwei/hydroecology/FLUXNET_PLUMBER2_WEI/flux/{station_list['filename'][i]}"


            print(f"processing observation site: {station_list['SiteName'][i]}")
            print(f"{obsdata}")
            print(f"{simdata}")
        
            with xr.open_dataset(obsdata) as dobs:
                try: 
                    obs=dobs['%s'%(obsvarname)].squeeze()#("lon","lat")
                except:
                    obsvarname=obsvarname[:-4]
                    obs=dobs['%s'%(obsvarname)].squeeze()#("lon","lat")
            del dobs
            with xr.open_dataset(simdata) as dsim:
                sim=dsim['%s'%(simvarname)].squeeze() #("lon","lat")
            del dsim

            #            dfx1=dfx.sel(time=slice(f'{startyear}-{startmon}-{startday}T{starthour}',f'{endyear}-{endmon}-{endday}T{endhour}'))
            if (compar_tim_res=="Month"):
                obs=obs.resample(time='1M').mean()#.reduce(np.nan)
                sim=sim.resample(time='1M').mean()#.reduce(np.nan)
            elif (compar_tim_res=="Hour"):
                obs=obs.resample(time='1H').mean()#.reduce(np.nan)
                sim=sim.resample(time='1H').mean()#.reduce(np.nan)
            elif (compar_tim_res=="Day"):
                obs=obs.resample(time='1D').mean()#.reduce(np.nan)
                sim=sim.resample(time='1D').mean()#.reduce(np.nan)
            elif (compar_tim_res=="Year"):
                obs=obs.resample(time='1Y').mean()#.reduce(np.nan)
                sim=sim.resample(time='1Y').mean()#.reduce(np.nan)
            else:
                exit()  
 
            try:    
                station_list['PBIAS'].values[i]=self.pc_bias(sim.values[:],obs.values[:])
                station_list['APB'].values[i]=self.apb(sim.values[:],obs.values[:])
                station_list['RMSE'].values[i]=self.rmse(sim.values[:],obs.values[:])
                station_list['MAE'].values[i]=self.mae(sim.values[:],obs.values[:])
                station_list['BIAS'].values[i]=self.bias(sim.values[:],obs.values[:])
                station_list['NSE'].values[i]=self.NS(sim.values[:],obs.values[:])
                station_list['L'].values[i]=self.L(sim.values[:],obs.values[:])
                station_list['R'].values[i]=self.correlation(sim.values[:],obs.values[:])
                station_list['KGE'].values[i]=self.KGE(sim.values[:],obs.values[:])
            except:
                station_list['PBIAS'].values[i]=-9999.0
                station_list['APB'].values[i]=-9999.0
                station_list['RMSE'].values[i]=-9999.0
                station_list['MAE'].values[i]=-9999.0
                station_list['BIAS'].values[i]=-9999.0
                station_list['NSE'].values[i]=-9999.0
                station_list['L'].values[i]=-9999.0
                station_list['R'].values[i]=-9999.0
                station_list['KGE'].values[i]=-9999.0
            print(station_list['KGE'].values[i])
            station_list.to_csv(f'../cases/{casename}/metrics.csv',index=False)

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
            plt.savefig(f"../cases/{casename}/plot/{station_list['SiteName'][i]}_timeseries.png")
            plt.close(fig)
            #dfx=df['%s'%(obsvarname)]
            #dfx1=dfx.sel(time=slice(f'{startyear}-{startmon}-{startday}T{starthour}',f'{endyear}-{endmon}-{endday}T{endhour}'))
         
            #indata=f'../cases/{casename}/obs/'+f"obs_{station_list['SiteName'][i]}"+f"_{station_list['use_Syear'][i]}"+f"_{station_list['use_Eyear'][i]}.nc"
            #df                            =    xr.open_dataset(indata).squeeze()
            #
