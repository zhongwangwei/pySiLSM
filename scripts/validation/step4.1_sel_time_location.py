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



def check_namelist(casename,Model,Vdata):
    stnlist                 =f"cases/{casename}_{compar_tim_res}/selected_list_{Model}_{Vdata}.csv"
    station_list = pd.read_csv(stnlist,header=0)
    station_num=len(station_list['use_Sdate'])
    minyear=min(np.asarray(station_list['use_Sdate']))
    maxyear=max(np.asarray(station_list['use_Edate']))
    minyear=int(str(minyear)[0:4])
    maxyear=int(str(maxyear)[0:4])
    print(minyear,maxyear,station_num) 
    print("---------------------------------------------------------------------------------------")
    os.makedirs(f'cases/{casename}_{compar_tim_res}/tmp/sim', exist_ok=True)
    os.makedirs(f'cases/{casename}_{compar_tim_res}/tmp/obs', exist_ok=True)
    if (Sim_tim_res=="Month"):
        if (compar_tim_res=="Hour"):
            print('Sim_tim_res="Month"&compar_tim_res="Hour"')
            exit() 
        elif (compar_tim_res=="Day"):
            print('Sim_tim_res="Month"&compar_tim_res="Hour"')
            exit() 
    elif (Sim_tim_res=="Year"):
        if (compar_tim_res=="Hour"):
            print('Sim_tim_res="Year"&compar_tim_res="Hour"')
            exit() 
        elif (compar_tim_res=="Day"):
            print('Sim_tim_res="Year"&compar_tim_res="Day"')
            exit()  
        elif (compar_tim_res=="Month"):
            print('Sim_tim_res="Year"&compar_tim_res="Month"')
            exit() 
    elif (Sim_tim_res=="Day"):
        if (compar_tim_res=="Hour"):
            print('Sim_tim_res="Day"&compar_tim_res="Hour"')
            exit() 
    subprocess.run('rm '+f'cases/{casename}_{compar_tim_res}/tmp/obs/*',shell=True)
    subprocess.run('rm '+f'cases/{casename}_{compar_tim_res}/tmp/sim/*',shell=True)
    return station_list,minyear,maxyear,station_num
    
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
  
    station_list,minyear,maxyear,station_num=check_namelist(casename,Model,Vdata)

    for i in range(len(station_list['use_Sdate'])):
        stn=f"{Obs_data_dir}"+"/"+f"{station_list['filename'][i]}"
        print(f"processing observation site: {station_list['ID'][i]}")
        with xr.open_dataset(stn) as df:
            startyear=str(station_list['use_Sdate'].values[i])[0:4]
            startmon=str(station_list['use_Sdate'].values[i])[4:6]
            startday=str(station_list['use_Sdate'].values[i])[6:8]
            starthour=str(station_list['use_Sdate'].values[i])[8:10]

            endyear=str(station_list['use_Edate'].values[i])[0:4]
            endmon=str(station_list['use_Edate'].values[i])[4:6]
            endday=str(station_list['use_Edate'].values[i])[6:8]
            endhour=str(station_list['use_Edate'].values[i])[8:10]

            dfx=df['%s'%(obsvarname)]
            dfx1=dfx.sel(time=slice(f'{startyear}-{startmon}-{startday}T{starthour}',f'{endyear}-{endmon}-{endday}T{endhour}'))
            if (compar_tim_res=="Month"):
                dfx2=dfx1.resample(time='1M').mean()#.reduce(np.nan)
            elif (compar_tim_res=="Hour"):
                dfx2=dfx1.resample(time='1H').mean()#.reduce(np.nan)
            elif (compar_tim_res=="Day"):
                dfx2=dfx1.resample(time='1D').mean()#.reduce(np.nan)
            elif (compar_tim_res=="Year"):
                dfx2=dfx1.resample(time='1Y').mean()  #.reduce(np.nan)
            else:
                exit()   
            dfx2.to_netcdf(f'cases/{casename}_{compar_tim_res}/tmp/obs/'+f"obs_{station_list['ID'][i]}"+f"_{station_list['use_Sdate'][i]}"+f"_{station_list['use_Edate'][i]}.nc",engine='netcdf4')
            del dfx,dfx2,dfx1
        del df
    print("observation data done")

#deal with simulation data
    for ii in range((minyear),(maxyear)+1):
        if (Model=='CoLM'):
            if (Vdata=='GRDC'):
                VarFile=('%s/%s_hist_cama_%s*.nc'%(Sim_data_dir,casename,ii))
            elif(Vdata=='FLUXNET'):
                VarFile=('%s/%s_hist_%s*.nc'%(Sim_data_dir,casename,ii))
        elif(Model=='CaMa'):
            VarFile=('%s/o_%s%s.nc'%(Sim_data_dir,simvarname,ii))
        with xr.open_mfdataset(VarFile, combine='nested',concat_dim="time",parallel=True,decode_times=False) as combined:
            dfx=combined['%s'%(simvarname)]  
        num=len(dfx['time'])
#should be rewriten here
        if (Sim_tim_res=="Hour"):
            dfx['time'] = pd.date_range(f"{ii}-01-01", freq="H", periods=num)
        elif (Sim_tim_res=="Day"):
            dfx['time'] = pd.date_range(f"{ii}-01-01", freq="D", periods=num)
        elif (Sim_tim_res=="Month"):
            dfx['time'] = pd.date_range(f"{ii}-01-01", freq="M", periods=num)

        if (compar_tim_res=="Month"):
            dfx=dfx.resample(time='1M').mean() 
        elif (compar_tim_res=="Day"):
            dfx=dfx.resample(time='1D').mean() 
        elif (compar_tim_res=="Hour"):
            dfx=dfx.resample(time='1H').mean() 
        elif (compar_tim_res=="Year"):
            dfx=dfx.resample(time='1Y').mean()  
        else:
            exit()                 
        dfx.to_netcdf(f'cases/{casename}_{compar_tim_res}/tmp/sim/'+f'sim_{ii}.nc')
        print(f'Year {ii}: Simulation Files Combined')
        del combined


    VarFile=(f'cases/{casename}_{compar_tim_res}/tmp/sim/sim_*.nc')
    with xr.open_mfdataset(VarFile, combine='nested',concat_dim="time",parallel=True,autoclose=True) as ds1:
        ds1.to_netcdf(f'cases/{casename}_{compar_tim_res}/tmp/sim/'+f'{simvarname}_sim.nc')
 #       ds1.close()
    print(f'Simulation Files Combined')
    subprocess.run('rm '+VarFile,shell=True)


    with xr.open_dataset(f'cases/{casename}_{compar_tim_res}/tmp/sim/'+f'{simvarname}_sim.nc') as simx:
        #print(simx)
        for ik in range(len(station_list['use_Sdate'])):
            startyear=str(station_list['use_Sdate'].values[ik])[0:4]
            startmon=str(station_list['use_Sdate'].values[ik])[4:6]
            startday=str(station_list['use_Sdate'].values[ik])[6:8]
            starthour=str(station_list['use_Sdate'].values[ik])[8:10]
         
            endyear=str(station_list['use_Edate'].values[ik])[0:4]
            endmon=str(station_list['use_Edate'].values[ik])[4:6]
            endday=str(station_list['use_Edate'].values[ik])[6:8]
            endhour=str(station_list['use_Edate'].values[ik])[8:10]
            if (Vdata=='GRDC'):
                simx1 = simx['%s'%(simvarname)][:,station_list['iy1'].values[ik] - 1, station_list['ix1'].values[ik] - 1]
                #simx1=simx.sel(lat_cama=[station_list['lat'].values[ik]], lon_cama=[station_list['lon'].values[ik]], method="nearest")
            elif(Vdata=='FLUXNET'):
                simx1=simx.sel(lat=[station_list['lat'].values[ik]], lon=[station_list['lon'].values[ik]], method="nearest")
            simx2=simx1.sel(time=slice(f'{startyear}-{startmon}-{startday}T{starthour}',f'{endyear}-{endmon}-{endday}T{endhour}'))
            simx2.to_netcdf(f'cases/{casename}_{compar_tim_res}/tmp/sim/'+f"sim_{station_list['ID'][ik]}"+f"_{station_list['use_Sdate'][ik]}"+f"_{station_list['use_Edate'][ik]}.nc",engine='netcdf4')
            del simx1,simx2
            print (f"sim_{station_list['ID'][ik]}"+f"_{station_list['use_Sdate'][ik]}"+f"_{station_list['use_Edate'][ik]}.nc"+"  is ready!!")
        simx.close()

