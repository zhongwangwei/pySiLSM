# -*- coding: utf-8 -*-
__author__ = "Zhongwang Wei / zhongwang007@gmail.com"
__version__ = "1.0"
__release__ = "0.1"
__date__ = "Jun 2022"

import xarray as xr
from datetime import datetime
from datetime import timedelta
import pandas as pd
import glob, os, shutil,sys
import subprocess
def stn_list_initialization(Stationlist_dir,Vdata,Sim_spatial_res,Min_UpArea):
    if (Vdata=='GRDC'):
        station_list = pd.read_csv(f"{Stationlist_dir}/{Vdata}/list/{Vdata}_alloc_{Sim_spatial_res}.txt",delimiter=r"\s+",header=0)
    elif(Vdata=='FLUXNET'):
        station_list = pd.read_csv(f"{Stationlist_dir}/LIST_FLUXNET_PLUMBER2.csv",header=0)
    #initialization
    station_list['Flag']         = [False]   * len(station_list['ID'])        #[False for i in range(len(station_list['lon']))] #[False] * len(station_list['lon'])  #(station_list['lon']*0 -9999)*False
    station_list['use_Sdate']    = [-9999]   * len(station_list['ID'])          #int(station_list['lon']*0 -9999)
    station_list['use_Edate']    = [-9999]   * len(station_list['ID'])
    station_list['obs_Sdate']    = [-9999]   * len(station_list['ID'])
    station_list['obs_Edate']    = [-9999]   * len(station_list['ID'])
    if (('lat' in station_list.columns )):
        print('lat is avaliable' )
    else:
        station_list['lat']                = [-9999.0] * len(station_list['ID'])
    if (('lon' in station_list.columns )):
        print('lon is avaliable' )
    else:
        station_list['lon']                = [-9999.0] * len(station_list['ID'])

    if (('filename' in station_list.columns )):
        print('filename is avaliable is station_list.columns' )
    else:
        station_list['filename']                = [-9999.0] * len(station_list['ID'])
    
    if (Vdata=='GRDC'):
        for i in range(len(station_list['ID'])):
            if (compar_tim_res=="Month"):
                station_list.loc[i,'filename']=f"{station_list['ID'][i]}_Q_{compar_tim_res}.nc"
            elif (compar_tim_res=="Day"):
                station_list.loc[i,'filename']=f"{station_list['ID'][i]}_Q_{compar_tim_res}.Cmd.nc"
    elif (Vdata=='FLUXNET'):
        station_list['area1']                = Min_UpArea+1.0
        station_list['ix2']                  = -9999
    
    return station_list

if __name__=='__main__':
    #creat a configure file may be necesarry!
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
    Min_year               = float(argv[18])
    Max_lat                = float(argv[19])
    Min_lat                = float(argv[20])
    Max_lon                = float(argv[21])
    Min_lon                = float(argv[22])
    Max_UpArea             = float(argv[23])
    Min_UpArea             = float(argv[24])
    
 
    station_list=stn_list_initialization(Stationlist_dir,Vdata,Sim_spatial_res,Min_UpArea)

    for i in range(len(station_list['ID'])):  
        if (Vdata=='FLUXNET'):
            stn=f"{Obs_data_dir}"+"/"+f"{station_list['filename'][i]}"  
            with xr.open_dataset(stn) as df:
                station_list.loc[i,'lon']=df['longitude'][0,0].values
                station_list.loc[i,'lat']=df['latitude'][0,0].values
               # print( station_list.loc[i,'lon'], station_list.loc[i,'lat'])
            del df
                #df.close()
        elif (Vdata=='GRDC'):
            if(os.path.exists(f'{Obs_data_dir}/'+f"{station_list['filename'][i]}")):
                stn=f'{Obs_data_dir}/'+f"{station_list['filename'][i]}"
                
        try:
            with xr.open_dataset(stn) as df:
                ts = int((pd.to_datetime(str(df.time.values[0]))).strftime('%Y%m%d%H'))
                ts1 = int((pd.to_datetime(str(df.time.values[-1]))).strftime('%Y%m%d%H'))
                station_list.loc[i,'use_Sdate']=max(ts,sim_sdate)
                station_list.loc[i,'use_Edate']=min(ts1,sim_edate)
                start_dt = pd.to_datetime(str(station_list.loc[i,'use_Sdate']), format='%Y%m%d%H')
                end_dt   = pd.to_datetime(str(station_list.loc[i,'use_Edate']), format='%Y%m%d%H')
                dt = timedelta(hours=1)
                sim_time = (end_dt-start_dt)/dt
       
                if ((sim_time>=24*365*Min_year-1) &\
                    (station_list.loc[i,'lon']>=Min_lon) &\
                    (station_list.loc[i,'lon']<=Max_lon) &\
                    (station_list.loc[i,'lat']>=Min_lat) &\
                    (station_list.loc[i,'lat']<=Max_lat) &\
                    (station_list.loc[i,'area1']>=Min_UpArea) &\
                    (station_list.loc[i,'area1']<=Max_UpArea) &\
                    (station_list.loc[i,'ix2'] == -9999) 
                    ): 
                    try:
                        df['%s'%(obsvarname)].values[0]-1.0
                        station_list['Flag'].values[i]=True
                    except:
                        station_list['Flag'].values[i]=False

                    print('Station %s will be added !'%station_list['ID'].values[i])
                df.close()

            del ts,ts1,start_dt,end_dt,sim_time
            del(stn)
        except:
            station_list['Flag'].values[i]=False
    output      = f"cases/{casename}_{compar_tim_res}/selected_list_{Model}_{Vdata}.csv"
    ind         = station_list[station_list['Flag']==True].index
    data_select = station_list.loc[ind]
    print(' %s in total !'%len(data_select['ID']))
    
    data_select.to_csv(output,index=False)

