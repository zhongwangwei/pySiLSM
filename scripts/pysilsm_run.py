# -*- coding: utf-8 -*-
__author__ = "Zhongwang Wei / zhongwang007@gmail.com"
__version__ = "0.1"
__release__ = "0.1"
__date__ = "Mar 2023"
import sys,os
import pandas as pd
import xarray as xr
import numpy as np  
import subprocess
import pytz
from timezonefinder import TimezoneFinder
import datetime
from runvalidation import runvalidation
from silsm.Initialization import met_vars,soil_vars,veg_vars,basic_vars,energy_vars
from silsm.shuttleworth_wallace import shuttleworth_wallace 
from silsm.Penman_Monteith import Penman_Monteith
from silsm.write_output import write_output
def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))
def read_namelist(file_path):
    """
    Read a namelist from a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        dict: A dictionary containing the keys and values from the namelist.
    """
    namelist = {}
    current_dict = None

    with open(file_path, 'r') as f:
        for line in f:
            # Ignore comments (lines starting with '#')
            if line.startswith('#'):
                continue
            elif not line:
                continue
            # Ignore if line is empty，or only contains whitespace
            elif not line.strip():
                continue
            else:
                # Check if line starts with '&', indicating a new dictionary
                if line.startswith('&'):
                    dict_name = line.strip()[1:]
                    current_dict = {}
                    namelist[dict_name] = current_dict
                # Check if line starts with '/', indicating the end of the current dictionary
                elif line.startswith('/'):
                    current_dict = None
                # Otherwise, add the key-value pair to the current dictionary
                elif current_dict is not None:
                    line = line.split("#")[0]
                    if  not line.strip():
                        continue
                    else:
                        key, value = line.split('=')
                        current_dict[key.strip()] = value.strip()
                        #if the key value in dict is True or False, set its type to bool
                        if value.strip() == 'True' or value.strip() == 'true' or value.strip() == 'False' or value.strip() == 'false':
                            current_dict[key.strip()] = bool(strtobool(value.strip()))
                        #if the key str value in dict is a positive or negative int (), set its type to int
                        elif value.strip().isdigit() or value.strip().replace('-','',1).isdigit():
                            current_dict[key.strip()] = int(value.strip())
                        #if the key str value in dict is a positive or negative float (), set its type to int or float
                        elif value.strip().replace('.','',1).isdigit() or value.strip().replace('.','',1).replace('-','',1).isdigit():
                            current_dict[key.strip()] = float(value.strip())
                        #else set its type to str
                        else:
                            current_dict[key.strip()] = str(value.strip())
    try:
        #if namelist['General']['Syear'] type is not int, report error and exit
        if not isinstance(namelist['General']['Syear'], int) or not isinstance(namelist['General']['Syear'], int) :
            print('Error: the Syear or Eyear type is not int!')
            sys.exit(1)

        #if namelist['General']['Min_year'] type is not float, report error and exit
        if (not isinstance(namelist['General']['Min_year'], float)):
            print('Error: the Min_year type is not float!')
            sys.exit(1)

        if (not isinstance(namelist['General']['Min_year'], float) or not isinstance(namelist['General']['Min_lat'], float) 
            or not isinstance(namelist['General']['Max_lat'], float) or not isinstance(namelist['General']['Min_lon'], float) or not isinstance(namelist['General']['Max_lon'], float)):
            print('Error: the Min_year or Max_lat or Min_lat or Max_lat or Min_lon or Max_lon type is not float!')
            sys.exit(1)


    except KeyError:
        print('Error: the namelist is not complete!')
        sys.exit(1)
    return namelist

def make_stnlist(nl):
    """
    make a list of the selected sites
    """
    sim_Syear              = nl['General']['Syear']
    sim_Eyear              = nl['General']['Eyear']
    Minimum_lenghth        = nl['General']['Min_year']
    Max_lat                = nl['General']['Max_lat']
    Min_lat                = nl['General']['Min_lat']
    Max_lon                = nl['General']['Max_lon']
    Min_lon                = nl['General']['Min_lon']
    casename               = nl['General']['casename']
    casedir                = nl['General']['casedir']+f"/{casename}/"

    ObsDataDir             = nl['General']['ObsDataDir']
    full_list              = nl['General']['stn_list']

    subprocess.run('mkdir -p '+f"{casedir}/input",shell=True)
    subprocess.run('mkdir -p '+f"{casedir}/obs",shell=True)
    subprocess.run('mkdir -p '+f"{casedir}/list",shell=True)
    subprocess.run('mkdir -p '+f"{casedir}/sim",shell=True)
    subprocess.run('mkdir -p '+f"{casedir}/plot",shell=True)

    station_list = pd.read_csv(full_list,header=0)
    station_list['use_Syear']=[-9999] * len(station_list['obs_Syear'])  #int(station_list['lon']*0 -9999)
    station_list['use_Eyear']=[-9999] * len(station_list['obs_Syear'])
    station_list['lon']=[-9999.0] * len(station_list['obs_Syear'])
    station_list['lat']=[-9999.0] * len(station_list['obs_Syear'])

    output                 =f"{casedir}/list/selected_list.txt"
    for i in range(len(station_list['SiteName'])):
        stn=f"{ObsDataDir}"+"/flux/"+f"{station_list['filename'][i]}"
        stn=f"{ObsDataDir}"+"/flux/"+f"{station_list['filename'][i]}"
        use_Syear=max(station_list['obs_Syear'].values[i],sim_Syear)
        station_list['use_Syear'].values[i]=use_Syear

        print(f"use_syear:{use_Syear}")
        station_list['use_Eyear'].values[i]=min(station_list['obs_Eyear'].values[i],sim_Eyear)
        print(f"use_Eyear:{station_list['use_Eyear'].values[i]}")
        with xr.open_dataset(stn) as df:
            station_list['lon'].values[i]=df['longitude'].values 
            station_list['lat'].values[i]=df['latitude'].values
            print(f"location:{station_list['lon'].values[i]},{station_list['lon'].values[i]}")
        if ((station_list['use_Eyear'].values[i]-use_Syear>=(Minimum_lenghth-1)) &\
                        (station_list['lon'].values[i]>=Min_lon) &\
                        (station_list['lon'].values[i]<=Max_lon) &\
                        (station_list['lat'].values[i]>=Min_lat) &\
                        (station_list['lat'].values[i]<=Max_lat) &\
                        (station_list['Run_Flag'].values[i]==1) 
                ): 
            station_list['Run_Flag'].values[i]=1
            print(station_list['SiteName'].values[i])
        else:
            station_list['Run_Flag'].values[i]=0
            print("exclude: "+station_list['SiteName'].values[i],station_list['lon'].values[i],station_list['lat'].values[i],station_list['use_Eyear'].values[i]-use_Syear,station_list['Run_Flag'].values[i])
    ind = station_list[station_list['Run_Flag']==True].index
    data_select = station_list.loc[ind]
    print(len(data_select['SiteName']))
    data_select.to_csv(output,index=False)
    return

def make_stninput(nl,sinfo):
    dt=nl['General']['dt']
    startx=int(sinfo['use_Syear'])
    endx  =int(sinfo['use_Eyear'])
    stn=f"{nl['General']['ObsDataDir']}/flux/"+f"{sinfo['filename']}"
    with xr.open_dataset(stn) as df:
        try:
            sinfo['canopy_height'] =  float(df.attrs['canopy_height'][:-1]) #df['canopy_height'][:].values
        except:
            sinfo['canopy_height'] =  np.squeeze(df['canopy_height']).values    
          
        sinfo['elevation']=  np.squeeze(df['elevation']).values 
        print(sinfo['elevation'])
        print(sinfo['canopy_height'] )
        print(sinfo['Km'])
        sinfo['reference_height']=np.squeeze(df['reference_height']).values
        if (sinfo['reference_height']<=sinfo['canopy_height']):
            print(['error in reference_height or canopy_height  data'])
            exit()
        print(sinfo)
        temp1=df[['GPP','Qg','Qh','Qh_cor','Qle','Ustar','Rnet','Resp','NEE','Qle_cor']]
        temp2=temp1.sel(time=slice(f'{startx}-01-01',f'{endx}-12-31'))
        temp3=temp2.resample(time=f'{dt}H').mean() #.reduce(np.nan)
        print('stn data read')

    stn1=f"{nl['General']['ObsDataDir']}/"+"/met/"+f"{sinfo['filename'][:-7]}"+"Met.nc"
    with xr.open_dataset(stn1) as df1:
        temp4=df1[['CO2air','LAI','LAI_alternative','LWdown','Precip','Psurf','Qair','RH','SWdown','Tair','VPD','Wind']]
        temp5=temp4.sel(time=slice(f'{startx}-01-01',f'{endx}-12-31'))
        temp6=temp5.resample(time=f'{dt}H').mean() #.reduce(np.nan)
        temp6['esat']= 0.6108*np.exp(17.27*(temp6['Tair']-273.15)/((temp6['Tair']-273.15) + 237.3))  #; %Kpa
        temp6['eair']= 0.6108*np.exp(17.27*(temp6['Tair']-273.15)/((temp6['Tair']-273.15) + 237.3))*(temp6['RH']/100.0)  #; %Kpa
    print('stn1 data read')

    stn2=f"{nl['General']['ObsDataDir']}/"+"/soilmoisture/ERA5LAND_"+f"{sinfo['SiteName']}_{sinfo['obs_Syear']}_{sinfo['obs_Eyear']}_SM.nc" 
    with xr.open_dataset(stn2) as df2:
        temp7  = df2[['swc1','swc2','swc3','swc4','swc_root','swc_surf']]
        temp8  = temp7.sel(time=slice(f'{startx}-01-01',f'{endx}-12-31'))
        temp9  = temp8.resample(time=f'{dt}H').mean()
    print('stn2 data read')

    stn3   = f"{nl['General']['ObsDataDir']}/"+"/soilproperty/"+f"{sinfo['SiteName']}_soil_property.nc" 
    temp10 = xr.open_dataset(stn3)
    print('stn3 data read')

    dk = xr.Dataset({
            'CO2air': (('time', 'lat', 'lon'), temp6['CO2air'].values),
            'LAI': (('time', 'lat', 'lon'), temp6['LAI'].values),
            'LAI_alternative': (('time', 'lat', 'lon'), temp6['LAI_alternative'].values),
            'LWdown': (('time', 'lat', 'lon'), temp6['LWdown'].values),
            'Precip': (('time', 'lat', 'lon'), temp6['Precip'].values),
            'Psurf': (('time', 'lat', 'lon'), temp6['Psurf'].values/1000.),  #Kpa
            'Qair': (('time', 'lat', 'lon'), temp6['Qair'].values),
            'RH': (('time', 'lat', 'lon'), temp6['RH'].values),
            'SWdown': (('time', 'lat', 'lon'), temp6['SWdown'].values),
            'Tair': (('time', 'lat', 'lon'), temp6['Tair'].values),   
            'VPD': (('time', 'lat', 'lon'), temp6['VPD'].values/10.0),   #Kpa
            'esat': (('time', 'lat', 'lon'), temp6['esat'].values ),   #Kpa
            'eair': (('time', 'lat', 'lon'), temp6['eair'].values ),   #Kpa

            'Wind': (('time', 'lat', 'lon'), temp6['Wind'].values),
            'GPP': (('time', 'lat', 'lon'), temp3['GPP'].values),
            'Qg': (('time', 'lat', 'lon'), temp3['Qg'].values),
            'Qh': (('time', 'lat', 'lon'), temp3['Qh'].values),
            'Qh_cor': (('time', 'lat', 'lon'), temp3['Qh_cor'].values),
            'Qle': (('time', 'lat', 'lon'), temp3['Qle'].values),
            'Qle_cor': (('time', 'lat', 'lon'), temp3['Qle_cor'].values),
            'Ustar': (('time', 'lat', 'lon'), temp3['Ustar'].values),
            'Rnet': (('time', 'lat', 'lon'), temp3['Rnet'].values),
            'Resp': (('time', 'lat', 'lon'), temp3['Resp'].values),
            'NEE': (('time', 'lat', 'lon'), temp3['NEE'].values),
            #soil
            'swc1': (('time', 'lat', 'lon'), temp9['swc1'].values),
            'swc2': (('time', 'lat', 'lon'), temp9['swc2'].values),
            'swc3': (('time', 'lat', 'lon'), temp9['swc3'].values),
            'swc4': (('time', 'lat', 'lon'), temp9['swc4'].values),
            'swc_root': (('time', 'lat', 'lon'), temp9['swc_root'].values),
            'swc_surf': (('time', 'lat', 'lon'), temp9['swc_surf'].values),
            'alp_surf': ( ( 'lat', 'lon'),temp10['alp_1'].values),
            'alp_mean': ( ( 'lat', 'lon'),temp10['alp_mean'].values),
            'bld_surf': (( 'lat', 'lon'), temp10['bld_1'].values),
            'bld_mean': (( 'lat', 'lon'), temp10['bld_mean'].values),
            'Ks_surf': ( ( 'lat', 'lon'),temp10['Ks_1'].values),
            'Ks_mean': (( 'lat', 'lon'), temp10['Ks_mean'].values),
            'n_surf': ( ( 'lat', 'lon'),temp10['n_1'].values),
            'n_mean': ( ( 'lat', 'lon'),temp10['n_mean'].values),
            'omega_mean': ( ( 'lat', 'lon'),temp10['omega_mean'].values),
            'omega_zero': ( ( 'lat', 'lon'),temp10['omega_zero'].values),
            'sfc_surf': ( ( 'lat', 'lon'),temp10['sfc_1'].values),
            'sfc_mean': ( ( 'lat', 'lon'),temp10['sfc_mean'].values), #filed capacity of soil saturation
            'sh_surf': ( ( 'lat', 'lon'),temp10['sh_1'].values),
            'sh_mean': ( ( 'lat', 'lon'),temp10['sh_mean'].values),
            'slt_surf': ( ( 'lat', 'lon'),temp10['slt_1'].values),
            'slt_mean': (( 'lat', 'lon'), temp10['slt_mean'].values),
            'sn_surf': (( 'lat', 'lon'), temp10['sn_1'].values),
            'sn_mean': ( ( 'lat', 'lon'),temp10['sn_mean'].values),
            'snd_surf': (( 'lat', 'lon'), temp10['snd_1'].values),
            'snd_mean': ( ( 'lat', 'lon'),temp10['snd_mean'].values),
            'wiltingpoint_surf': ( ( 'lat', 'lon'),temp10['sw_1'].values),
            'wiltingpoint_mean': ( ( 'lat', 'lon'),temp10['sw_mean'].values),    
            'snd_surf': ( ( 'lat', 'lon'),temp10['snd_1'].values),
            'snd_mean': ( ( 'lat', 'lon'),temp10['snd_mean'].values),
               
             'canopy_height': (sinfo['canopy_height']),
             'elevation'    : (sinfo['elevation']),
             'reference_height': (sinfo['reference_height']),
             'IGBP_CLASS'   : (sinfo['IGBP_CLASS']),
             'z_T': (sinfo['z_T']),
             'z_U': (sinfo['z_U']),
             'dt':  (sinfo['dt']),
             'C3': (sinfo['C3']),
             'D0': (sinfo['D0']),
             'rss_v': (sinfo['rss_v']),
             'g_min_c': (sinfo['g_min_c']),
             'Kr': (sinfo['Kr']),
             'dl': (sinfo['dl']),
             'z0_soil': (sinfo['z0_soil']),
             'Km': (sinfo['Km']),
             'cd': (sinfo['cd']),
        },
        coords={'time': (('time'), temp6['time'].values),
                'lat': (('lat'), [sinfo['lat']]),
                'lon': (('lon'), [sinfo['lon']]),
                 })
    dk.to_netcdf(f"{nl['General']['casedir']}/{nl['General']['casename']}/input/"+f"input_{sinfo['SiteName']}"+f"_{sinfo['use_Syear']}"+f"_{sinfo['use_Eyear']}.nc",engine='netcdf4')
    del df,df1,df2, temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,dk #,ds

if __name__=='__main__':
    nl = read_namelist('define.nml')
    print(nl)
    casename=nl['General']['casename']
    casedir=nl['General']['casedir']+f"/{casename}"
    make_stnlist(nl)
    stnlist                 =f"{casedir}/list/selected_list.txt"
    print(stnlist)
    station_list = pd.read_csv(stnlist,header=0)

    for i in np.arange(len(station_list['use_Syear'])):
        print(f"processing site: {station_list['SiteName'][i]}")
        sinfo=station_list.iloc[i]
        print(sinfo)
        indata=f'{casedir}/input/'+f"input_{station_list['SiteName'][i]}"+f"_{station_list['use_Syear'][i]}"+f"_{station_list['use_Eyear'][i]}.nc"
        #if indata is not exist, print error and exit
        if not os.path.exists(indata):
            print(f'Error: the input file {indata} is not exist!')
            print('prepare the input file first')
            make_stninput(nl,sinfo)
        df                            =    xr.open_dataset(indata).squeeze()
        bas=basic_vars(df,nl)
        met=met_vars(df,nl)
        soil=soil_vars(df,nl)
        veg=veg_vars(df,bas,met,nl)
        ene=energy_vars(df,met,bas,veg,nl)
        print('finish initialization')
        print("=======================================================================================") 
        print("")
        print("")
        print("Start running the model")
        print("")
        print("")
        print("=======================================================================================")
        print(nl['ModSet']['ET_model'])
        #if nl['ModSet']['ET_model'] is = 'Penman_Monteith'
        #run Penman_Monteith
        if nl['ModSet']['ET_model']== 'shuttleworth_wallace':
            pp1=shuttleworth_wallace(met,bas,soil,veg,ene,nl)
            bas_r, met_r, soil_r, veg_r, ene_r  = pp1.shuttleworth_wallace()
            pp2=write_output(bas_r, met_r, soil_r, veg_r, ene_r,nl,station_list['SiteName'][i])
            pp2.write_nc()
        elif (nl['ModSet']['ET_model']== 'Penman_Monteith'):
            print('start running Penman_Monteith')
            pp1=Penman_Monteith(met,bas,soil,veg,ene,nl)
            bas_r, met_r, soil_r, veg_r, ene_r = pp1.Penman_Monteith()
            pp2=write_output(bas_r, met_r, soil_r, veg_r, ene_r,nl,station_list['SiteName'][i])
            pp2.write_nc()
        print('finish running the model')
        p1=runvalidation(casename,'Qle_cor','LE',nl['General']['compare_res'])
        p1.validation()
        print("=======================================================================================")

        

