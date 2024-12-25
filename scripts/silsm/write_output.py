import xarray as xr
import numpy as np

class write_output:
    def __init__(self,bas,met,soil,veg,ene,nl,filename):
        self.name    = '******'
        self.version = '0.1'
        self.release = '0.1'
        self.date    = 'Mar 2023'
        self.author  = "Zhongwang Wei / zhongwang007@gmail.com"
        self.bas     = bas
        self.met     = met
        self.soil    = soil
        self.veg     = veg
        self.ene     = ene
        self.nl      = nl
        self.filename= filename

    def write_nc(self):
        '''
        write output to netcdf file
        '''   
        lonx =  [self.bas.lon]
        latx =  [self.bas.lat]
        time1= self.bas.time.values
        LE=self.ene.LE.reshape((len(time1),len(latx),len(lonx)))
        HE=self.ene.HE.reshape((len(time1),len(latx),len(lonx)))
        # Create a new dataset
        ds = xr.Dataset({'LE': (('time','lat','lon'), LE),
                         'HE': (('time','lat','lon'), HE),},
                 coords={'lon': lonx, 'lat':latx,'time': (('time'), time1)})
        #create longitude attrs
        ds.lon.attrs['Long name']                                   = "Longitude"
        ds.lon.attrs['units']                                       = "Degrees_east"

        # create latitude attrs
        ds.lat.attrs['Long name']                                    = "Latitude"
        ds.lat.attrs['units']                                        = "Degrees_north"

        # creat LE and HE attrs
        ds.LE.attrs['Long name']                                  = "Latent heat flux"
        ds.LE.attrs['units']                                      = "w/m2"

        ds.HE.attrs['Long name']                                  = "Sensible heat flux"
        ds.HE.attrs['units']                                      = "w/m2"


        if self.nl['outputs']['LEs']:
            #add LEs to the dataset
            LEs=self.ene.LEs.reshape((len(time1),len(latx),len(lonx)))
            ds['LEs'] = (('time','lat','lon'), LEs)
            # creat LEs attrs
            ds.LEs.attrs['Long name']                                  = "Latent heat flux from soil"
            ds.LEs.attrs['units']                                      = "w/m2"
        if self.nl['outputs']['HEs']:
            #add HEs to the dataset
            HEs=self.ene.HEs.reshape((len(time1),len(latx),len(lonx)))
            ds['HEs'] = (('time','lat','lon'), HEs)
            # creat HEs attrs
            ds.HEs.attrs['Long name']                                  = "Sensible heat flux from soil"
            ds.HEs.attrs['units']                                      = "w/m2"
        if self.nl['outputs']['LEi']:
            #add HEi to the dataset
            LEi=self.ene.LEi.reshape((len(time1),len(latx),len(lonx)))
            ds['LEi'] = (('time','lat','lon'), LEi)
            # creat HEs attrs
            ds.LEi.attrs['Long name']                                  = "Latent heat flux from canopy interception"
            ds.LEi.attrs['units']                                      = "w/m2"
        if self.nl['outputs']['HEc']:
            #add HEc to the dataset
            HEc=self.ene.HEc.reshape((len(time1),len(latx),len(lonx)))
            ds['HEc'] = (('time','lat','lon'), HEc)
            # creat HEc attrs
            ds.HEc.attrs['Long name']                                  = "Sensible heat flux from canopy"
            ds.HEc.attrs['units']                                      = "w/m2"
        if self.nl['outputs']['LEc']:
            #add LEc to the dataset
            LEc=self.ene.LEc.reshape((len(time1),len(latx),len(lonx)))
            ds['LEc'] = (('time','lat','lon'), LEc)
            # creat LEc attrs
            ds.LEc.attrs['Long name']                                  = "Latent heat flux from canopy"
            ds.LEc.attrs['units']                                      = "w/m2"
        
        if self.nl['outputs']['LAI']:
            #add LEc to the dataset
            LAI=self.veg.LAI.reshape((len(time1),len(latx),len(lonx)))
            ds['LAI'] = (('time','lat','lon'), LAI)
            # creat LEc attrs
            ds.LAI.attrs['Long name']                                  = "leaf area index"
            ds.LAI.attrs['units']                                      = "m2/m2"

        if self.nl['outputs']['swc_surf']:
            #add LEc to the dataset
            swc_surf=self.soil.swc_surf.reshape((len(time1),len(latx),len(lonx)))
            ds['swc_surf'] = (('time','lat','lon'), swc_surf)
            # creat LEc attrs
            ds.swc_surf.attrs['Long name']                                  = "soil surface water content"
            ds.swc_surf.attrs['units']                                      = "m3/m3"

        if self.nl['outputs']['swc_root']:
            #add LEc to the dataset
            swc_root=self.soil.swc_root.reshape((len(time1),len(latx),len(lonx)))
            ds['swc_root'] = (('time','lat','lon'), swc_root)
            # creat LEc attrs
            ds.swc_root.attrs['Long name']                                  = "soil root water content"
            ds.swc_root.attrs['units']                                      = "m3/m3"
        if self.nl['outputs']['fwet']:
            #add fwet to the dataset
            fwet=self.met.fwet.reshape((len(time1),len(latx),len(lonx)))
            ds['fwet'] = (('time','lat','lon'), fwet)
            # creat LEc attrs
            ds.fwet.attrs['Long name']                                  = "wet fraction of the canopy"
            ds.fwet.attrs['units']                                      = "m2/m2"            

        if self.nl['outputs']['RH']:
            #add fwet to the dataset
            RH=self.met.RH.reshape((len(time1),len(latx),len(lonx)))
            ds['RH'] = (('time','lat','lon'), RH)
            # creat LEc attrs
            ds.RH.attrs['Long name']                                  = "relative humidity"
            ds.RH.attrs['units']                                      = "-"      
        print(self.nl)
        ds.to_netcdf(f"{self.nl['General']['casedir']}/{self.nl['General']['casename']}/sim/{self.filename}.nc")
        
        

        return  
