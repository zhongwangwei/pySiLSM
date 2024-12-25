import silsm.metlib as metlib
import silsm.resistance as reslib
from silsm.physiology import canopy_conductance_Jarvis1976
import numpy as np
class Penman_Monteith:
    def __init__(self,met,bas,soil,veg,ene,nl):
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

    def Penman_Monteith(self):
        self.met.rav=reslib.calc_R_A_Norman1995(self.bas.z_T, self.met.Ustar, 
                                                       self.met.ObukhovLength, self.met.d0, self.met.z0m)
        Ga = 1./self.met.rav
        if self.nl['ModSet']['Canopy_conductance_model'] == 'Jarvis1976':
            rsmin = 72.      # sm-1  see Alfieri et al. 2008 abstract
            gD    = 0.1914   # Kpa  see Alfieri et al. 2008  
            p1=canopy_conductance_Jarvis1976(self.ene.SWdown, self.soil.FC, self.soil.swc_root, self.soil.WP, gD, 
                                             self.met.VPD, rsmin, self.met.Tair_K, self.veg.LAI)
            rc=p1.canopy_conductance()
            r_sfc=rc*self.veg.LAI/(0.3*self.veg.LAI+1.2)
        Gs=1./rc  #r_sfc*2

        LE = (self.met.delta * (self.ene.Rnet - self.ene.Qg) + self.met.rho *self.met.Cp * self.met.VPD * Ga) / \
                 (self.met.delta + self.met.Psy  * (1 + Ga / Gs))

        Gs_test = Ga/(((self.met.delta * (self.ene.Rnet - self.ene.Qg) + self.met.rho *self.met.Cp * self.met.VPD * Ga)/self.ene.Qle_cor-self.met.delta )/self.met.Psy-1)
        print('check gs')
        print(np.mean(self.ene.Rnet))
        print('check gs done')

        
        HE=self.ene.Rnet-LE-self.ene.Qg
        self.ene.HE=HE
        self.ene.LE=LE
        return self.bas, self.met, self.soil, self.veg, self.ene