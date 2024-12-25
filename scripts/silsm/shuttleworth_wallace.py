import numpy as np

from collections import deque
import time
import silsm.metlib as metlib
import silsm.energy as enelib
import silsm.resistance as reslib
from silsm.physiology import canopy_conductance_Leuning1995,canopy_conductance_Liang2022
# ==============================================================================
# List of constants used in Meteorological computations
# ==============================================================================

# gas constant for dry air, J/(kg*degK)
R_d = 287.04
# acceleration of gravity (m s-2)
GRAVITY = 9.8
# mimimun allowed friction velocity
U_FRICTION_MIN = 0.01

# Maximum number of interations
ITERATIONS = 40

# Arithmetic error. BAD data, it should be discarded
F_INVALID = 255  

F_ALL_FLUXES = 0  # All fluxes produced with no reduction of PT parameter (i.e. positive soil evaporation)

#??
TAUD_STEP_SIZE_DEG = 5

LOWEST_TC_DIFF = 5.  # Lowest Canopy to Air temperature difference
LOWEST_TS_DIFF = 5.  # Lowest Soil to Air temperature difference
F_LOW_TS_TC = 254  # Low Soil and Canopy Temperature flag
F_LOW_TS = 253  # Low Soil Temperature flag
F_LOW_TC = 252  # Low Canopy Temperature flag
T_DIFF_THRES = 0.1
STABILITY_THRES = -0.01


# Soil heat flux formulation constants
G_CONSTANT = 0
G_RATIO = 1
G_TIME_DIFF = 2
G_TIME_DIFF_SIGMOID = 3

# Size of the normalized height bins in Massman wind profile
BIN_SIZE = 0.0001

U_C_MIN = 0.01

UNSTABLE_THRES = None
STABLE_THRES = None



# Threshold for relative change in Monin-Obukhov lengh/ temperature to stop the iterations
L_thres = 0.001
T_thres = 0.1
class shuttleworth_wallace:
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

    #TODO: this function is not finished yet, need to rewrite it
    def Tc_Taylor_Expansion_theory(self,Ta,Rh,rsc,rav,Tr,P,Cp,Psy):
        '''
        Canopy temperature calulated using Taylor expansion theory    
        '''
        ea_air   = 0.6108 * np.exp(17.27*Ta / (Ta + 237.3)) * Rh        # %Kpa
        esat_air = 0.6108 * np.exp(17.27*Ta / (Ta + 237.3))             # %Kpa
        rhoa = 1.293 * 273.15 / (273.15 + Ta ) * P / 101.325 * (1.0 - 0.378 * ea_air / P)



        diff_eas=-(1527.0*np.exp((1727*Ta)/(100.*(Ta + 237.3)))*((1727.0*Ta)/(100.0*(Ta + 237.3)**2.0) - 1727./(100.*(Ta + 237.3))))/2500.0
        gw=1.0 / (rav + rsc)

        DD=Tr * Psy /(Cp * rhoa * gw)
        D=esat_air - ea_air
        Tc=(DD - D) / diff_eas + Ta
        return Tc

    def ETsw(self,Zc,delta,Psy,rou,A,Cp,VPD,As,rac,rss,ras,raa,rsc2,fwet,Rnc):
        Rc     = (delta+Psy)*rac + Psy*rsc2
        Rs     = (Zc+Psy)*ras + Psy*rss
        Ra     = (Zc+Psy)*raa  
        wc     =  1.0 / (1.0 + Rc * Ra / ( Rs  * ( Rc + Ra)))  #(indata['Zc+indata['Psy)*indata['raa  
        ws     =  1.0 / (1.0 + Rs  * Ra / ( Rc * (Rs + Ra)))
        PMc    = (delta * A + (rou * Cp * VPD-delta * rac * As) /
                      ( raa+rac))/(delta+Psy * (1+rsc2/( raa+ rac))) 
        PMs    =  (delta * A + (rou * Cp * VPD-delta * ras * (A-As)) / 
                      (raa+ras)) / (delta+Psy * (1+rss/( raa+ ras)))
        Transpirtion = wc *  PMc*(1-fwet)   
        SoilEvap     = ws *  PMs
        CanopyEvap   = fwet*1.26*Rnc*delta/(delta+Psy)
        return Transpirtion, SoilEvap, CanopyEvap

    def shuttleworth_wallace(self):
        '''
        Shuttleworth and Wallace [Shuttleworth1995]_ dual source energy combination model.
        Calculates turbulent fluxes using meteorological and crop data for a
        dual source system in series.

   
        References
        ----------
        [Shuttleworth1995] W.J. Shuttleworth, J.S. Wallace, Evaporation from
            sparse crops - an energy combinatino theory,
            Quarterly Journal of the Royal Meteorological Society , Volume 111, Issue 469,
            Pages 839-855,
            http://dx.doi.org/10.1002/qj.49711146910.
        '''    


        # Initially assume stable atmospheric conditions and set variables for
        # iteration of the Monin-Obukhov length
   
        # Initially assume stable atmospheric conditions and set variables for
        T_A_K   =  self.met.Tair_K
        T_C_K   =  self.met.Tair_K - 2.0
        T_C_K_0 =  T_C_K+10.0
        iterations=np.zeros(T_A_K.shape, np.float32)+np.NaN
        flag=np.zeros(T_A_K.shape, np.float32)+np.NaN
        max_iterations = ITERATIONS
    
        Tck_converged = np.asarray(np.zeros(T_A_K.shape)).astype(bool)
        Tck_diff_max  = np.inf
        Tck_diff      = np.abs(T_C_K-T_C_K_0)

        # Outer loop for estimating stability.
        # Stops when difference in consecutives L is below a given threshold
        start_time = time.time()
        loop_time = time.time()
        #water stress factor
        beta            =      (self.soil.swc_root-self.soil.WP)/(self.soil.FC-self.soil.WP)
        beta            =      np.where(beta > 1.0, 1.0, beta  )  #for C4
        beta            =      np.where(beta < 0.0, 0.0, beta   )  #for C4
        f5              =      2.0 * beta  - beta  * beta  

        for n_iterations in range(max_iterations):
            if np.all(np.abs(Tck_diff) < T_thres):
                print(f"Finished iteration with a max. Tck_diff: {np.max(Tck_diff)}")
                break
            print(f'number of iterations: {n_iterations}')
            i = np.logical_and(Tck_diff >= L_thres, flag != F_INVALID)

            print(f"Number of points to iterate: {np.sum(i)}")
            self.met.rss[i]=reslib.calc_r_ss_sellers1992(self.soil.swc_surf[i],self.met.rss_v[i])

            self.met.rav[i]=reslib.calc_R_A_Norman1995(self.bas.z_T[i], self.met.Ustar[i], 
                                                       self.met.ObukhovLength[i], self.met.d0[i], self.met.z0m[i])
                
            self.met.rac[i]=reslib.calc_rac_Shuttleworth1990(self.veg.LAI[i], self.veg.dl[i], self.veg.Zc[i], self.bas.z_U[i], 
                                                                self.met.Wind[i], self.met.Km[i])

            self.met.raa[i]=reslib.calc_raa_Shuttleworth1990(self.met.Ustar[i],self.veg.Zc[i],self.bas.z_U[i],self.met.z0m[i],self.met.d0[i],self.met.Km[i])

            self.met.ras[i]=reslib.calc_ras_Shuttleworth1990(self.met.Ustar[i],self.veg.Zc[i],self.met.z0_soil[i],self.met.d0[i],self.met.Km[i],self.met.z0m[i])

                # Calculate the bulk canopy resistance
            if self.nl['ModSet']['Canopy_conductance_model'] == 'Leuning1995':
                p1=canopy_conductance_Leuning1995(self.ene.PAR_t[i],self.soil.swc_root[i],self.veg.LAI[i],
                                                      self.veg.C_s[i],self.soil.WP[i],
                                                      self.soil.FC[i],self.veg.Kr[i],self.veg.g_m_298[i],
                                                      self.veg.Q_10_g_m[i],self.veg.T_1_g_m[i],self.veg.T_2_g_m[i],
                                                      self.veg.Gamma_298[i],self.veg.alpha_0[i],
                                                      self.veg.A_m_max_298[i],self.veg.Q_10_A_m[i],self.veg.T_1_A_m[i],
                                                      self.veg.T_2_A_m[i],self.veg.g_min_c[i],
                                                      self.veg.a_1[i],self.veg.D0[i],T_C_K[i],self.veg.C_i[i], self.veg.D_s[i],f5[i])
                    
                self.veg.A_g[i],self.veg.A_n[i],self.veg.A_m[i],self.veg.g_m[i],self.veg.g_cc[i], self.veg.Gamma[i], self.veg.d1[i],self.veg.d2[i],self.met.R_d[i] =p1.canopy_conductance()
                self.veg.g_cw[i]=self.veg.g_cc[i]*1.6
                    
                self.veg.rsc[i] =1.0/self.veg.g_cw[i]*1000.0

                #TODO: need to set a range for rsc
                self.veg.rsc[i] = np.where(self.veg.rsc[i]<0.0, 0.0, self.veg.rsc[i])  
                self.veg.rsc[i] = np.where(self.veg.rsc[i]>100000.0, 100000.0, self.veg.rsc[i])
                self.ene.LEc[i],self.ene.LEs[i],self.ene.LEi[i]= self.ETsw(self.veg.Zc[i],self.met.delta[i],self.met.Psy[i],
                                                                     self.met.rho[i],self.ene.A[i],self.met.Cp[i], 
                                                                     self.met.VPD[i],self.ene.As[i],self.met.rac[i], self.met.rss[i], 
                                                                     self.met.ras[i], self.met.raa[i],self.veg.rsc[i],self.met.fwet[i],self.ene.Rnc[i])
                self.ene.HEs[i]=self.ene.As[i]-self.ene.LEs[i]
                self.ene.HEc[i]=(self.ene.A[i]-self.ene.As[i])-self.ene.LEc[i]-self.ene.LEi[i]

                T_C_K[i]    = self.Tc_Taylor_Expansion_theory(T_A_K[i]-273.15,self.met.RH[i],self.veg.rsc[i],self.met.rav[i],self.ene.LEc[i], self.met.Psurf[i], self.met.Cp[i],self.met.Psy[i]) + 273.15
          
                self.veg.C_i[i]         = self.veg.C_s[i]  -self.veg.A_n[i]  / (self.veg.g_cc[i] / 1000.0) # % converting g_cc to m/s
                self.veg.C_i_min[i]     = self.veg.C_s[i] - self.veg.A_n[i]  / (self.veg.g_cc[i] / 1000.0 + self.veg.A_g[i]  /(  self.veg.C_s[i]  - self.veg.Gamma[i])) #; % converting g_cc to m/s
                self.veg.C_i_min[i]     = np.where( self.veg.C_i_min[i]<0.0,400.0, self.veg.C_i_min[i])  
                self.veg.C_i[i]         = np.where( self.veg.C_i[i]-self.veg.C_i_min[i]<0.0,self.veg.C_i_min[i],  self.veg.C_i[i]) 
                #self.met.lambda[i]???
                self.veg.D_s[i]         = (self.met.Psurf[i]/0.622) * self.ene.LEc[i] / (self.met.rho[i] * self.met.lhv[i] * (self.veg.g_cw[i]/1000.))
                self.veg.D_s[i]         = np.where(np.isnan(self.veg.D_s[i]),1.15,self.veg.D_s[i]) 
                self.veg.C_s[i]         = np.where(self.veg.C_s[i]<200.0,668.0,self.veg.C_s[i])  
                self.ene.HE[i]          = self.ene.HEc[i] + self.ene.HEs[i]
                self.ene.LE[i]          = self.ene.LEc[i] + self.ene.LEs[i]+self.ene.LEi[i]


                Tck_diff[i] = np.abs(T_C_K[i] - T_C_K_0[i])
                T_C_K_0[i] = T_C_K[i]

            elif  self.nl['ModSet']['Canopy_conductance_model'] == 'Jarvis1976':
                 print('Jarvis1994')
            elif  self.nl['ModSet']['Canopy_conductance_model'] == 'Liang2022':
                true_indices= np.where(i)[0]
                for j in true_indices:
                    #print(j)
                    p2=canopy_conductance_Liang2022(T_C_K[j]-273.15, self.met.RH[j],self.ene.PAR_t[j],self.veg.lmda[j],self.veg.C_s[j],self.met.Psurf[j],self.veg.LAI[j],f5[j])
                    self.veg.rsc[j] =  p2.A_Farquhar_gm()
                    self.veg.rsc[j]= self.veg.rsc[j]*1000.0

                    self.veg.rsc[j] = np.where(self.veg.rsc[j]<0.0, 0.0, self.veg.rsc[j])  
                    self.veg.rsc[j] = np.where(self.veg.rsc[j]>100000.0, 100000.0, self.veg.rsc[j])

                    self.ene.LEc[j],self.ene.LEs[j],self.ene.LEi[j]= self.ETsw(self.veg.Zc[j],self.met.delta[j],self.met.Psy[j],
                                                                     self.met.rho[j],self.ene.A[j],self.met.Cp[j], 
                                                                     self.met.VPD[j],self.ene.As[j],self.met.rac[j], self.met.rss[j], 
                                                                     self.met.ras[j], self.met.raa[j],self.veg.rsc[j],self.met.fwet[j],self.ene.Rnc[j])
                    self.ene.HEs[j]=self.ene.As[j]-self.ene.LEs[j]
                    self.ene.HEc[j]=(self.ene.A[j]-self.ene.As[j])-self.ene.LEc[j]-self.ene.LEi[j]
                    T_C_K[j]    = self.Tc_Taylor_Expansion_theory(T_A_K[j]-273.15,self.met.RH[j],self.veg.rsc[j],self.met.rav[j],self.ene.LEc[j], self.met.Psurf[j], self.met.Cp[j],self.met.Psy[j]) + 273.15

                    self.ene.HE[j]          = self.ene.HEc[j] + self.ene.HEs[j]
                    self.ene.LE[j]          = self.ene.LEc[j] + self.ene.LEs[j]+self.ene.LEi[j]
                    Tck_diff[j] = np.abs(T_C_K[j] - T_C_K_0[j])
                    T_C_K_0[j] = T_C_K[j]

                    '''
                    self.ene.LEc[j],self.ene.LEs[j]= self.ETsw(self.veg.Zc[j],self.met.delta[j],self.met.Psy[j],
                                                                     self.met.rho[j],self.ene.A[j],self.met.Cp[j], 
                                                                     self.met.VPD[j],self.ene.As[j],self.met.rac[j], self.met.rss[j], 
                                                                     self.met.ras[j], self.met.raa[j],self.veg.rsc[j])
                    self.ene.HEs[j]=self.ene.As[j]-self.ene.LEs[j]
                    self.ene.HEc[j]=(self.ene.A[j]-self.ene.As[j])-self.ene.LEc[j]

                    T_C_K[j]    = self.Tc_Taylor_Expansion_theory(T_A_K[j]-273.15,self.met.RH[j],self.veg.rsc[j],self.met.rav[j],self.ene.LEc[j], self.met.Psurf[j], self.met.Cp[j],self.met.Psy[j]) + 273.15
                    self.ene.HE[j]          = self.ene.HEc[j] + self.ene.HEs[j]
                    self.ene.LE[j]          = self.ene.LEc[j] + self.ene.LEs[j]
                    '''
            elif  self.nl['ModSet']['Canopy_conductance_model'] == 'Medlyn':
                print('Medlyn')

        return self.bas, self.met, self.soil, self.veg, self.ene








            