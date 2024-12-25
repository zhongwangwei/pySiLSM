from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from pylab import rcParams
import matplotlib
from mpl_toolkits.basemap import Basemap
import numpy as np


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

def plot_validation_metric(dir_fig, gauge_lon, gauge_lat, metric, cmap, norm, ticks,var,Model,Vdata,obsvarname):

    fig = plt.figure()
    #fig.set_tight_layout(True)
    M = Basemap(projection='robin', resolution='l', lat_0=15, lon_0=0)
    M.drawmapboundary(fill_color='white', zorder=-1)
    M.fillcontinents(color='0.8', lake_color='white', zorder=0)
    M.drawcoastlines(color='0.6', linewidth=0.1)
    M.drawcountries(color='0.6', linewidth=0.1)
    M.drawparallels(np.arange(-60.,60.,30.), dashes=[1,1], linewidth=0.25, color='0.5')
    M.drawmeridians(np.arange(0., 360., 60.), dashes=[1,1], linewidth=0.25, color='0.5')

    loc_lon, loc_lat = M(gauge_lon, gauge_lat)
    cs = M.scatter(loc_lon, loc_lat, 15, metric, cmap=cmap, norm=norm, marker='.', edgecolors='none', alpha=0.9)
   # cbaxes = fig.add_axes([0.26, 0.31, 0.5, 0.015])
    cbaxes = fig.add_axes([0.26, 0.31, 0.5, 0.015])

    cb = fig.colorbar(cs, cax=cbaxes, ticks=ticks, orientation='horizontal', spacing='uniform')
    cb.solids.set_edgecolor("face")
    #cb.set_label('KGE change', position=(0.5, 1.5), labelpad=-35)
    cb.set_label('%s'%(var), position=(0.5, 1.5), labelpad=-35)
    #cb.set_label('log10(RMSE)', position=(0.5, 1.5), labelpad=-35)
    #cb.set_label('PBIAS', position=(0.5, 1.5), labelpad=-35)

    plt.savefig('%s/validation_%s_%s_%s_%s.png' % (dir_fig,Model,Vdata,obsvarname,var),  format='png',dpi=400)
    #plt.show()


if __name__=='__main__':
    argv                    = sys.argv
    Model                   = str(argv[1])
    Vdata                   = str(argv[2])
    obsvarname              = str(argv[3])
    simvarname              = str(argv[4])
    casename                = str(argv[5]) #"03min"    #
    compar_tim_res           = str(argv[6]) 
    Vars  = ['PBIAS','APB','RMSE', 'MAE', 'BIAS', 'NSE', 'L', 'R', 'KGE']
    mins  = [-9998.0,-9998.0,-9998.0,-9998.0,-9998.0,-9998.0,-9998.0,-9998.0,-9998.0]
    maxs  = [10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0,10000.0]
    vmins = [-100.0,-100.0,-100.0,-100.0,-100.0,-1.0,-100.0,-1.0,-1.0]
    vmaxs = [100.0,100.0,100.0,100.0,100.0,1.0,100.0,1.0,1.0]

    data = pd.read_csv(f'cases/{casename}_{compar_tim_res}/metrics/'+f"{casename}_{compar_tim_res}_metrics_{Model}_{Vdata}_{obsvarname}_vs_{simvarname}.csv",header=0)
    for var, minx, maxx,vmin,vmax in zip(Vars, mins, maxs, vmins, vmaxs):
            ind0 = data[data['%s'%(var)]>minx].index
            data_select0 = data.loc[ind0]
            ind1 = data_select0[data_select0['%s'%(var)]<maxx].index
            data_select = data_select0.loc[ind1]

            lon_select = data_select['lon'].values
            lat_select = data_select['lat'].values  
            plotvar=data_select['%s'%(var)].values
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            plot_validation_metric(f'cases/{casename}_{compar_tim_res}/metrics/', lon_select, lat_select, plotvar, cmap, norm, bnd,var,Model,Vdata,obsvarname)
