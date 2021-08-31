import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import scipy as sc
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

## check lon
def check_lon(dqa):
    if np.min(dqa.lon.values)<0.0:
        dqa.lon.values[dqa.lon.values<0.0]=360.0+dqa.lon.values[dqa.lon.values<0.0]
        dqa = dqa.reindex(lon=np.sort(dqa.lon.values))
    return dqa 

## check level
def check_level(dua):
    """
    check if latitude are in rverese order.
    Then select the specified level.
    """
    
    dy       = np.gradient(dua.level) 
    if np.sum(dy)>0:
        dua  = dua.reindex(level=dua.level[::-1])

    return dua

## check lat
def check_lat(uname,uvarname):
    """
    check if latitude are in rverese order.
    Then select the specified level.
    """
    
    dua1             =     xr.open_dataset(uname)
    dua              =     dua1[uvarname].rename({dua1[uvarname].dims[1]: "level"})
    print(dua)
    dy               = np.gradient(dua.lat) 
    
    if np.unique(dy)[0]<0:
        dua  = dua.reindex(lat=dua.lat[::-1])
        

    return dua1.sel(lat=slice(-30,30))

def create_mesh(u,grid):
    """
    create a mesh grid based on the data dimension
    """
    if len(u.shape) == 4:
        time            =   np.arange(u.shape[0])
        _,_,latv, lonv  =   np.meshgrid(time,grid[0],grid[1], grid[2], indexing='ij')
    if len(u.shape) == 3:
        time            =   np.arange(u.shape[0])
        _,latv, lonv    =   np.meshgrid(time,grid[0],grid[1], indexing='ij')
    elif len(u.shape)== 2:
        latv, lonv      =   np.meshgrid(grid[0],grid[1], indexing='ij')

    return latv,lonv

def hmfc(q,u,v,grid):
    dtr = np.pi/180        ### degree to radian
    r   = 6.371*(10**6)

    latv,lonv        =   create_mesh(u,grid)
    dQX              =   np.gradient(q,axis=-1)
    dQY              =   np.gradient(q*np.cos(latv*dtr),axis=-2)
    dUX              =   np.gradient(u,axis=-1)
    dVY              =   np.gradient(v*np.cos(latv*dtr),axis=-2)
    dX               =   np.gradient(lonv*dtr,axis=-1)
    dY               =   np.gradient(latv*dtr,axis=-2) 
    madvection       =   (u*dQX/dX + v*dQY/dY)/(r*np.cos(latv*dtr))
    convergence      =   (dUX/dX  + dVY/dY)/(r*np.cos(latv*dtr))
    mconvergence     =   q* convergence

    MFC              =   (madvection+mconvergence)
    return MFC, madvection, mconvergence,convergence

def vert_integration_pres(field):
    mb_to_pa = 100
    g = 9.81
    integrated = -(1/g)*field.integrate("level")*mb_to_pa
    return integrated

def mfc_yearwise_nc(i,dqa,dua,dva,qvarname,uvarname,vvarname):
    dua1 = dua.sel(time =(dua.time.dt.year==int(i)))
    dva1 = dva.sel(time =(dva.time.dt.year==int(i)))
    dqa1 = dqa.sel(time =(dqa.time.dt.year==int(i)))
#     print(dua1)
    time1 = dua1.time;level1= dua1.level;lat1= dua1.lat;lon1 =dua1.lon

    q      =  dqa1.values
    u      =  dua1.values
    v      =  dva1.values
    
    if u.shape!=v.shape:
        raise ValueError("Dimension_mismatch between U and V")
        
    grid   =  [dua1.level.values,dua1.lat.values,dua1.lon.values]
    MFC, madvection, mconvergence,convergence   =  hmfc(q,u,v,grid)
    mfc_ds = xr.Dataset({'mfc': (('time','level', 'lat','lon'), MFC)}, coords={'time': time1,'level': level1,'lat': lat1,'lon': lon1})
    madv_ds = xr.Dataset({'madv': (('time','level', 'lat','lon'), madvection)}, coords={'time': time1,'level': level1,'lat': lat1,'lon': lon1})
    mconv_ds = xr.Dataset({'mconv': (('time','level', 'lat','lon'), mconvergence)}, coords={'time': time1,'level': level1,'lat': lat1,'lon': lon1})
    
    
    mfc_ds_vint      = vert_integration_pres(mfc_ds)
    mconv_ds_vint    = vert_integration_pres(mconv_ds)
    madv_ds_vint     = vert_integration_pres(madv_ds)
    
    mfc_ds_vint.to_netcdf('mfc_ds_vint_'+str(i)+'.nc')
    madv_ds_vint.to_netcdf('madv_ds_vint_'+str(i)+'.nc')
    mconv_ds_vint.to_netcdf('mconv_ds_vint_'+str(i)+'.nc')
    
    
def mfc_comp_yr(i,dqa,dua,dva,qvarname,uvarname,vvarname):
    
    dua1 = dua.sel(time =(dua.time.dt.year==int(i)))
    dva1 = dva.sel(time =(dva.time.dt.year==int(i)))
    dqa1 = dqa.sel(time =(dqa.time.dt.year==int(i)))
#     print(dua1)
    time1 = dua1.time;level1= dua1.level;lat1= dua1.lat;lon1 =dua1.lon

    q      =  dqa1.values
    u      =  dua1.values
    v      =  dva1.values
    
    if u.shape!=v.shape:
        raise ValueError("Dimension_mismatch between U and V")
        
    grid                                        =  [dua1.level.values,dua1.lat.values,dua1.lon.values]
    MFC, madvection, mconvergence,convergence   =  hmfc(q,u,v,grid)
    
    mconv_ds = xr.Dataset({'mconv': (('time','level', 'lat','lon'), mconvergence)}, coords={'time': time1,'level': level1,'lat': lat1,'lon': lon1})
    conv_ds = xr.Dataset({'conv': (('time','level', 'lat','lon'), convergence)}, coords={'time': time1,'level': level1,'lat': lat1,'lon': lon1})
    

    
    
    mconv_ds.to_netcdf('mconv1_ds_'+str(i)+'.nc')
    conv_ds.to_netcdf('conv_ds_'+str(i)+'.nc')

