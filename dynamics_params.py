import xarray as xr
import numpy as np
from scipy import integrate

def check_lat(uname,uvarname):
    """
    check if latitude are in rverese order.
    Then select the specified level.
    """
    
    dua      = xr.open_dataset(uname)
    dy       = np.gradient(dua.lat) 
    if np.unique(dy)[0]<0:
        dua  = dua.reindex(lat=dua.lat[::-1])

    return dua



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


def hdivergence(u,v,grid):
    dtr = np.pi/180        ### degree to radian
    r   = 6.371*(10**6)

    latv,lonv       =   create_mesh(u,grid)
    dV              =   np.gradient(v*np.cos(latv*dtr),axis=-2)
    dU              =   np.gradient(u,axis=-1)
    dX              =   np.gradient(lonv*dtr,axis=-1)
    dY              =   np.gradient(latv*dtr,axis=-2) 
    DIV             =  (dU/dX + dV/dY)/(r*np.cos(latv*dtr))
    
    return DIV  


def hvorticity(u,v,grid):
    dtr = np.pi/180        ### degree to radian
    r   = 6.371*(10**6)  
    latv,lonv       =   create_mesh(u,grid)
    dV              =   np.gradient(v,axis=-1)
    dU              =   np.gradient(u*np.cos(latv*dtr),axis=-2)
    dX              =   np.gradient(lonv*dtr,axis=-1)
    dY              =   np.gradient(latv*dtr,axis=-2)               
    VOR           =     (dV/dX - dU/dY)/(r*np.cos(latv*dtr))
    return VOR

def hadvection(T,u,v,grid):
    dtr = np.pi/180        ### degree to radian
    r   = 6.371*(10**6)

    latv,lonv       =   create_mesh(u,grid)
    dTX             =   np.gradient(T,axis=-1)
    dTY             =   np.gradient(T*np.cos(latv*dtr),axis=-2)
    dX              =   np.gradient(lonv*dtr,axis=-1)
    dY              =   np.gradient(latv*dtr,axis=-2) 
    ADV             =   (u*dTX/dX+v*dTY/dY)/(r*np.cos(latv*dtr))
    return ADV

def hmfc(q,u,v,grid):
    dtr = np.pi/180        ### degree to radian
    r   = 6.371*(10**6)

    latv,lonv       =   create_mesh(u,grid)
    dQX             =   np.gradient(q,axis=-1)
    dQY             =   np.gradient(q*np.cos(latv*dtr),axis=-2)
    dUX             =   np.gradient(U,axis=-1)
    dVY             =   np.gradient(V*np.cos(latv*dtr),axis=-2)
    dX              =   np.gradient(lonv*dtr,axis=-1)
    dY              =   np.gradient(latv*dtr,axis=-2) 
    madvection       =   u*dQX/dX + v*dQY/dY
    mconvergence     =   q(dU/dX  + dV/dY)
    MFC             =   (madvection+mconvergence)/(r*np.cos(latv*dtr))
    return MFC

def hdivergence_xarray(uname,vname,sellevel='all',uvarname='uwnd',vvarname='vwnd'):
    """
    input:
    uname    : u filename
    uvarname : u variable name ;'uwnd'
    vname    : v filename
    vvarname : v variable name ;'vwnd'
    dimension should be in (time,level,lon,lat)
    output:
    divergence  = (du/dx + dv/dy)    s-1
    """
    
    dua = check_lat(uname,uvarname)
    dva = check_lat(vname,vvarname)
    
    if sellevel=='all':
        
        u      =  dua[uvarname].values
        v      =  dva[vvarname].values
        grid  =  [dua.level.values,dua.lat.values,dua.lon.values]
        
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
        return
       
        DIV = hdivergence(u,v,grid)

        div_ds = xr.Dataset({'DIV': (('time','level','lat','lon'), DIV)}, coords={'time': dua.time,'level':dua.level,'lat': dua.lat,'lon': dua.lon})

    elif sellevel=='None':
        
        u = dua[uvarname].values
        v = dva[vvarname].values
        grid  =  [dua.lat.values,dua.lon.values]

        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return

        DIV = hdivergence(u,v,grid)

        div_ds = xr.Dataset({'DIV': (('time', 'lat','lon'), DIV)}, coords={'time': dua.time,'lat': dua.lat,'lon': dua.lon})
    
    else:
        u = dua.sel(level=sellevel)[uvarname].values
        v = dva.sel(level=sellevel)[vvarname].values
        grid  =  [dua.lat.values,dua.lon.values]
        
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
       
        DIV = hdivergence(u,v,grid)

        div_ds = xr.Dataset({'DIV': (('time', 'lat','lon'), DIV)}, coords={'time': dua.time,'lat': dua.lat,'lon': dua.lon})
#     div_ds.to_netcdf(output_name)
      
    return div_ds



def hvorticity_xarray(uname,vname,sellevel='all',uvarname='uwnd',vvarname='vwnd'):
    """
    input:
    uname    : u filename
    uvarname : u variable name ;'uwnd'
    vname    : v filename
    vvarname : v variable name ;'vwnd'
    dimension should be in (time,level,lon,lat)
    output:
    ds_vort : vorticity =    (dv/dx-du/dy)     s-1
    """ 
    
    dua = check_lat(uname,uvarname)
    dva = check_lat(vname,vvarname)
  
    if sellevel=='all':
        
        u      =  dua[uvarname].values
        v      =  dva[vvarname].values
        grid  =  [dua.level.values,dua.lat.values,dua.lon.values]
        
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
       
        VOR = hvorticity(u,v,grid)

        vor_ds = xr.Dataset({'VOR': (('time','level','lat','lon'), VOR)}, coords={'time': dua.time,'level':dua.level,'lat': dua.lat,'lon': dua.lon})

    elif sellevel == 'None':
        
        u = dua[uvarname].values
        v = dva[vvarname].values
        grid  =  [dua.lat.values,dua.lon.values]
        
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
        print(u.shape)

        VOR = hvorticity(u,v,grid)

        vor_ds = xr.Dataset({'VOR': (('time', 'lat','lon'), VOR)}, coords={'time': dua.time,'lat': dua.lat,'lon': dua.lon})
    
    else:
        
        u = dua.sel(level=sellevel)[uvarname].values
        v = dva.sel(level=sellevel)[vvarname].values
        grid  =  [dua.lat.values,dua.lon.values]
        
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
        print(u.shape)

        VOR = hvorticity(u,v,grid)

        vor_ds = xr.Dataset({'VOR': (('time', 'lat','lon'), VOR)}, coords={'time': dua.time,'lat': dua.lat,'lon': dua.lon})
    
    
    return vor_ds


def hadvection_xarray(uname,vname,tname,sellevel='all',tvarname='temperature',uvarname='uwnd',vvarname='vwnd'):
    
    """
    input:
    uname    : u filename
    uvarname : u variable name ;'uwnd'
    vname    : v filename
    vvarname : v variable name ;'vwnd'
    tname    : scalar variable name
    tvarname : scalar variable name
    dimension should be in (time,level,lon,lat)
    output:
    ds_vort : advection =    u*dT/dx+v*dT/dy     s-1
    """ 
    dua = check_lat(uname,uvarname)
    dva = check_lat(vname,vvarname)
    dta = check_lat(tname,tvarname)
   
    if sellevel=='all':
        T      =  dta[tvarname].values
        u      =  dua[uvarname].values
        v      =  dva[vvarname].values
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
        grid   =  [dua.level.values,dua.lat.values,dua.lon.values]
        ADV    =  hadvection(T,u,v,grid)
        adv_ds = xr.Dataset({'ADV': (('time','level', 'lat','lon'), ADV)}, coords={'time': dua.time,'level': dua.level,'lat': dua.lat,'lon': dua.lon})
    
    
    elif sellevel=='none':
        
        T = dta[tvarname].values
        u = dua[uvarname].values
        v = dva[vvarname].values
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
        grid   =  [dua.lat.values,dua.lon.values]
        ADV    =  hadvection(T,u,v,grid)
        adv_ds = xr.Dataset({'ADV': (('time', 'lat','lon'), ADV)}, coords={'time': dua.time,'lat': dua.lat,'lon': dua.lon})
        
    
    else:
        T = dta.sel(level=sellevel)[tvarname].values
        u = dua.sel(level=sellevel)[uvarname].values
        v = dva.sel(level=sellevel)[vvarname].values
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
        grid   =  [dua.lat.values,dua.lon.values]
        ADV    =  hadvection(T,u,v,grid)
        adv_ds = xr.Dataset({'ADV': (('time', 'lat','lon'), ADV)}, coords={'time': dua.time,'lat': dua.lat,'lon': dua.lon})
    
    return adv_ds

def hmfc_xarray(uname,vname,qname,sellevel='all',qvarname='shum',uvarname='uwnd',vvarname='vwnd'):
    
    """
    input:
    uname    : u filename
    uvarname : u variable name ;'uwnd'
    vname    : v filename
    vvarname : v variable name ;'vwnd'
    qname    : shum.nc
    qvarname : shum
    dimension should be in (time,level,lon,lat)
    output:
    ds_vort : MFC =    -((udq/dx+vdq/dy) +q(du/dx+dv/dy))  
    """ 
    dua = check_lat(uname,uvarname)
    dva = check_lat(vname,vvarname)
    dqa = check_lat(qname,qvarname)
   
    if sellevel=='all':
        q      =  dqa[qvarname].values
        u      =  dua[uvarname].values
        v      =  dva[vvarname].values
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
        grid   =  [dua.level.values,dua.lat.values,dua.lon.values]
        MFC    =  hmfc(q,u,v,grid)
        mfc_ds = xr.Dataset({'MFC': (('time','level', 'lat','lon'), MFC)}, coords={'time': dua.time,'level': dua.level,'lat': dua.lat,'lon': dua.lon})
    
    elif sellevel=='none':
        T = dta[tvarname].values
        u = dua[uvarname].values
        v = dva[vvarname].values
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
        grid   =  [dua.lat.values,dua.lon.values]
        MFC    =  hmfc(q,u,v,grid)
        mfc_ds = xr.Dataset({'MFC': (('time', 'lat','lon'), MFC)}, coords={'time': dua.time,'lat': dua.lat,'lon': dua.lon})    
        
    else:
        T = dta.sel(level=sellevel)[tvarname].values
        u = dua.sel(level=sellevel)[uvarname].values
        v = dva.sel(level=sellevel)[vvarname].values
        if u.shape!=v.shape:
            raise ValueError("Dimension_mismatch between U and V")
            return
        grid   =  [dua.lat.values,dua.lon.values]
        MFC    =  hmfc(q,u,v,grid)
        mfc_ds = xr.Dataset({'MFC': (('time', 'lat','lon'), MFC)}, coords={'time': dua.time,'lat': dua.lat,'lon': dua.lon})
    
    return mfc_ds




def vert_integration_pres(field):
    mb_to_pa = 100
    g = 9.81
    integrated = -(1/g)*field.integrate("level")*mb_to_pa
    return integrated
    