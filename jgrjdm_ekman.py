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

def curl(u,v,grid):
    dtr = np.pi/180        ### degree to radian
    r   = 6.371*(10**6)

    latv,lonv       =   create_mesh(u,grid)
    dU              =   np.gradient(u*np.cos(latv*dtr),axis=-2)
    dV              =   np.gradient(v,axis=-1)
    dX              =   np.gradient(lonv*dtr,axis=-1)
    dY              =   np.gradient(latv*dtr,axis=-2) 
    curl             =  (dV/dX-dU/dY)/(r*np.cos(latv*dtr))
    
    return curl 
    
def altimpy_curl(u10,v10):
    '''
    Input: u10 and v10 with dimension (time X lat X lon)
    Outpur: Curl (np.array)
    link: https://github.com/fspaolo/altimpy/blob/master/build/lib/altimpy/util.py
    '''
    grid             =  [u10.lat.values,u10.lon.values]      
    latv,lonv        =   create_mesh(u10,grid)
    
    dy = np.abs(latv[0,1,0] - latv[0,0,0]) # dim =1 LAT
    dx = np.abs(lonv[0,0,1] - lonv[0,0,0]) # dim =2 LON
    
    dy *= 110575. # scalar in m
    dx *= 111303. * np.cos(latv * np.pi/180) 
    
    dudy = np.gradient(u10,axis=1)/dy
    dvdx = np.gradient(v10,axis=2)/dx
    
    wc1  = dvdx - dudy
    
    return wc1

def wind_stress(u10_1, v10_1, rho_air=1.225, Cd_const=True):
    """"
     Parameters:
        u10 (numpy.ndarray): Wind speed in the x-direction (10 m above the surface).
        v10 (numpy.ndarray, optional): Wind speed in the y-direction (10 m above the surface).
        Cd (float or numpy.ndarray, optional): Drag coefficient. Default is 1.25e-3.
        rho_air (float or numpy.ndarray, optional): Air density. Default is 1.225 kg/m^3.
    
    Returns:
        Tuple containing:
            - Tau or (Taux, Tauy): Wind stress in Pascals or zonal and meridional components.
    Notes:
        Function to compute wind stress from wind field data is based on Gill,
        (1982)[1]. Formula and a non-linear drag coefficient (cd) based on
        Large and Pond (1981)[2], modified for low wind speeds (Trenberth et
        al., 1990)[3]

        [1] A.E. Gill, 1982, Atmosphere-Ocean Dynamics, Academy Press.
        [2] W.G. Large & S. Pond., 1981,Open Ocean Measurements in Moderate
        to Strong Winds, J. Physical Oceanography, v11, p324-336.
        [3] K.E. Trenberth, W.G. Large & J.G. Olson, 1990, The Mean Annual
        Cycle in Global Ocean Wind Stress, J. Physical Oceanography, v20,
        p1742-1760.
    Fernando Paolo <fpaolo@ucsd.edu>
    Mar 7, 2016

    Estimates wind stress on the ocean from wind speed.
    """
    u10 = u10_1.copy()
    v10 = v10_1.copy()
    
    w = np.sqrt(u10**2 + v10**2) # wind speed (m/s) 
    if Cd_const:
        Cd = 1.25e-3
    else:
        # wind-dependent drag
        cond1 = (w<=1)
        cond2 = (w>1) & (w<=3)
        cond3 = (w>3) & (w<10)
        cond4 = (w>=10)
        Cd = np.zeros_like(w)
        Cd[cond1] = 2.18e-3 
        Cd[cond2] = (0.62 + 1.56/w[cond2]) * 1e-3
        Cd[cond3] = 1.14e-3
        Cd[cond4] = (0.49 + 0.065*w[cond4]) * 1e-3

    U   = np.hypot(u10, v10)
    Tau = rho_air * Cd * (u10**2 + v10**2)
    
    Taux = Tau * u10/ U
    Tauy = Tau * v10/ U
    return Tau,Taux, Tauy

def ekman(u10_1, v10_1, rho_water= 1025):
    """
    Estimates the classical Ekman transport and upwelling/downwelling from 10 m winds.
    
    Parameters:
        u10 (numpy.ndarray): Zonal wind speed at 10 m (2D or 3D array).
        v10 (numpy.ndarray): Meridional wind speed at 10 m (2D or 3D array).
        *args: Optional arguments for 'Cd', 'ci', and 'rho'.
        **kwargs: Optional keyword arguments for 'Cd', 'ci', and 'rho'.
    
    Returns:
        Tuple containing:
            - UE: Zonal Ekman transport (m^2/s).
            - VE: Meridional Ekman transport (m^2/s).
            - wE: Vertical velocity (m/s) from Ekman pumping.
            - dE:  Ekman layer depth.
            - WSC: Wind stress curl.
    """
    
    dtr             =   np.pi/180        ### degree to radian
    r               =   6.371*(10**6)    ### Radius of Earth
    omega           =   7.292115e-5 # rotation rate of the Earth (rad/s)

    
    u10 = u10_1.copy()
    v10 = v10_1.copy()
    
    ### Calculate total and components of windstress from U10 and V10 ####
    Tau,Taux,Tauy   =   wind_stress(u10, v10)
    
    ### Calculate f based on the lon lat grid #############
    grid            =   [u10.lat.values,u10.lon.values]      
    latv,lonv       =   create_mesh(u10,grid) 
    f               =   2 * omega * np.sin(latv * np.pi/180) # (rad/s)
    beta            =   2 * omega * np.cos(latv * np.pi/180)/ r
    
    ### calculate the dX and dY based in the grid spacing ######
    dx              =   np.gradient(lonv*dtr,axis=2)*(r*np.cos(latv*dtr))
    dy              =   np.gradient(latv*dtr,axis=1)*(r) 

    ### Calculate the Ekman Layer depth from Eqn 9.16 of Introduction to Physical Oceanography, Robert H. Stewart, September 2008 ####
    dE              =   7.6 * np.hypot(u10, v10) / np.sqrt(np.sin(np.abs(latv) * np.pi/180))

    
    ############# Calculate Ekman transports ########
    UE =  Tauy / (rho_water * f)  ### Zonal Ekman Transport
    VE = -Taux / (rho_water * f)  ### Meridional Ekman Transport
    

    #####    Calculate Ekman Pumping ###################
    dUdx           = np.gradient(UE, axis=2)/dx
    dVdy           = np.gradient(VE, axis=1)/dy
    
    wE            = dUdx + dVdy

    ########## Calculate Windstress curl ###########
    
    TX = u10.copy();TX.values =  Taux.copy()
    TY = v10.copy();TY.values =  Tauy.copy()
    WSC =  mpcalc.vorticity(TX,TY)

    wE1 = WSC/(rho_water * f)
    wE2 = (beta*TX)/(rho_water * f**2)

    wE = xr.Dataset({'wE': (('time','lat','lon'), wE)}, coords={'time': u10.time,'lat': u10.lat,'lon': u10.lon})
    wE1 = xr.Dataset({'wE1': (('time','lat','lon'), wE1.values)}, coords={'time': u10.time,'lat': u10.lat,'lon': u10.lon})
    wE2 = xr.Dataset({'wE2': (('time','lat','lon'), wE2.values)}, coords={'time': u10.time,'lat': u10.lat,'lon': u10.lon})

    WSC = xr.Dataset({'WSC': (('time','lat','lon'), WSC.values)}, coords={'time': u10.time,'lat': u10.lat,'lon': u10.lon})
   
    return UE,VE,wE,wE1,wE2,dE,WSC
