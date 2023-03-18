import numpy as np
from scipy.optimize import curve_fit

import pds4_tools as pds
from datetime import datetime, timedelta

aspect_ratio = 16./9 
BoiseState_blue = "#0033A0"
BoiseState_orange = "#D64309"

str2date = lambda x: datetime.strptime(x, '00133M%H:%M:%S.%f')
str2dates = lambda xs: [str2date(xs[i]) for i in range(len(xs))]

zs = np.array([75., 150., 300., 600., 1200.])
sampling_duration = 10. # seconds to sample an altitude

def calc_zstar_from_slope_and_intercept(z0, slope, intercept):
    return z0*np.exp(-intercept/slope)

def calc_ustar_from_slope(slope, kappa=0.4):
    return kappa*slope

def calculate_zstar_from_profile(heights, winds):
    x = np.log(heights/np.min(heights))
    y = winds

    popt, pcov = curve_fit(lin_fit, x, y)

    zstar = calc_zstar_from_slope_and_intercept(np.min(heights), *popt)
    return zstar

def lin_fit(x, m, b):
    return m*x + b

def wind_profile(z, ustar, zstar, kappa=0.4):
    return ustar/kappa*np.log(z/zstar)

def fit_wind_profile(z, ustar, zstar, kappa=0.4):
    x = np.log(z/np.min(z))
    
    # slope
    m = ustar/kappa
    # intercept
    b = -ustar/kappa*np.log(zstar/np.min(z))
    
    return lin_fit(x, m, b)

def calc_tilt(pitch, roll):
    # https://math.stackexchange.com/questions/2563622/vertical-inclination-from-pitch-and-roll
    return np.degrees(np.arctan(np.sqrt(np.tan(np.radians(roll))**2 +\
                                        np.tan(np.radians(pitch))**2)))

def chisqg(ydata,ymod,sd=None):
    """
    Returns the chi-square error statistic as the sum of squared errors between
    Ydata(i) and Ymodel(i). If individual standard deviations (array sd) are supplied,
    then the chi-square error statistic is computed as the sum of squared errors
    divided by the standard deviations.     Inspired on the IDL procedure linfit.pro.
    See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

    x,y,sd assumed to be Numpy arrays. a,b scalars.
    Returns the float chisq with the chi-square statistic.

    Rodrigo Nemmen
    http://goo.gl/8S1Oo
    """
    # Chi-square statistic (Bevington, eq. 6.9)
    if np.all(sd==None):
        chisq=np.sum((ydata-ymod)**2)
    else:
        chisq=np.sum( ((ydata-ymod)/sd)**2 )

    return chisq

def redchisqg(ydata,ymod,deg=2,sd=None):
    """
    Returns the reduced chi-square error statistic for an arbitrary model,
    chisq/nu, where nu is the number of degrees of freedom. If individual
    standard deviations (array sd) are supplied, then the chi-square error
    statistic is computed as the sum of squared errors divided by the standard
    deviations. See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.
    
    ydata,ymod,sd assumed to be Numpy arrays. deg integer.
      
      Usage:
          >>> chisq=redchisqg(ydata,ymod,n,sd)
          where
          ydata : data
          ymod : model evaluated at the same x points as ydata
          n : number of free parameters in the model
          sd : uncertainties in ydata
          
          Rodrigo Nemmen
          http://goo.gl/8S1Oo
    """
    # Chi-square statistic
    if np.all(sd==None):
        chisq=np.sum((ydata-ymod)**2)
    else:
        chisq=np.sum( ((ydata-ymod)/sd)**2 )

    # Number of degrees of freedom
    nu=ydata.size-1-deg

    return chisq/nu

def calc_analytic_sigma_intercept(sigma, N):
    return np.sqrt(2.*sigma**2*(2*N - 1)/N/(N + 1))

def calc_analytic_sigma_slope(delta_x, sigma, N):
    return np.sqrt(12*sigma**2/delta_x**2/N/(N**2 - 1))

def calc_sigma_intercept_slope(delta_x, sigma, N):
    return 6*sigma**2/N/(N + 1)/delta_x

def calc_analytic_sigma_ustar(delta_x, sigma, N, kappa=0.4):
    return kappa*calc_analytic_sigma_slope(delta_x, sigma, N)

def calc_analytic_sigma_zstar(z0, slope, intercept, delta_x, sigma, N, kappa=0.4):
    zstar = calc_zstar_from_slope_and_intercept(z0, slope, intercept)
    
    sigma_intercept = calc_analytic_sigma_intercept(sigma, N)
    sigma_slope = calc_analytic_sigma_slope(delta_x, sigma, N)
    sigma_intercept_slope = calc_sigma_intercept_slope(delta_x, sigma, N)
    
    return zstar*intercept/slope*np.sqrt((sigma_slope/slope)**2 +\
            (sigma_intercept/intercept)**2 +\
            2*(sigma_intercept_slope/intercept/slope))

def rescale_sigma(data, mod, sigma, dof=2):
    #_NR_, 3rd ed, p. 783 - This equation provides a way to rescale
    # uncertainties, enforcing reduced chi-squared = 1

    chisq = chisqg(data, mod, sd=sigma)

    return sigma*np.sqrt(chisq/(len(data) - dof))

def calc_S(sigma):
    return np.sum(1./sigma**2)

def calc_Sx(x, sigma):
    return np.sum(x/sigma**2)

def calc_Sxx(x, sigma):
    return np.sum(x**2/sigma**2)

def calc_Sy(y, sigma):
    return np.sum(y/sigma**2)

def calc_Syy(y, sigma):
    return np.sum(y**2/sigma**2)

def calc_Sxy(x, y, sigma):
    return np.sum(x*y/sigma**2)

def calc_Delta(x, sigma):
    S = calc_S(sigma)
    Sxx = calc_Sxx(x, sigma)
    Sx = calc_Sx(x, sigma)
    
    return S*Sxx - Sx**2

def calc_intercept(x, y, sigma):
    Sxx = calc_Sxx(x, sigma)
    Sy = calc_Sy(y, sigma)
    Sx = calc_Sx(x, sigma)
    Sxy = calc_Sxy(x, y, sigma)
    Delta = calc_Delta(x, sigma)
    
    return (Sxx*Sy - Sx*Sxy)/Delta

def calc_slope(x, y, sigma):
    S = calc_S(sigma)
    Sxy = calc_Sxy(x, y, sigma)
    Sx = calc_Sx(x, sigma)
    Sy = calc_Sy(y, sigma)
    Delta = calc_Delta(x, sigma)
    
    return (S*Sxy - Sx*Sy)/Delta

def calc_cov(x, sigma):
    return -calc_Sx(x, sigma)/calc_Delta(x, sigma)

def sigma_intercept(x, sigma):
    Sxx = calc_Sxx(x, sigma)
    Delta = calc_Delta(x, sigma)
    
    return np.sqrt(Sxx/Delta)

def sigma_slope(x, sigma):
    S = calc_S(sigma)
    Delta = calc_Delta(x, sigma)
    
    return np.sqrt(S/Delta)

# For the analytic solutions, some functions want scalar sigma, 
# others want vector sigma.
def calc_analytic_intercept(delta_x, sigma, N, x, y):
    Sxx = calc_analytic_Sxx(delta_x, sigma[0], N)
    Sy = calc_Sy(y, sigma)
    Sx = calc_analytic_Sx(delta_x, sigma[0], N)
    Sxy = calc_Sxy(x, y, sigma)
    Delta = calc_analytic_Delta(delta_x, sigma[0], N)

    return (Sxx*Sy - Sx*Sxy)/Delta

def calc_analytic_slope(delta_x, sigma, N, x, y):
    S = calc_analytic_S(sigma[0], N)
    Sxy = calc_Sxy(x, y, sigma)
    Sx = calc_analytic_Sx(delta_x, sigma[0], N)
    Sy = calc_Sy(y, sigma)
    Delta = calc_analytic_Delta(delta_x, sigma[0], N)

    return (S*Sxy - Sx*Sy)/Delta

def calc_analytic_S(sigma, N):
    return N/sigma**2

def calc_analytic_Delta(delta_x, sigma, N):
    return delta_x**2/12/sigma**4*N**2*(N**2 - 1)

def calc_analytic_S(sigma, N):
    return N/sigma**2

def calc_analytic_Sx(delta_x, sigma, N):
    return 0.5*delta_x/sigma**2*(N - 1)*N

def calc_analytic_Sxx(delta_x, sigma, N):
    return delta_x**2/sigma**2*(N - 1)*N*(2*N - 1)/6

def calc_analytic_fractional_zstar_uncertainty(N, kappa=0.4, z0_over_zstar=25., sigma_over_ustar=1.,
                                               delta_x=np.log(2)):
    term1 = np.log(z0_over_zstar)**2*12/N/(N**2 - 1)/delta_x**2
    term2 = 2*(2*N - 1)/N/(N + 1)
    term3 = 2*np.log(z0_over_zstar)*6/N/(N + 1)/delta_x
    analytic_fraction_sigma_zstar = kappa*sigma_over_ustar*np.sqrt(term1 + term2 + term3)

    return analytic_fraction_sigma_zstar

def calc_analytic_fractional_ustar_uncertainty(N, sigma_over_ustar=1., delta_x=np.log(2), kappa=0.4):

    return kappa*np.sqrt(12./(N*(N**2 - 1)*delta_x**2))*sigma_over_ustar

def calc_sigma_zstar(z0, slope, intercept, sigma_slope, sigma_intercept, kappa=0.4):
    zstar = calc_zstar_from_slope_and_intercept(z0, slope, intercept)
    return zstar*intercept/slope*np.sqrt((sigma_slope/slope)**2 + (sigma_intercept/intercept)**2)

def calc_sigma_ustar(sigma_slope, kappa=0.4):
    return kappa*sigma_slope

def retrieve_time_wind(xml_file,
    start_time=datetime(1900, 1, 1, 7, 0, 0),
    end_time=datetime(1900, 1, 1, 8, 0, 0)):

    struct_list = pds.read(xml_file)

    time = np.array(str2dates(struct_list['TABLE']['LMST']))
    wind = np.array(struct_list['TABLE']['HORIZONTAL_WIND_SPEED'])

    if((start_time is None) or (end_time is None)):
        ind = wind != 999999999
        time = time[ind]
        wind = wind[ind]

    else:
        ind = (wind != 999999999) & (time > start_time) & (time < end_time)

        time = time[ind]
        wind = wind[ind]

    return time, wind

def calculate_scaled_windspeed(windspeeds_to_scale, z, zstar, z0=150.):
    """
    Returns wind speed time series scaled as if it were measured at a different elevation,
    assuming a logarithmic wind profile

    Args:
        windspeeds_to_scale (float array): wind speeds to scale
        z (float): altitude at which measurement is assumed to take place
        zstar (float): the roughness scale - https://en.wikipedia.org/wiki/Roughness_length
        z0 (float, optional): altitude at which actual wind speed measurement was made; defaults to 1.5 meters
        kappa (float, optional): von Karman parameter; defaults to 0.4

    Returns:
        Scaled wind speed time series

    """

    # Calculate ustar assuming average windspeeds_to_scale
    return windspeeds_to_scale*np.log(z/zstar)/np.log(z0/zstar)

def create_synthetic_wind_profile(windspeeds_to_scale, zs, zstar, z0=150.):

    scaled_windspeeds = calculate_scaled_windspeed(windspeeds_to_scale, zs[0], 
        zstar, z0=z0)
    for i in range(1, len(zs)):    
        scaled_windspeeds = np.vstack([scaled_windspeeds,
            calculate_scaled_windspeed(windspeeds_to_scale, zs[i], zstar, 
            z0=z0)])

    return scaled_windspeeds

def retrieve_relevant_times(time, t0, sample_time):
    return (t0 <= time) & (time <= t0 + sample_time)

def sample_wind_profile(sample_time, t0, time, windspeeds, heights):
    """
    Return wind speeds from four height sampled over the given sample time

    Args:
        sample_time (float): time over which to average in seconds
        t0 (float, optional): time at which at start averages
        times (float array): measured times
        windspeeds (float array): wind speed time-series referenced by anemometer height

    Returns:
        Wind speeds averaged for sample time from different anemometer times series, one after another
    """

    # Run through each height, assuming the first one in windspeeds is the 
    # lowest and on up
    cur_t0 = t0
    averaged_windspeeds = np.zeros_like(heights)
    std_windspeeds = np.zeros_like(heights)
    for i in range(len(heights)):
        ind = retrieve_relevant_times(time, cur_t0, sample_time)

        averaged_windspeeds[i] = np.mean(windspeeds[i][ind])

        # error of the mean
        std_windspeeds[i] = np.std(windspeeds[i][ind])/\
            (np.sqrt(len(windspeeds[i][ind]) - 1.))

        cur_t0 += sample_time

    return averaged_windspeeds, std_windspeeds
