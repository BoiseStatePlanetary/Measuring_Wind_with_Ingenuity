import numpy as np
from scipy.optimize import curve_fit

import pds4_tools as pds
from datetime import datetime, timedelta
import matplotlib.dates as md

aspect_ratio = 16./9 
BoiseState_blue = "#0033A0"
BoiseState_orange = "#D64309"

str2date = lambda x: datetime.strptime(x, '%H:%M:%S.%f')
str2dates = lambda xs: [str2date(xs[i]) for i in range(len(xs))]

zs = np.array([75., 150., 300., 600., 1200.])
sampling_duration = 10. # seconds to sample an altitude
sample_time = timedelta(seconds=sampling_duration) # as a time delta

start_time = datetime(1900, 1, 1, 7, 0, 0)
end_time = datetime(1900, 1, 1, 8, 0, 0)

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

def retrieve_time_wind(xml_file, start_time=start_time, end_time=end_time):

    struct_list = pds.read(xml_file)
    time_list = list(map(lambda x: x.split('M')[1],
        struct_list['TABLE']['LMST']))
    time = np.array(str2dates(time_list))
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

def make_plot_of_original_and_scaled_windspeeds(time, wind, zs, sample_time, t0,
    scaled_windspeeds, averaged_windspeeds, ax):

    from itertools import cycle
    colors = cycle([BoiseState_blue, BoiseState_orange, "green", "purple"])

    ax.plot(time, wind, lw=3, color=BoiseState_blue, label="MEDA")

    cur_t0 = t0
    for i in range(len(zs)):
        ind = retrieve_relevant_times(time, cur_t0, sample_time)
        cur_color = next(colors)
        ax.plot(time[ind], scaled_windspeeds[i][ind],
             label=r'$%g\, {\rm cm}$' % zs[i], lw=6, color=cur_color, alpha=0.5)
        ax.plot([np.min(time[ind]), np.max(time[ind])],
            [averaged_windspeeds[i], averaged_windspeeds[i]], lw=6, ls='--', 
            color='k')
        cur_t0 += sample_time


    xfmt = md.DateFormatter('%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.grid(True)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=36)
    ax.set_xlabel("LMST - %g" % (start_time.hour), fontsize=36)
    ax.set_ylabel(r'$U\ \left( {\rm cm\ s^{-1}} \right)$', fontsize=36)
    ax.legend(loc='best', fontsize=18)
    return ax

def fit_lin_fit(zs, winds, std_winds):

    log_z = np.log(zs/np.min(zs))
    popt, pcov = curve_fit(lin_fit, log_z, winds, sigma=std_winds)
    unc = np.sqrt(np.diag(pcov))

    return log_z, popt, unc, pcov

def fit_wind_profile_and_drop_outliers(zs, averaged_windspeeds, std_windspeeds, 
    drop_outliers=True, num_sigma=5., rescale_unc=True):

    inlier_zs = zs
    inlier_averaged_windspeeds = averaged_windspeeds
    inlier_std_windspeeds = std_windspeeds

    outlier_zs = np.array([])
    outlier_averaged_windspeeds = np.array([])
    outlier_std_windspeeds = np.array([])

    if(drop_outliers):
        inlier_zs = zs[0]
        inlier_averaged_windspeeds = averaged_windspeeds[0]
        inlier_std_windspeeds = std_windspeeds[0]

        for i in range(1, len(zs)):
            if(np.all(averaged_windspeeds[i] > averaged_windspeeds[0:i])):
                inlier_zs = np.append(inlier_zs, zs[i])
                inlier_averaged_windspeeds =\
                    np.append(inlier_averaged_windspeeds, 
                    averaged_windspeeds[i])
                inlier_std_windspeeds =\
                    np.append(inlier_std_windspeeds, std_windspeeds[i])
            else:
                outlier_zs = np.append(outlier_zs, zs[i])
                outlier_averaged_windspeeds =\
                    np.append(outlier_averaged_windspeeds, 
                    averaged_windspeeds[i])
                outlier_std_windspeeds = np.append(outlier_std_windspeeds, 
                    std_windspeeds[i])

        log_z, popt, unc, pcov = fit_lin_fit(inlier_zs, 
            inlier_averaged_windspeeds, inlier_std_windspeeds)

        # And now let's toss outliers
        inliers = np.abs(inlier_averaged_windspeeds -\
            np.polyval(popt, log_z))/inlier_std_windspeeds <= num_sigma

        if(len(inlier_averaged_windspeeds[~inliers]) > 0):
            outlier_zs = np.append(outlier_zs, inlier_zs[~inlier])
            outlier_averaged_windspeeds =\
                np.append(outlier_averaged_windspeeds,
                inlier_averaged_windspeeds[~inlier])
            outlier_std_windspeeds = np.append(outlier_std_windspeeds,
                inlier_std_windspeeds[~inlier])        

        inlier_zs = inlier_zs[inliers]
        inlier_averaged_windspeeds = inlier_averaged_windspeeds[inliers]
        inlier_std_windspeeds = inlier_std_windspeeds[inliers]

    log_z, popt, unc, pcov = fit_lin_fit(inlier_zs, inlier_averaged_windspeeds,
        inlier_std_windspeeds)

    # Rescale uncertainties
    #_NR_, 3rd ed, p. 783 - This equation provides a way to rescale
    # uncertainties, enforcing reduced chi-squared = 1
    if(rescale_unc):
        mod = np.polyval(popt, log_z)
        chisq = chisqg(inlier_averaged_windspeeds, mod,
            sd=inlier_std_windspeeds)

        red_chi_sq = chisq/(len(inlier_averaged_windspeeds) - len(popt))

        unc *= np.sqrt(red_chi_sq)
        inlier_std_windspeeds *= np.sqrt(red_chi_sq)
        pcov *= red_chi_sq

    return inlier_zs, inlier_averaged_windspeeds, inlier_std_windspeeds,\
        outlier_zs, outlier_averaged_windspeeds, outlier_std_windspeeds,\
        popt, unc, pcov

def collect_fit_values_and_unc(popt, unc, pcov, kappa=0.4):

    ustar = calc_ustar_from_slope(kappa, popt[0])
    zstar = calc_zstar_from_slope_and_intercept(np.min(zs), popt[0], popt[1])
    sigma_ustar = 0.4*unc[0]
    sigma_zstar = zstar*popt[1]/popt[0]*np.sqrt((unc[0]/popt[0])**2 +\
        (unc[1]/popt[1])**2 - 2*pcov[1,0]/popt[0]/popt[1])

    return ustar, zstar, sigma_ustar, sigma_zstar

def make_plot_of_wind_data_and_profile(inlier_zs, inlier_averaged_windspeeds, 
    inlier_std_windspeeds, outlier_zs, outlier_averaged_windspeeds, 
    outlier_std_windspeeds, popt, unc, pcov, ax):

    ustar, zstar, sigma_ustar, sigma_zstar =\
        collect_fit_values_and_unc(popt, unc, pcov)

    ax.errorbar(inlier_averaged_windspeeds, inlier_zs, 
        xerr=inlier_std_windspeeds, 
        marker='o', markersize=10, color=BoiseState_blue, ls='')

    # Show tossed points
    ax.errorbar(outlier_averaged_windspeeds, outlier_zs,
    xerr=outlier_std_windspeeds,
    marker='x', markersize=10, color='k', ls='')

    log_z = np.log(zs/np.min(zs))
    ax.plot(np.polyval(popt, log_z), zs, 
        lw=6, color=BoiseState_orange, ls='--', zorder=-1)

    ax.grid(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(labelsize=24)
    ax.set_xlabel(r'$\langle U \rangle \, \left( {\rm cm\ s^{-1} } \right)$', 
        fontsize=36)
    ax.set_ylabel(r'$z\, \left( {\rm cm } \right)$', fontsize=36)

    ax.text(0.05, 0.90, "(b)", fontsize=48, transform=ax.transAxes)
    ax.text(0.05, 0.825, 
        r'$u_\star = \left( %.0f\pm%.0f \right)\,{\rm cm\ s^{-1}}$' %\
        (ustar, sigma_ustar), fontsize=28, transform=ax.transAxes, 
        color=BoiseState_orange)
    ax.text(0.05, 0.775, r'$z_\star = \left( %.2f\pm%.2f \right)\,{\rm cm}$' %\
        (zstar, sigma_zstar), fontsize=28, transform=ax.transAxes, 
        color=BoiseState_orange)

    return ax
