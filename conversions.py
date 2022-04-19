"""This module contains coordinate, unit and possibly other conversions.
"""

import numpy as np


# constants
deg_to_h = 24/360
deg_to_min = 24*60/360
deg_to_s = 24*3600/360


def decimal_ra_dec_to_formatted(ra, dec, string=False):
    """Formats decimal RA, DEC coordinates to: h m s. deg am as."""
    ra_h = np.floor(ra * deg_to_h).astype(int)
    ra_m = np.floor((ra - ra_h / deg_to_h) * deg_to_min).astype(int)
    ra_s = (ra - (ra_h / deg_to_h) - (ra_m / deg_to_min)) * deg_to_s
    dec_deg = (np.sign(dec) * np.floor(np.abs(dec))).astype(int)
    dec_am = np.floor(np.abs(dec - dec_deg) * 60).astype(int)
    dec_as = np.abs(dec - dec_deg - np.sign(dec_deg)*dec_am/60) * 3600
    if string:
        output = f'{ra_h:2} {ra_m:2} {ra_s:2.8} {dec_deg:2} {dec_am:2} {dec_as:2.8}'
    else:
        output = ra_h, ra_m, ra_s, dec_deg, dec_am, dec_as
    return output


def julian_day(year, month, day, gregorian=1582):
    """Gives the Julian day.
    The day is decimal, starting at 1, so noon on the first day of the month is 1.5!
    Set gregorian to None for Julian calendar instead.
    """
    year = np.atleast_1d(year)
    month = np.atleast_1d(month)
    day = np.atleast_1d(day)
    
    m = (month + 12) * (month < 3) + month * (month >= 3)
    y = year + 4716 - 1 * (month < 3)
    
    if gregorian is None:
        # Julian calendar
        B = 0
    else:
        # Gregorian calendar
        A = (year / 100).astype(int)
        B = (2 - A + (A / 4).astype(int)) * (year > gregorian) + 0 * (year <= gregorian)
    
    jd = (365.25 * y).astype(int) + (30.6001 * (m + 1)).astype(int) + day + B - 1524.5
    return jd


def counts_to_mag(flux_counts, m_0):
    """Converts from flux (counts) to magnitude via a magnitude zero point."""
    return m_0 - 2.5 * np.log10(flux_counts)


def counts_to_ppo(flux_counts, flux_err=None):
    """Converts from flux (counts) to parts per one.
    The result varies around one.
    Give error values to have them converted too.
    """
    med = np.median(flux_counts)
    if flux_err is None:
        return flux_counts / med
    else:
        return flux_counts / med, flux_err / med


def counts_to_ppm(flux_counts, flux_err=None):
    """Converts from flux (counts) to parts per million.
    The result varies around zero.
    """
    if flux_err is None:
        flux_ppo = counts_to_ppo(flux_counts)
        return 1e6 * (flux_ppo - 1)
    else:
        flux_ppo, flux_err_ppo = counts_to_ppo(flux_counts, flux_err)
        return 1e6 * (flux_ppo - 1), 1e6 * flux_err_ppo


def ppm_to_ppo(flux_ppm):
    """Converts from parts per million to parts per one.
    It is assumed that the ppm is around zero. The result varies around one.
    """
    return (flux_ppm / 1e6) + 1


def ppo_to_ppm(flux_ppo):
    """Converts from parts per one to parts per million.
    It is assumed that the ppo is around one. The result varies around zero.
    """
    return (flux_ppo - 1) * 1e6


def ppm_to_mag(flux_ppm):
    """Converts from parts per million to magnitude (varying around zero)."""
    return -2.5 * np.log10(ppm_to_ppo(flux_ppm))


def mag_to_ppm(mag):
    """Converts from magnitude (varying around zero) to parts per million."""
    return 10**(-0.4 * mag)
