"""STAR SHADOW
Satellite Time series Analysis Routine using
Sinusoids and Harmonics in an Automated way for Double stars with Occultations and Waves

This Python script is meant to be run before first use,
it ensures that the Just-In-Time compiler has done its job.
If your own use case involves time series longer than a few
thousand data points, this is strongly recommended.
If not, this is less important, but do keep in mind that the
first run will be slower.

Code written by: Luc IJspeert
"""

import os
import star_shadow as sts


# get the path to the test light curve
script_dir = os.path.dirname(os.path.abspath(__file__))  # absolute dir the script is in
data_dir = script_dir.replace('star_shadow', 'data')
file = os.path.join(data_dir, 'sim_000_lc.dat')
# execute the code
sts.analyse_lc_from_file(file, p_orb=0, i_sectors=None, stage='all', method='fitter', data_id=None, overwrite=True,
                         verbose=True)
