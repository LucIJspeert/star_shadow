# STAR SHADOW
### Satellite Time-series Analysis Routine using Sinusoids and Harmonics of Double stars with Occultations and Waves


## What is STAR SHADOW?
STAR SHADOW is a Python code that is aimed at the automatic analysis of space based light curves of eclipsing binaries and provide a measurement of eccentricity, among other parameters. It provides a recipe to measure timings of eclipses using the time derivatives of the light curves, using a model of orbital harmonics obtained from an initial iterative prewhitening of sinusoids. Since the algorithm extracts the harmonics from the rest of the sinusoidal variability eclipse timings can be measured even in the presence of other (astrophysical) signals.
The aim is to determine the orbital eccentricity automatically from the light curve along with information about the other variability present in the light curve. The output includes, but is not limited to, a sinusoid plus piece-wise linear model of the light curve, the orbital period, the eccentricity, argument of periastron and inclination. See the documentation for more information.


### Reference Material

* This algorithm has been documented, tested and applied in the publication: [link](link)


## Getting started


**STAR SHADOW has only been tested in Python 3.8**. 


**Package dependencies:** The following package versions have been used in the development of this code, meaning older versions can in principle work, but this is not guaranteed. 

### Example use



### Explanation of output



## Bugs and Issues

Despite all the testing, I am certain that there are still bugs in this code, or will be created in future versions. 

If you happen to come across any bugs or issues, *please* contact me. Only known bugs can be resolved.
This can be done through opening an issue on the STAR SHADOW GitHub page: [LucIJspeert/star_shadow/issues](https://github.com/LucIJspeert/star_shadow/issues), or by contacting me directly (see below).

If you are (going to be) working on new or improved features, I would love to hear from you and see if it can be implemented in the source code.


## Contact

For questions and suggestions, please contact:

* luc.ijspeert(at)kuleuven.be

**Developer:** Luc IJspeert (KU Leuven)
