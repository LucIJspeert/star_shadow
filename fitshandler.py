"""This module contains the interactions with fits files.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.visualization import astropy_mpl_style


def print_info(file_name):
    """Shows the info of the HDUlist"""
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
        
    # clean way to open file (closes it automagically)
    with fits.open(file_name) as hdul:
        hdul.info()
    return

    
def print_keys(file_name, index=0):
    """Shows the keywords for the cards. Optional arg: HDUlist index."""
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    
    with fits.open(file_name) as hdul:
        hdr = hdul[index].header
        print(list(hdr.keys()))
    return

    
def print_hdr(file_name, index=0, hdr_range=None):
    """Prints the header. Optional args: HDUlist index, header range."""
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'

    if not hdr_range:
        hdr_range = [0, -1]
    
    with fits.open(file_name) as hdul:
        hdr = hdul[index].header[hdr_range[0]:hdr_range[1]]
        print(repr(hdr), '\n')
    return

    
def print_card(file_name, keyword, index=0, card_index=None):
    """Prints card: keyword (str or int). Optional arg: HDUlist index, card index."""
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    
    with fits.open(file_name) as hdul:
        if (card_index is None):
            crd = (str.upper(keyword) + ' = ' + str(hdul[index].header[keyword]) 
                   + '       / ' + hdul[index].header.comments[keyword])
        else:
            # for history or comment cards
            crd = str.upper(keyword) + ' = ' + str(hdul[index].header[keyword][card_index])
        print(crd)
    return


def print_data(file_name, index=0):
    """Prints the data. Optional arg: HDUlist index."""
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
        
    with fits.open(file_name) as hdul:
        print(hdul[index].data)
    return
    
    
def change_hdr(file_name, keyword, value, comment='', index=0):
    """Adds/updates card 'keyword' (str) in the current file. 
    Input: 'value' (str or number) and optionally 'comment' (str). 
    Optional arg: HDUlist index.
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    
    with fits.open(file_name, mode='update') as hdul:
        hdul[index].header.set(keyword, value, comment)
    return


def change_data(file_name, input_data, index=0):
    """Changes (and saves) the data in the current fits file. 
    Optional arg: HDUlist index.
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    
    with fits.open(file_name, mode='update') as hdul:
        hdul[index].data = input_data
    return


def get_card_value(file_name, keyword, index=0):
    """Returns the value of card 'keyword' (str). Returns 0 if value is a string. 
    Optional arg: HDUlist index.
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    
    with fits.open(file_name) as hdul:
        value = hdul[index].header[keyword]
    return value


def get_data(file_name, index=0):
    """Returns the requested data. [NOTE: data[1, 4] gives pixel value at x=5, y=2.] 
    Optional arg: HDUlist index.
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
            
    with fits.open(file_name) as hdul:
        return hdul[index].data


def new_hdr(keywords, values, comments=None):
    """Returns a new header object. 
    Inputs are lists of the keywords (str), values (str or number) and optional comments (str).
    """
    if (len(keywords) != len(values)):
        raise ValueError('Must enter as much values as keywords.')
    elif ((not hasattr(keywords, '__len__')) | (not hasattr(values, '__len__'))):
        raise ValueError('Arguments have length.')

    if not comments:
        comments = ['' for i in range(len(keywords))]
    elif (len(comments) != len(keywords)):
        raise ValueError('Must enter as much comments as keywords.')

    hdr = fits.Header()
    for i in range(len(keywords)):
        hdr.set(keywords[i], values[i], comments[i])
    return hdr


def new_fits(file_name, input_data, input_header=None):
    """Saves the input_data to a new file 'file_name'. 
    Optional arg: input_header (header object)
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    
    fits.writeto(file_name, input_data, header=input_header)
    return


def add_to_fits(file_name, input_data, input_header=None):
    """Appends the header/data to fits file if 'file_name' exists, creates one if not. 
    Optional arg: input_header (header object).
    """
    if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
        file_name += '.fits'
    
    fits.append(file_name, input_data, header=input_header)
    return


def plot_fits(file_name, index=0, colours='gray', scale='lin', grid=False, chip='single', show=True):
    """Displays the image in a fits file. Optional args: HDUlist index, colours.
    Can also take image objects directly.
    scale can be set to 'lin', 'sqrt', and 'log'
    chip='single': plots single data array at given index.
        ='full': expects data in index 1-9 and combines it.
    """
    if isinstance(file_name, str):
        if ((file_name[-5:] != '.fits') & (file_name[-4:] != '.fit')):
            file_name += '.fits'
            
        if (chip == 'single'):
            image_data = get_data(file_name, index)
        elif (chip == 'full'):
            image_data = [get_data(file_name, i + 1) for i in range(9)]
        else:
            raise ValueError('Chip configuration not recognised.')
    else:
        if (chip == 'single'):
            image_data = file_name[index].data
        elif (chip == 'full'):
            image_data = [file_name[i+1].data for i in range(9)]
        else:
            raise ValueError('Chip configuration not recognised.')
            
    if (chip == 'full'):
        image_data_r1 = np.concatenate(image_data[6], image_data[7],
                                       image_data[8], axis=1)
        image_data_r2 = np.concatenate(image_data[5], image_data[4],
                                       image_data[3], axis=1)
        image_data_r3 = np.concatenate(image_data[0], image_data[1],
                                       image_data[2], axis=1)
        # stitch all chips together
        image_data = np.concatenate(image_data_r1, image_data_r2, image_data_r3, axis=0)
            
    if (scale == 'log'):
        image_data = np.log10(image_data - np.min(image_data))
    elif (scale == 'sqrt'):
        image_data = (image_data - np.min(image_data))**(1/2)
    
    # use nice plot parameters
    plt.style.use(astropy_mpl_style)
    
    fig, ax = plt.subplots(figsize=[12.0, 12.0])
    ax.grid(grid)
    cax = ax.imshow(image_data, cmap=colours)
    fig.colorbar(cax)
    plt.tight_layout()
    if show:
        plt.show()
    return






















