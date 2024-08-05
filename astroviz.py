"""
MIT License

Copyright (c) 2024 ethanChou314

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author: Hsuan-I (Ethan) Chou

Description of module:
    This module is a comprehensive tool for advanced data analysis and visualization in 
    radio astronomy. The module includes functionalities for importing FITS files, 
    handling and manipulating data cubes, spatial maps, and position-velocity (PV) diagrams. 
    Users can intuitively perform common Python operations (addition, subtraction, 
    multiplication, etc.) on the image objects and numpy functions (the objects act like 
    numpy arrays!). For example, to visualize the square root of the addition of two images:
    
    import astroviz as av
    import numpy as np
    image1 = av.importfits('directory_to_image1.fits')
    image2 = av.importfits('directory_to_image2.fits')
    added_image = image1 + image2
    sqrt_image = np.sqrt(added_image)
    sqrt_image.imview()
    
    Other key features include generating moment maps, extracting PV diagrams and 
    spatial/spectral profiles, two-dimensional and one-dimensional Gaussian fitting,
    rotating/shifting images, regridding images into different cell sizes, convolution, 
    estimating RMS noise levels using sigma clipping, plotting regions on maps, 
    extracting channel maps from data cubes, visualizing both two-dimensional images 
    (PV diagrams or spatial maps like moment maps and continuum maps) and 
    three-dimensional images (channel maps), and many more. Additionally, this module
    is fully aware of astropy units and includes smaller intuitive features such as 
    automatically adjusting plot labels based on intensity/axis units. The module's 
    flexibility allows for straightforward customization and extension.
    
    This module consists of the main classes: Datacube, Spatialmap, PVdiagram, and Region. 
    The former three classes are three main image types in radio astronomy, while the class,
    Region, can be inputted in various methods (e.g., Datacube.pvextractor, 
    Spatialmap.get_xyprofile, etc.). The 'importfits' function can be used adaptively to 
    generate either of the three classes. 
    
    Regions can be set using the 'Region' class constructor by reading a given DS9 file or 
    inputing the necessary parameters. 
    For example:
        import astroviz as av
        region1 = av.Region('directory_to_region_file')                # read given file
        region2 = av.Region(shape='line', start=(0, 1), end=(1, 3), 
                            unit='arcsec', relative=True)              # input parameters
        region3 = av.Region(shape='ellipse', 
                            center='05h55m38.198s +02d11m33.58s',
                            semimajor=3.5, semiminor=2.1, pa=36)       # another example
"""


# utils package (standard tools)
from utils import (_unnormalize_location, _normalize_location, _change_cell_size, _trim_data, _plt_cmap, 
                   _prevent_overwriting, _plt_cbar_and_set_aspect_ratio, _find_file, _get_contour_beam_loc,
                   _true_round_whole, _get_hduheader, _icrs2relative, _customizable_scale,
                   _relative2icrs, _to_apu, _apu_to_str, _apu_to_headerstr, _best_match_line,
                   _unit_plt_str, _convert_Bunit, _Qrot_linear, _get_beam_dim_along_pa,
                   _Qrot_linear_polyatomic, _match_limits, _get_optimal_columns,)

# numerical packages
import numpy as np
import pandas as pd

# standard packages
import copy
import os
import sys
import datetime as dt
import string
import inspect
import warnings
import itertools
from typing import *

# scientific packages
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy import interpolate
from astropy import units as u, constants as const
from astropy.units import Unit, def_unit, UnitConversionError
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from astropy.coordinates import Angle, SkyCoord
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian2DKernel, Box2DKernel, convolve, convolve_fft

# parallel processing
from concurrent.futures import ProcessPoolExecutor

# data visualization packages
import matplotlib as mpl
from matplotlib import cm, rcParams, ticker, patches, colors, pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# variables to control all map fonts globally
_fontfamily = "Times New Roman"
_mathtext_fontset = "stix"
_mathtext_tt = "Times New Roman"


def importfits(fitsfile, hduindex=0, spatialunit="arcsec", specunit="km/s", quiet=False):
    """
    This function reads the given FITS file directory and returns the corresponding image object.
    Parameters:
        fitsfile (str): the directory of the FITS file. 
        hduindex (int): the index of the HDU in the list of readable HDUs
        spatialunit (str): the unit of the spatial axes
        specunit (str): the unit of the spectral axis
        quiet (bool): Nothing will be printed to communicate with user. 
                      Useful for iterative processes, as it allows for faster file reading time.
    Returns:
        The image object (Datacube/Spatialmap/PVdiagram)
    ----------
    Additional notes:
        - If given fitsfile directory does not exist, the function will attempt to recursively 
        find the file within all subdirectories.
        - A temporary FITS file will be made from the CASA image format and then deleted 
          if the specified file is a directory. This would require the 'casatasks' module to 
          be imported within this function.
    """
    
    if isinstance(fitsfile, str):
        # try to recursively find file if relative directory does not exist
        if not os.path.exists(fitsfile):
            if not quiet:
                print(f"Given directory '{fitsfile}' does not exist as a relative directory. " + \
                       "Recursively finding file...")
            maybe_filename = _find_file(fitsfile)
            if maybe_filename is not None:
                fitsfile = maybe_filename
                if not quiet:
                    print(f"Found a matching filename: '{fitsfile}'")
            else:
                raise FileNotFoundError(f"Filename '{fitsfile}' does not exist.")

        # obtain the HDU list
        if os.path.isfile(fitsfile):  
            hdu_lst = fits.open(fitsfile)
        else:  # read as CASA image if it is a directory
            import casatasks
            dt_str = str(dt.datetime.now())

            # convert CASA image format to a temporary FITS file, then read the FITS file.
            temp_fitsfile = dt_str.replace(" ", "") + ".fits"
            temp_fitsfile = _prevent_overwriting(temp_fitsfile)
            casatasks.exportfits(fitsfile, temp_fitsfile, history=False)
            hdu_lst = fits.open(temp_fitsfile)
            os.remove(temp_fitsfile)  # remove the temporary FITS file
        
        # read HDU list
        if len(hdu_lst) == 1:
            data = hdu_lst[0].data
            hdu_header = dict(hdu_lst[0].header)
        else:
            good_hdu = [item for item in hdu_lst \
                        if item.data is not None and item.header is not None \
                        and not isinstance(item, fits.hdu.table.BinTableHDU)]
            if len(good_hdu) == 0:
                raise Exception("Cannot find any readable HDUs.")
            else:
                if len(good_hdu) >= 2:
                    if not quiet:
                        print()
                        print(f"Found {len(good_hdu)} readable HDUs." +  
                              "Change 'hduindex' parameter to read a different HDU.")
                        print("Below are the readable HDUs and their corresponding 'hduindex' parameters:")
                        for i in range(len(good_hdu)):
                            print(f"[{i}] {good_hdu[i]}")
                        print()
                if not isinstance(hduindex, int):
                    hduindex = int(hduindex)
                data = good_hdu[hduindex].data
                hdu_header = dict(good_hdu[hduindex].header)
    elif isinstance(fitsfile, (fits.hdu.image.PrimaryHDU, fits.PrimaryHDU)):
        hdu_lst = [fitsfile]
        data = fitsfile.data 
        hdu_header = fitsfile.header
        fitsfile = ''  # still make it represent a directory (empty string denotes not imported from anywhere)
    else:
        raise ValueError("'fitsfile' must be a absolute/relative directory or a 'PrimaryHDU' instance.")

    # store ctypes as list, 1-based indexing
    ctype = [hdu_header.get(f"CTYPE{i}", "") for i in range(1, data.ndim+1)] 
    
    # get rest frequency
    if "RESTFRQ" in hdu_header:
        restfreq = hdu_header["RESTFRQ"]
    elif "RESTFREQ" in hdu_header:
        restfreq = hdu_header["RESTFREQ"]
    elif "FREQ" in hdu_header:
        restfreq = hdu_header["FREQ"]
    else:
        restfreq = np.nan
        warnings.warn("Failed to read rest frequency. It is set to NaN in the header.")
    
    # stokes axis
    nstokes = hdu_header.get(f"NAXIS{ctype.index('STOKES')+1}", 1) if "STOKES" in ctype else 1

    # frequency axis
    if "FREQ" in ctype:
        freq_axis_num = ctype.index("FREQ") + 1
        nchan = hdu_header.get(f"NAXIS{freq_axis_num}")
        is_vel = False
    elif "VRAD" in ctype:
        freq_axis_num = ctype.index("VRAD") + 1
        nchan = hdu_header.get(f"NAXIS{freq_axis_num}")
        is_vel = True
    elif "VELO" in ctype:
        freq_axis_num = ctype.index("VELO") + 1
        nchan = hdu_header.get(f"NAXIS{freq_axis_num}")
        is_vel = True
    elif "VELO-LSR" in ctype:
        freq_axis_num = ctype.index("VELO-LSR") + 1
        nchan = hdu_header.get(f"NAXIS{freq_axis_num}")
        is_vel = True
    else:
        nchan = 1
       
    # cannot combine with previous statement because it can be an image with only 1 channel
    if nchan == 1:
        dv = None
        vrange = None
        vtype = None
        vunit = None
    else:
        vunit = hdu_header.get(f"CUNIT{freq_axis_num}", ("km/s" if is_vel else "Hz"))
        vrefpix = hdu_header.get(f"CRPIX{freq_axis_num}", 1)
        vrefval = hdu_header.get(f"CRVAL{freq_axis_num}", np.nan)
        dv = hdu_header.get(f"CDELT{freq_axis_num}", np.nan)
        startv = dv*(vrefpix-1) + vrefval
        endv = dv*(vrefpix+nchan-2) + vrefval
        if _to_apu(vunit) != _to_apu(specunit):
            equiv = u.doppler_radio(restfreq*u.Hz)
            startv = u.Quantity(startv, vunit).to_value(specunit, equivalencies=equiv)
            endv = u.Quantity(endv, vunit).to_value(specunit, equivalencies=equiv)
            dv = (endv-startv)/(nchan-1)
            if specunit == "km/s":
                rounded_startv = round(startv, 7)
                if np.isclose(startv, rounded_startv):
                    start = rounded_startv
                rounded_endv = round(endv, 7)
                if np.isclose(endv, rounded_endv):
                    endv = rounded_endv
                rounded_dv = round(dv, 7)
                if np.isclose(dv, rounded_dv):
                    dv = rounded_dv
        vrange = [startv, endv]
        
    # initialize projection parameter, updates if finds one
    projection = None
    
    # right ascension / offset axis
    ra_mask = ["RA" in item for item in ctype]
    if any(ra_mask):  # checks if "RA" axis exists
        xidx = ra_mask.index(True)
        projection = ctype[xidx].split("-")[-1]
        xaxis_num = xidx + 1
        xunit = hdu_header.get(f"CUNIT{xaxis_num}", "deg")
        refx = hdu_header.get(f"CRVAL{xaxis_num}")
        nx = hdu_header.get(f"NAXIS{xaxis_num}")
        dx = hdu_header.get(f"CDELT{xaxis_num}")
        refnx = int(hdu_header.get(f"CRPIX{xaxis_num}"))
        if xunit != spatialunit:
            if dx is not None:
                dx = u.Quantity(dx, xunit).to_value(spatialunit)
            if refx is not None:
                refx = u.Quantity(refx, xunit).to_value(spatialunit)
    elif "OFFSET" in ctype:
        xaxis_num = ctype.index("OFFSET") + 1
        xunit = hdu_header.get(f"CUNIT{xaxis_num}", "deg")
        refx = hdu_header.get(f"CRVAL{xaxis_num}")
        nx = hdu_header.get(f"NAXIS{xaxis_num}")
        dx = hdu_header.get(f"CDELT{xaxis_num}")
        refnx = int(hdu_header.get(f"CRPIX{xaxis_num}"))
        if xunit != spatialunit:
            if dx is not None:
                dx = u.Quantity(dx, xunit).to_value(spatialunit)
            if refx is not None:
                refx = u.Quantity(refx, xunit).to_value(spatialunit)
    else:
        nx = 1
        refx = None
        dx = None
        refnx = None
        warnings.warn("Failed to read right ascension / offset axis information.")
    
    # declination axis
    dec_mask = ["DEC" in item for item in ctype]
    if any(dec_mask):    # if dec axis exists
        yidx = dec_mask.index(True) 
        if projection is None:
            projection = ctype[yidx].split("-")[-1]
        yaxis_num = yidx + 1
        yunit = hdu_header.get(f"CUNIT{yaxis_num}", "deg")
        refy = hdu_header.get(f"CRVAL{yaxis_num}")
        ny = hdu_header.get(f"NAXIS{yaxis_num}")
        dy = hdu_header.get(f"CDELT{yaxis_num}")
        refny = int(hdu_header.get(f"CRPIX{yaxis_num}"))
        if yunit != spatialunit:
            if dy is not None:
                dy = u.Quantity(dy, yunit).to_value(spatialunit)
            if refy is not None:
                refy = u.Quantity(refy, yunit).to_value(spatialunit)
    else:
        ny = 1
        refy = None
        dy = None
        refny = None
        
    # set reference coordinates
    if refx is not None and refy is not None:
        refcoord = SkyCoord(ra=u.Quantity(refx, spatialunit), 
                            dec=u.Quantity(refy, spatialunit)).to_string('hmsdms')
    else:
        refcoord = None
    
    # determine image type and reshape data
    if nx > 1 and ny > 1 and nchan > 1:
        imagetype = "datacube"
        newshape = (nstokes, nchan, ny, nx)
        if data.shape != newshape:
            data = data.reshape(newshape)
    elif nx > 1 and ny > 1 and nchan == 1:
        imagetype = "spatialmap"
        newshape = (nstokes, nchan, ny, nx)
        if data.shape != newshape:
            data = data.reshape(newshape)
    elif nchan > 1 and nx > 1 and ny == 1:
        imagetype = "pvdiagram"
        newshape = (nstokes, nchan, nx)
        if data.shape != newshape:
            if data.shape == (nstokes, nx, nchan):
                data = data[0].T[None, :, :]  # transpose if necessary
            else:
                data = data.reshape(newshape)
                warnings.warn("Data of PV diagram will be reshaped.")
    else:
        raise Exception("Image cannot be read as 'datacube', 'spatialmap', or 'pvdiagram'.")
    
    # get beam size
    if all(keyword in hdu_header for keyword in ("BMAJ", "BMIN", "BPA")):
        bmaj = hdu_header["BMAJ"] * u.deg  # assume unit is degrees
        bmin = hdu_header["BMIN"] * u.deg  # assume unit is degrees
        bpa =  hdu_header["BPA"]  # assume unit is degrees

    elif len(hdu_lst) > 1 and isinstance(hdu_lst[1], fits.hdu.table.BinTableHDU):
        # sometimes the primary beam info is stored as second file in hdu list:
        beam_info: fits.hdu.table.BinTableHDU = hdu_lst[1]
        names: List[str] = beam_info.columns.names
        beam_data: fits.fitsrec.FITS_rec = beam_info.data

        # make beam header in correct format (only keep 'hashable keys' when reversed):
        beam_header = dict((key, value) for key, value in dict(beam_info.header).items() \
                            if isinstance(value, str))
        
        # swap keys and values in the dictionary (to find index of information needed):
        beam_header_swapped = dict(zip(beam_header.values(), beam_header.keys())) 
        
        # major axis
        if "BMAJ" in names:
            bmaj_idx = beam_header_swapped["BMAJ"][-1]  # index (1-based) in header
            bmaj_median = np.median(beam_data["BMAJ"])  # use median value
            bmaj_unit = beam_header["TUNIT"+bmaj_idx]  # find unit in header
            bmaj = u.Quantity(bmaj_median, bmaj_unit)
        else:
            bmaj = np.nan

        # minor axis
        if "BMIN" in names:
            bmin_idx = beam_header_swapped["BMIN"][-1]  # index (1-based) in header
            bmin_median = np.median(beam_data["BMIN"])  # use median value 
            bmin_unit = beam_header["TUNIT"+bmin_idx]  # find unit in header
            bmin = u.Quantity(bmin_median, bmin_unit)
        else:
            bmin = np.nan

        # position angle
        if "BPA" in names:
            bpa_idx = beam_header_swapped["BPA"][-1]
            bpa_median = np.median(beam_data["BPA"])
            bpa_unit = beam_header["TUNIT"+bpa_idx]
            bpa = u.Quantity(bpa_median, bpa_unit).to_value(u.deg)  # convert directly to degrees
        else:
            bpa = np.nan

    else:
        # assign NaN values if information cannot be located:
        bmaj = np.nan 
        bmin = np.nan 
        bpa = np.nan
    
    if spatialunit in ("deg", "degrees", "degree"):  # convert beam size unit if necessary
        if not np.isnan(bmaj):
            bmaj = bmaj.value
        if not np.isnan(bmin):
            bmin = bmin.value
    else:
        if not np.isnan(bmaj):
            bmaj = bmaj.to_value(spatialunit)
        if not np.isnan(bmin):
            bmin = bmin.to_value(spatialunit)
    
    # eliminate rounding error due to float64
    if dx is not None:
        dx = np.round(dx, 7)
    if dy is not None:
        dy = np.round(dy, 7)
        
    # store beam as a tuple
    beam = (bmaj, bmin, bpa)

    # raise warning if one of the beam dimensions cannot be read.
    if np.any(np.isnan(beam)):
        warnings.warn("Failed to read all beam dimensions. It is set to NaN in the header. " + \
              "Certain unit conversions may be inaccurate.")
            
    # input information into dictionary as header information of image
    header = {"filepath": fitsfile,
              "shape": newshape,
              "imagetype": imagetype,
              "nstokes": nstokes,
              "vrange": vrange,
              "dv": dv,
              "nchan": nchan,
              "dx": dx, 
              "nx": nx,
              "refnx": refnx,
              "dy": dy,
              "ny": ny,
              "refny": refny,
              "refcoord": refcoord,
              "restfreq": restfreq,
              "beam": beam,
              "specframe": hdu_header.get("RADESYS", "ICRS"),
              "unit": spatialunit,
              "specunit": '' if imagetype == "spatialmap" else _apu_to_headerstr(_to_apu(specunit)),
              "bunit": hdu_header.get("BUNIT", ""),
              "projection": projection,
              "object": hdu_header.get("OBJECT", ""),
              "instrument": hdu_header.get("INSTRUME", ""),
              "observer": hdu_header.get("OBSERVER", ""),
              "obsdate": hdu_header.get("DATE-OBS", ""),
              "date": hdu_header.get("DATE", ""),
              "origin": hdu_header.get("ORIGIN", ""),
              }

    # return image
    if imagetype == "datacube":
        return Datacube(header=header, data=data)
    elif imagetype == "spatialmap":
        return Spatialmap(header=header, data=data)
    elif imagetype == "pvdiagram":
        return PVdiagram(header=header, data=data)


def set_font(font):
    """
    Function to control the font of all images globally.
    Parameters:
        font (str): the font of the image. 
                    Supported options include 'Times New Roman', 'Helvetica', 'Arial',
                    'Georgia', 'Garamond', 'Verdana', 'Calibri', 'Roboto', 'Courier New',
                    'Consolas'.
    """
    global _fontfamily, _mathtext_fontset, _mathtext_tt
    
    font_case_insensitive = font.lower()
    if font_case_insensitive == "times new roman":
        _fontfamily = "Times New Roman"
        _mathtext_fontset = "stix"
        _mathtext_tt = "Times New Roman"
    elif font_case_insensitive == "helvetica":
        _fontfamily = "Helvetica"
        _mathtext_fontset = "stixsans"
        _mathtext_tt = "Helvetica"
    elif font_case_insensitive == "arial":
        _fontfamily = "Arial"
        _mathtext_fontset = "custom"
        _mathtext_tt = "Arial"
    elif font_case_insensitive == "georgia":
        _fontfamily = "Georgia"
        _mathtext_fontset = "stix"
        _mathtext_tt = "Georgia"
    elif font_case_insensitive == "verdana":
        _fontfamily = "Verdana"
        _mathtext_fontset = "custom"
        _mathtext_tt = "Verdana"
    elif font_case_insensitive == "courier new":
        _fontfamily = "Courier New"
        _mathtext_fontset = "custom"
        _mathtext_tt = "Courier New"
    else:
        print("Unsupported font. Please manually enter the 'font.family', 'mathtext.fontset', " + \
              "and 'mathtext.tt' attributes of matplotlib.")
        _fontfamily = input("font.family: ")
        _mathtext_fontset = input("mathtext.fontset: ")
        _mathtext_tt = input("mathtext.tt: ")


class Datacube:
    """
    A class for handling FITS data cubes in astronomical imaging.

    This class provides functionality to load, process, and manipulate FITS data cubes. 
    It supports operations like arithmetic calculations between data cubes, rotation, 
    normalization, regridding, and more. The class can handle FITS files directly and acts
    like a numpy array.

    Note:
        The class performs several checks on initialization to ensure that the provided data
        is in the correct format. It can handle FITS files with different configurations and
        is designed to be flexible for various data shapes and sizes.
    """
    def __init__(self, fitsfile=None, header=None, data=None, hduindex=0, 
                 spatialunit="arcsec", specunit="km/s", quiet=False):
        if fitsfile is not None:
            fits = importfits(fitsfile, hduindex=hduindex, spatialunit=spatialunit, 
                              specunit=specunit, quiet=False)
            self.header = fits.header
            self.data = fits.data
        elif header is not None:
            self.header = header
            self.data = data
        if self.header["imagetype"] != "datacube":
            raise TypeError("The given FITS file cannot be read as a data cube.")
        self.__updateparams()
        
        if isinstance(self.data, u.quantity.Quantity):
            self.value = Datacube(header=self.header, data=self.data.value)
        
        self._peakshifted = False
        self.__pltnax = 0
        self.__pltnchan = 0
        
    def __updateparams(self):
        self.spatialunit = self.unit = self.axisunit = self.header["unit"]
        nx = self.nx = self.header["nx"]
        dx = self.dx = self.header["dx"]
        refnx = self.refnx = self.header["refnx"]
        ny = self.ny = self.header["ny"]
        dy = self.dy = self.header["dy"]
        refny = self.refny = self.header["refny"]
        self.xaxis, self.yaxis = self.get_xyaxes()
        self.shape = self.header["shape"]
        self.size = self.data.size
        self.restfreq = self.header["restfreq"]
        self.bmaj, self.bmin, self.bpa = self.beam = self.header["beam"]
        self.resolution = np.sqrt(self.beam[0]*self.beam[1]) if self.beam is not None else None
        self.refcoord = self.header["refcoord"]
        if isinstance(self.data, u.Quantity):
            self.bunit = self.header["bunit"] = _apu_to_headerstr(self.data.unit)
        else:
            self.bunit = self.header["bunit"]
        xmin, xmax = self.xaxis[[0, -1]]
        ymin, ymax = self.yaxis[[0, -1]]
        self.imextent = [xmin-0.5*dx, xmax+0.5*dx, 
                         ymin-0.5*dy, ymax+0.5*dy]
        self.widestfov = max(self.xaxis[0], self.yaxis[-1])
        self.specunit = self.header["specunit"]
        if self.specunit == "km/s":
            rounded_dv = round(self.header["dv"], 5)
            if np.isclose(self.header["dv"], rounded_dv):
                self.dv = self.header["dv"] = rounded_dv
            else:
                self.dv = self.header["dv"]
            specmin, specmax = self.header["vrange"]
            rounded_specmin = round(specmin, 5)
            rounded_specmax = round(specmax, 5)
            if np.isclose(specmin, rounded_specmin):
                specmin = rounded_specmin
            if np.isclose(specmax, rounded_specmax):
                specmax = rounded_specmax
            self.vrange = self.header["vrange"] = [specmin, specmax]
        else:
            self.dv = self.header["dv"]
            self.vrange = self.header["vrange"]
        self.nv = self.nchan = self.header["nchan"]        
        self.vaxis = self.get_vaxis()
        
        # magic methods to define operators
    def __add__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return Datacube(header=self.header, data=self.data+other.data)
        return Datacube(header=self.header, data=self.data+other)
    
    def __radd__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return Datacube(header=self.header, data=other.data+self.data)
        return Datacube(header=self.header, data=other+self.data)
        
    def __sub__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return Datacube(header=self.header, data=self.data-other.data)
        return Datacube(header=self.header, data=self.data-other)
    
    def __rsub__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return Datacube(header=self.header, data=other.data-self.data)
        return Datacube(header=self.header, data=other-self.data)
        
    def __mul__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data*other.data)
        return Datacube(header=self.header, data=self.data*other)
    
    def __rmul__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=other.data*self.data)
        return Datacube(header=self.header, data=other*self.data)
    
    def __pow__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data**other.data)
        return Datacube(header=self.header, data=self.data**other)
        
    def __rpow__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=other.data**self.data)
        return Datacube(header=self.header, data=other**self.data)
        
    def __truediv__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data/other.data)
        return Datacube(header=self.header, data=self.data/other)
    
    def __rtruediv__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=other.data/self.data)
        return Datacube(header=self.header, data=other/self.data)
        
    def __floordiv__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data//other.data)
        return Datacube(header=self.header, data=self.data//other)
    
    def __rfloordiv__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=other.data//self.data)
        return Datacube(header=self.header, data=other//self.data)
    
    def __mod__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data%other.data)
        return Datacube(header=self.header, data=self.data%other)
    
    def __rmod__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=other.data%self.data)
        return Datacube(header=self.header, data=other%self.data)
    
    def __lt__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data<other.data)
        return Datacube(header=self.header, data=self.data<other)
    
    def __le__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data<=other.data)
        return Datacube(header=self.header, data=self.data<=other)
    
    def __eq__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data==other.data)
        return Datacube(header=self.header, data=self.data==other)
        
    def __ne__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data!=other.data)
        return Datacube(header=self.header, data=self.data!=other)

    def __gt__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data>other.data)
        return Datacube(header=self.header, data=self.data>other)
        
    def __ge__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Datacube(header=self.header, data=self.data>=other.data)
        return Datacube(header=self.header, data=self.data>=other)

    def __abs__(self):
        return Datacube(header=self.header, data=np.abs(self.data))
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return Datacube(header=self.header, data=-self.data)
    
    def __invert__(self):
        return Datacube(header=self.header, data=~self.data)
    
    def __getitem__(self, indices):
        try:
            try:
                return Datacube(header=self.header, data=self.data[indices])
            except:
                warnings.warn("Returning value after reshaping image data to 2 dimensions.")
                return self.data.copy[:, indices[0], indices[1]]
        except:
            return self.data[indices]
    
    def __setitem__(self, indices, value):
        newdata = self.data.copy()
        newdata[indices] = value
        return Datacube(header=self.header, data=newdata)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Extract the Datacube object from inputs
        inputs = [x.data if isinstance(x, Datacube) else x for x in inputs]
        
        if ufunc == np.round:
            # Process the round operation
            if 'decimals' in kwargs:
                decimals = kwargs['decimals']
            elif len(inputs) > 1:
                decimals = inputs[1]
            else:
                decimals = 0  # Default value for decimals
            return Datacube(header=self.header, data=np.round(self.data, decimals))
        
        # Apply the numpy ufunc to the data
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Return a new Datacube instance with the result if the ufunc operation was successful
        if method == '__call__' and isinstance(result, np.ndarray):
            return Datacube(header=self.header, data=result)
        else:
            return result
        
    def __array__(self, *args, **kwargs):
        return np.array(self.data, *args, **kwargs)

    def min(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the minimum value of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's min function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's min function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the minimum.

        Returns:
            scalar or array: Minimum value of the data array. If the data array is multi-dimensional, 
                             returns an array of minimum values along the specified axis.
        """
        if ignore_nan:
            return np.nanmin(self.data, *args, **kwargs)
        return np.min(self.data, *args, **kwargs)

    def max(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the maximum value of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's max function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's max function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the maximum.

        Returns:
            scalar or array: Maximum value of the data array. If the data array is multi-dimensional, 
                             returns an array of maximum values along the specified axis.
        """
        if ignore_nan:
            return np.nanmax(self.data, *args, **kwargs)
        return np.max(self.data, *args, **kwargs)

    def mean(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the mean value of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's mean function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's mean function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the mean.

        Returns:
            scalar or array: Mean value of the data array. If the data array is multi-dimensional, 
                             returns an array of mean values along the specified axis.
        """
        if ignore_nan:
            return np.nanmean(self.data, *args, **kwargs)
        return np.mean(self.data, *args, **kwargs)

    def sum(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the sum of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's sum function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's sum function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the sum.

        Returns:
            scalar or array: Sum of the data array. If the data array is multi-dimensional, 
                             returns an array of sum values along the specified axis.
        """
        if ignore_nan:
            return np.nansum(self.data, *args, **kwargs)
        return np.sum(self.data, *args, **kwargs)
    
    def to(self, unit, *args, **kwargs):
        """
        This method converts the intensity unit of original image to the specified unit.
        """
        return Datacube(header=self.header, data=self.data.to(unit, *args, **kwargs))
    
    def to_value(self, unit, *args, **kwargs):
        """
        Duplicate of astropy.unit.Quantity's 'to_value' method.
        """
        image = self.copy()
        image.data = image.data.to_value(unit, *args, **kwargs)
        image.bunit = image.header["bunit"] = _apu_to_headerstr(_to_apu(unit))
        return image
    
    def copy(self):
        """
        This method creates a copy of the original image.
        """
        return copy.deepcopy(self)
    
    def get_xyaxes(self, grid=False, unit=None):
        """
        Obtain the 1-dimensional or 2-dimensional x- and y-axes of the image.
        Parameters:
            grid (bool): True to return the x and y grids 
            unit (str): The angular unit of the positional axes. None to use the default unit (self.unit).
        Returns:
            xaxis (ndarray, if grid=False): the x-axis
            yaxis (ndarray, if grid=False): the y-axis
            xx (ndarray, if grid=True): the x grid
            yy (ndarray, if grid=True): the y grid
        """
        # create axes in the same unit as self.unit
        xaxis = self.dx * (np.arange(self.nx)-self.refnx+1)
        yaxis = self.dy * (np.arange(self.ny)-self.refny+1)
        
        # convert to the specified unit if necessary
        if unit is not None:
            xaxis = u.Quantity(xaxis, self.unit).to_value(unit)
            yaxis = u.Quantity(yaxis, self.unit).to_value(unit)
        
        # make 2D grid if grid is set to True
        if grid:
            xx, yy = np.meshgrid(xaxis, yaxis)
            return xx, yy
        return xaxis, yaxis
    
    def get_vaxis(self, specunit=None):
        """
        To get the vaxis of the data cube.
        Parameters:
            specunit (str): the unit of the spectral axis. 
                            Default is to use the same as the header unit.
        Returns:
            vaxis (ndarray): the spectral axis of the data cube.
        """
        vaxis = np.linspace(self.vrange[0], self.vrange[1], self.nchan)
        if specunit is None:
            if self.specunit == "km/s":
                vaxis = np.round(vaxis, 7)
            return vaxis
        try:
            # attempt direct conversion
            vaxis = u.Quantity(vaxis, self.specunit).to_value(specunit)
        except UnitConversionError:
            # if that fails, try using equivalencies
            equiv = u.doppler_radio(self.restfreq*u.Hz)
            vaxis = u.Quantity(vaxis, self.specunit).to_value(specunit, equivalencies=equiv)
        if _to_apu(specunit).is_equivalent(u.km/u.s):
            round_mask = np.isclose(vaxis, np.round(vaxis, 10))
            vaxis[round_mask] = np.round(vaxis, 10)
            return vaxis
        return vaxis
    
    def conv_specunit(self, specunit, inplace=True):
        """
        Convert the spectral axis into the desired unit.
        """
        image = self if inplace else self.copy()
        vaxis = image.get_vaxis(specunit=specunit)
        image.header["dv"] = vaxis[1]-vaxis[0]
        image.header["vrange"] = vaxis[[0, -1]].tolist()
        image.header["specunit"] = _apu_to_headerstr(_to_apu(specunit))
        image.__updateparams()
        return image

    def rotate(self, angle=0, ccw=True, unit="deg", inplace=False):
        """
        Rotate the image by the specified degree.
        Parameters:
            angle (float): angle by which image will be rotated
            ccw (bool): True to rotate counterclockwise. False to rotate clockwise.
            unit (str): unit of angle
            inplace (bool): whether to replace data of this image with the rotated data.
        Return: 
            The rotated image (Datacube)
        """
        if unit not in ["deg", "degree", "degrees"]:
            angle = u.Quantity(angle, unit).to_value(u.deg)
        if ccw:
            angle = -angle
        
        # rotate data
        newdata = ndimage.rotate(self.data, angle, order=1, axes=(2, 3))
        
        # take the center of the rotated image
        y, x = newdata.shape[2:]
        startx, starty = x//2 - self.nx//2, y//2 - self.ny//2
        newdata = newdata[:, :, starty:starty+self.ny, startx:startx+self.nx]
        
        if inplace:
            self.data = newdata
            return self
        return Datacube(header=self.header, data=newdata.reshape(self.shape))
    
    def immoments(self, moments=[0], vrange=None, chans=None, threshold=None, 
                  vsys=None, keep_nan=True):
        """
        Parameters:
            moments (list[int]): a list of moment maps to be outputted
            vrange (list[float]): a list of [minimum velocity, maximum velocity] in km/s. 
                                  Default is to use the entire velocity range.
            chans (list[float]): a list of [minimum channel, maximum channel] using 1-based indexing.
                                 Default is to use all channels.
            threshold (float): a threshold to be applied to data cube. Default is not to use a threshold.
            vsys (float): the systemic velocity of reference for 'vrange' (signifying velocity offset.)
                          Default is to assume vsys = 0. 
        Returns:
            A list of moment maps (Spatialmap objects).
        -------
        Additional notes:
            This method uses the following CASA definition of moment maps:
               -1 - mean value of the spectrum
                0 - integrated value of the spectrum
                1 - intensity weighted coordinate; traditionally used to get “velocity fields”
                2 - intensity weighted dispersion of the coordinate
                3 - median value of the spectrum
                4 - median coordinate
                5 - standard deviation about the mean of the spectrum
                6 - root mean square of the spectrum
                7 - absolute mean deviation of the spectrum
                8 - maximum value of the spectrum
                9 - coordinate of the maximum value of the spectrum
                10 - minimum value of the spectrum
                11 - coordinate of the minimum value of the spectrum
        """
        # initialize parameters:
        if vrange is None:
            vrange = []
        if chans is None:
            chans = []
            
        vrange = np.array(vrange).flatten()
        # subtract systemic velocity if needed
        if vrange.size != 0:
            if vsys:
                vrange += vsys
        chans = np.array(chans).flatten()
        
        # check if moments exceed 11. 
        if isinstance(moments, (int, float)):
            if moments > 11:
                raise ValueError("Moments > 11 are not defined.")
            elif moments < -1:
                raise ValueError("Moments < -1 are not defined.")
            moments = [moments]
        elif hasattr(moments, "__iter__"):
            if any(moment > 11 for moment in moments):
                raise ValueError("Moments > 11 are not defined.")
            if any(moment < -1 for moment in moments):
                raise ValueError("Moments < -1 are not defined.")
        
        # apply threshold
        if threshold is None:
            data = self.data
        else:
            data = np.where(self.data<threshold, np.nan, self.data)
        
        # get nx and ny 
        nx, ny = self.nx, self.ny
        
        # if 'chans' is given instead of 'vrange', convert it to 'vrange':
        if chans.size > 0 and chans.size % 2 == 0:
            indicies = np.array(chans)-1  # 1-based indexing -> 0-based indexing
            vrange = self.vaxis[indicies] 
            
        # start truncating channels
        vaxis = self.vaxis
        if vrange.size == 2:
            # prevent floating point errors:
            min_vel = vrange[0] - 0.1*self.dv
            max_vel = vrange[1] + 0.1*self.dv
            # create mask:
            vmask = (min_vel<=vaxis)&(vaxis<=max_vel)
            vaxis = vaxis[vmask]
            data = data[:, vmask, :, :]
        elif vrange.size > 2 and vrange.size % 2 == 0:  # even number of vranges
            vmask = np.full(self.vaxis.shape, False)
            for (v1, v2) in np.vstack((vrange[::2], vrange[1::2])).T:
                # prevent floating point errors:
                v1 -= 0.1*self.dv
                v2 += 0.1*self.dv
                # 'concatenate' mask:
                vmask = vmask | ((v1<=vaxis)&(vaxis<=v2))
            vaxis = vaxis[vmask]
            data = data[:, vmask, :, :]

        # parallel processing
        num_moments = len(moments)
        moment_maps = [_get_moment_map(moment, data, vaxis, self.ny, self.nx,
                                       keep_nan, self.bunit, self.specunit, 
                                       self.header) for moment in moments]
            
        # return output
        return moment_maps[0] if len(moment_maps) == 1 else moment_maps
        
    def sum_over_chans(self, vrange=None, chans=None, threshold=None, vsys=None):
        """
        Method to sum over all channels at each pixel.
        Parameters:
            vrange (list[float]): a list of [minimum velocity, maximum velocity] in km/s. 
                                  Default is to use the entire velocity range.
            chans (list[float]): a list of [minimum channel, maximum channel] using 1-based indexing.
                                 Default is to use all channels.
            threshold (float): a threshold to be applied to data cube. Default is not to use a threshold.
            vsys (float): the systemic velocity of reference for 'vrange' (signifying velocity offset.)
                          Default is to assume vsys = 0. 
        Returns:
            sum_img (Spatialmap): a map with pixels being the sum of the entire spectrum 
        """
        # prevent floating point errors:
        sum_img = self.immoments(moments=0, vrange=vrange, chans=chans, 
                                 threshold=threshold, vsys=vsys) / self.dv
        sum_img = (sum_img * _to_apu(self.bunit)).value  # change unit back to unit of this image.
        return sum_img
    
    def imview(self, contourmap=None, title=None, fov=None, ncol=None, nrow=None, 
               cmap="inferno", figsize=(11.69, 8.27), center=None, vrange=None, chans=None, nskip=1, 
               vskip=None, tickscale=None, tickcolor="w", txtcolor='w', crosson=True, crosslw=0.5, 
               crosscolor="w", crosssize=0.3, dpi=400, vmin=None, vmax=None, xlim=None,
               ylim=None, xlabel=None, ylabel=None, xlabelon=True, ylabelon=True, crms=None, 
               clevels=np.arange(3, 21, 3), ccolors="w", clw=0.5, vsys=None, fontsize=12, 
               decimals=2, vlabelon=True, cbarloc="right", cbarwidth="3%", cbarpad=0., 
               cbarlabel=None, cbarticks=None, cbartick_width=None, cbartick_length=3.,
               cbartick_direction="in", cbarlabelon=True, beamon=True, beamcolor="skyblue", 
               beamloc=(0.1225, 0.1225), nancolor="k", labelcolor="k", axiscolor="w", axeslw=0.8, 
               labelsize=10, tickwidth=1., ticksize=3., tickdirection="in", vlabelsize=12, 
               vlabelunit=False, cbaron=True, title_fontsize=14, scale="linear", gamma=1.5, 
               percentile=None, grid=None, savefig=None, plot=True):
        """
        To plot the data cube's channel maps.
        Parameters:
            contourmap (Spatialmap/Datacube): The contour map to be drawn. Default is to not plot contour.
            fov (float): the field of view of the image in the same spaital unit as the data cube.
            ncol (int): the number of columns to be drawn. 
            nrow (int): the number of rows to be drawn. 
                        Default is the minimum rows needed to plot all specified channels.
            cmap (str): the color map of the color image.
            figsize (tuple(float)): the size of the figure
            center (tuple(float)): the center coordinate of the channel maps
            vrange (list(float)): the range of channels to be drawn.
            nskip (float): the channel intervals
            vskip (float): the velocity interval. An alternative to nskip. Default is the same as 'nskip'.
            tickscale (float): the x- and y-axes tick interval. Default is to use matplotlib's default.
            tickcolor (str): the color of the ticks.
            txtcolor (str): the color of the texts plotted on the image.
            crosson (bool): True to include a central cross indicating the protostellar position.
            crosslw (float): the line width of the central cross, if applicable.
            crosscolor (str): the color of the central cross, if applicable
            crosssize (float): a number between 0 and 1. 
                               The central cross, if applicable, will have a length of this parameter times the fov.
            dpi (float): the dots per inch of the image.
            vmin (float): the minimum value of the color map.
            vmax (float): the maximum value of the color map.
            xlabel (str): the label on the xaxis. Default is to use 'Relative RA (unit)'.
            ylabel (str): the label on the yaxis. Default is to use 'Relative Dec (unit)'.
            xlabelon (bool): True to add a label on the xaxis.
            ylabelon (bool): True to add a label on the yaxis.
            crms (float): The rms noise level of the contour map regarded as the base contour level.
                          The default is an estimation of the contour map noise level using sigma clipping.
            clevels (ndarray): the relative contour levels (the values multiplied by crms).
            ccolors (str): the color of the contour image.
            clw (float): the contour line width.
            vsys (float): the systemic velocity. The vaxis will be subtracted by this value.
            fontsize (float): the size of the font.
            decimals (float): the number of decimals shown on the velocity labels.
            vlabelon (bool): True to plot velocity labels.
            cbarloc (str): the location of the color bar.
            cbarwidth (str/float): the width of the color bar.
            cbarpad (float): the distance between the color bar and the image.
            cbarlabel (str): the color bar label.
            cbarlabelon (bool): True to add a color bar label.
            addbeam (bool): True to add an ellipse representing the beam dimensions 
                            in the bottom left corner of the image.
            beamcolor (str): the color of the beam.
            nancolor (str): the color representing the NaN values on the color map.
            labelcolor (str): the color of the label text.
            axiscolor (str): the color of the axes.
            axeslw (str): the line width of the axes.
            labelsize (float): the size of the tick labels and axis labels.
            tickwidth (float): the width of the ticks.
            ticksize (float): the size (length) of the ticks 
            tickdirection (str): the direction fo the ticks.
            vlabelsize (float): the fontsize of the velocity labels.
            vlabelunit (bool): True to add units in the velocity labels.
            cbaron (bool): True to add a color bar to the image.
            savefig (dict): list of keyword arguments to be passed to 'plt.savefig'.
            plot (bool): True to execute 'plt.show()'
            
        Returns:
            The image grid with the channel maps.
        """
        # initialize parameters:
        if cbarticks is None:
            cbarticks = []

        if vsys is None:
            vsys = 0.

        if fov is None:
            fov = self.widestfov

        if fov < 0:
            fov = -fov

        if center is None:
            center = [0., 0.]

        if xlim is None:
            xlim = [center[0]+fov, center[0]-fov]
        else:
            if xlim[0] < xlim[1]:
                xlim[0], xlim[1] = xlim[1], xlim[0]
            center[0] = (xlim[0]+xlim[1])/2

        if ylim is None:
            ylim = [center[1]-fov, center[1]+fov]
        else:
            if ylim[1] < ylim[0]:
                ylim[0], ylim[1] = ylim[1], ylim[0]
            center[1] = (ylim[0]+ylim[1])/2

        if tickscale is not None:
            ticklist = np.arange(0, fov, tickscale) 
            ticklist = np.append(-ticklist, ticklist)

        if xlabel is None:
            xlabel = f"Relative RA ({self.unit})"

        if ylabel is None:
            ylabel = f"Relative Dec ({self.unit})"

        if not isinstance(clevels, np.ndarray):
            clevels = np.array(clevels)

        if cbarlabel is None:
            cbarlabel = "(" + _unit_plt_str(_apu_to_str(_to_apu(self.bunit))) + ")"

        vaxis = self.vaxis - vsys

        if chans is not None:
            chans = np.array(chans).flatten() - 1  # 1-based indexing -> 0-based indexing
            vrange = vaxis[chans]

        if vrange is None:
            vrange = [vaxis.min(), vaxis.max()]
            
        # this prevents floating point error
        velmin, velmax = vrange
        velmin -= 0.1*self.dv
        velmax += 0.1*self.dv
        
        if vskip is None:
            vskip = self.dv*nskip
        else:
            nskip = int(_true_round_whole(vskip/self.dv))
        cmap = copy.deepcopy(mpl.colormaps[cmap]) 
        cmap.set_bad(color=nancolor) 
        if crms is None and contourmap is not None:
            try:
                crms = contourmap.noise()
                bunit = self.bunit.replace(".", " ")
                print(f"Estimated base contour level (rms): {crms:.4e} [{bunit}]")
            except Exception:
                contourmap = None
                print("Failed to estimate RMS noise level of contour map.")
                print("Please specify base contour level using 'crms' parameter.")
                
        data = self.data.value if isinstance(self.data, u.Quantity) else self.data.copy()
           
        # trim data along vaxis for plotting:
        vmask = (velmin <= vaxis) & (vaxis <= velmax)
        trimmed_data = self.data[:, vmask, :, :][:, ::nskip, :, :]
        trimmed_vaxis = vaxis[vmask][::nskip]

        if percentile is not None:
            vmin, vmax = clip_percentile(data=trimmed_data, area=percentile)
            
        # trim data along xyaxes for plotting:
        if xlim != [self.widestfov, -self.widestfov] \
           or ylim != [-self.widestfov, self.widestfov]:
            xmask = (xlim[1]<=self.xaxis) & (self.xaxis<=xlim[0])
            ymask = (ylim[0]<=self.yaxis) & (self.yaxis<=ylim[1])
            trimmed_data = trimmed_data[:, :, :, xmask]
            trimmed_data = trimmed_data[:, :, ymask, :]
        imextent = [xlim[0]-0.5*self.dx, xlim[1]+0.5*self.dx, 
                    ylim[0]-0.5*self.dy, ylim[1]+0.5*self.dy]
        
        # modify contour map to fit the same channels:
        if contourmap is not None:
            if isinstance(contourmap.data, u.Quantity):
                contmap = contourmap.value.copy()
            else:
                contmap = contourmap.copy()
            
            # make the image conditions the same if necessary:
            if contmap.refcoord != self.refcoord:  # same reference coordinates
                contmap = contmap.imshift(self.refcoord, printcc=False)
            if contmap.unit != self.unit:  # same spatial units
                contmap = contmap.conv_unit(self.unit)
            
            contmap_isdatacube = (contmap.header["imagetype"] == "datacube")
            if contmap_isdatacube:
                cvaxis = contmap.vaxis - vsys
            if xlim != [contmap.widestfov, -contmap.widestfov] \
               or ylim != [-contmap.widestfov, contmap.widestfov]:
                cxmask = (xlim[1]<=contmap.xaxis) & (contmap.xaxis<=xlim[0])
                cymask = (ylim[0]<=contmap.yaxis) & (contmap.yaxis<=ylim[1])
                trimmed_cdata = contmap.data[:, :, cymask, :]
                trimmed_cdata = trimmed_cdata[:, :, :, cxmask]
            else:
                trimmed_cdata = contmap.data
            contextent = [xlim[0]-0.5*contmap.dx, xlim[1]+0.5*contmap.dx, 
                          ylim[0]-0.5*contmap.dy, ylim[1]+0.5*contmap.dy]
        
        # figure out the number of images per row/column to plot:
        nchan = trimmed_vaxis.size
        if ncol is None and nrow is None:
            ncol = _get_optimal_columns(nchan)
            nrow = int(np.ceil(nchan/ncol))
        elif nrow is None and ncol is not None:
            nrow = int(np.ceil(nchan/ncol))
        elif nrow is not None and ncol is None:
            ncol = int(np.ceil(nchan/nrow))
            
        # setup matplotlib default parameters:
        params = {'axes.labelsize': fontsize,
                  'axes.titlesize': fontsize,
                  'font.size': fontsize,
                  'legend.fontsize': labelsize,
                  'xtick.labelsize': labelsize,
                  'ytick.labelsize': labelsize,
                  'xtick.top': True,   # draw ticks on the top side
                  'xtick.major.top': True,
                  'figure.figsize': figsize,
                  'figure.dpi': dpi,
                  'font.family': _fontfamily,
                  'mathtext.fontset': _mathtext_fontset,
                  'mathtext.tt': _mathtext_tt,
                  'axes.linewidth': axeslw,
                  'xtick.major.width': tickwidth,
                  'xtick.major.size': ticksize,
                  'xtick.direction': tickdirection,
                  'ytick.major.width': tickwidth,
                  'ytick.major.size': ticksize,
                  'ytick.direction': tickdirection,
                  }
        rcParams.update(params)
        
        # plotting preparation:
        if grid is None:
            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow, ncol),
                             axes_pad=0., share_all=True, cbar_mode='single',
                             cbar_location=cbarloc, cbar_size=cbarwidth,
                             cbar_pad=cbarpad, label_mode='1')

        # start plotting:
        if contourmap is not None and not contmap_isdatacube:
            thiscdata = trimmed_cdata[0, 0]
            
        # assign vmin and vmax -> important to ensure log/gamma scale works properly!
        if vmin is None: 
            vmin = np.nanmin(trimmed_data)
        if vmax is None:
            vmax = np.nanmax(trimmed_data)
            
        is_logscale: bool = (scale.lower() in ("log", "logarithm", "logscale"))
            
        for i in range(nrow*ncol):
            ax = grid[i] # the axes object of this channel
            if i < nchan:
                thisvel = trimmed_vaxis[i]         # the velocity of this channel
                thisdata = trimmed_data[0, i]      # 2d data in this channel

                # plot color map using helper function
                ax, imcolor = _plt_cmap(image_obj=self, 
                                        ax=ax, 
                                        two_dim_data=thisdata, 
                                        imextent=imextent, 
                                        cmap=cmap, 
                                        vmin=vmin, 
                                        vmax=vmax, 
                                        scale=scale, 
                                        gamma=gamma)
                
                # contour image
                if contourmap is not None:
                    if contmap_isdatacube:
                        thiscdata = trimmed_cdata[:, (cvaxis == thisvel), :, :][0, 0]  # the contour data of this channel
                    imcontour = ax.contour(thiscdata, colors=ccolors, origin='lower', 
                                           extent=contextent, levels=crms*clevels, linewidths=clw)
                
                # set ticks
                if tickscale is None:
                    ticklist = ax.get_xticks()
                ax.set_xticks(ticklist)
                ax.set_yticks(-ticklist)
                ax.set_aspect(1)
                ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, 
                               color=tickcolor, labelcolor=labelcolor, pad=9)
                
                # plot channel labels
                if vlabelon:
                    vlabel = f"%.{decimals}f "%thisvel + _unit_plt_str(_apu_to_str(_to_apu(self.specunit))) \
                             if vlabelunit else f"%.{decimals}f"%thisvel
                    ax.text(0.1, 0.9, vlabel, color=txtcolor, size=vlabelsize,
                            ha='left', va='top', transform=ax.transAxes)
                
                # plot central cross
                if crosson:
                    xfov = (xlim[0]-xlim[1])/2
                    yfov = (ylim[1]-ylim[0])/2
                    ax.plot([center[0]-crosssize*xfov, center[0]+crosssize*xfov], [center[1], center[1]], 
                             color=crosscolor, lw=crosslw, zorder=99)   # horizontal line
                    ax.plot([center[0], center[0]], [center[1]-crosssize*yfov, center[1]+crosssize*yfov], 
                            color=crosscolor, lw=crosslw, zorder=99)    # vertical line
                
                # plot the axis borders
                ax.spines["bottom"].set_color(axiscolor)
                ax.spines["top"].set_color(axiscolor)
                ax.spines["left"].set_color(axiscolor)
                ax.spines["right"].set_color(axiscolor)
            else:
                # don't show axis borders if finished plotting.
                ax.spines["bottom"].set_color('none')
                ax.spines["top"].set_color('none')
                ax.spines["left"].set_color('none')
                ax.spines["right"].set_color('none')
                ax.axis('off')
            # set field of view
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
                
        # axis labels
        if xlabelon or ylabelon or beamon:
            bottomleft_ax = grid[(nrow-1)*ncol]
        if xlabelon:
            bottomleft_ax.set_xlabel(xlabel)
            bottomleft_ax.xaxis.label.set_color(labelcolor)
        if ylabelon:
            bottomleft_ax.set_ylabel(ylabel)
            bottomleft_ax.yaxis.label.set_color(labelcolor)
        
        # add beam
        if beamon:
            bottomleft_ax = self._addbeam(bottomleft_ax, xlim=xlim, ylim=ylim, filled=True,
                                           beamcolor=beamcolor, beamloc=beamloc)

        # colorbar 
        if cbaron:
            cax = grid.cbar_axes[0]
            cbar = plt.colorbar(imcolor, cax=cax)
            cax.toggle_label(True)
            cbar.ax.spines["bottom"].set_color(axiscolor)  
            cbar.ax.spines["top"].set_color(axiscolor)
            cbar.ax.spines["left"].set_color(axiscolor)
            cbar.ax.spines["right"].set_color(axiscolor)
            if cbarlabelon:
                cbar.ax.set_ylabel(cbarlabel, color=labelcolor)
            
            # change color bar ticks:
            if len(cbarticks) != 0:
                cbar.set_ticks(cbarticks)
            elif is_logscale:
                cbarticks = np.linspace(vmin, vmax, 7)[1:-1]
                cbar.set_ticks(cbarticks)
                
            # change tick labels if logscale:
            if is_logscale:
                labels = (f"%.{decimals}f"%label for label in cbarticks)  # generator object
                labels = [label[:-1] if label.endswith(".") else label for label in labels]
                cbar.set_ticklabels(labels)
            
            # tick parameters
            cbar.ax.tick_params(labelsize=labelsize, width=cbartick_width, 
                                length=cbartick_length, direction=cbartick_direction, 
                                color=tickcolor)
            
            # disable minor ticks
            cbar.ax.minorticks_off()
        
        # these parameters are for looping if annotations are to be added in other methods
        self.__pltnax = nrow*ncol
        self.__pltnchan = nchan
        
        # add title at the center
        if title is not None:
            if ncol % 2 == 0:  # even number of columns
                middle_right_ax = grid[ncol//2]
                middle_right_ax.set_title(title, fontsize=title_fontsize, x=0.)
            else:  # odd number of columns
                middle_ax = grid[ncol//2]
                middle_ax.set_title(title, fontsize=title_fontsize)
        
        # save figure if parameters were specified
        if savefig:
            plt.savefig(**savefig)
        
        # show image
        if plot:
            plt.show()
            
        return grid
    
    def _addbeam(self, ax, xlim, ylim, beamcolor, filled=True, beamloc=(0.1225, 0.1225)):
        """
        This method adds an ellipse representing the beam size to the specified ax.
        """
        # coordinate of ellipse center 
        coords = _unnormalize_location(beamloc, xlim, ylim)

        if np.any(np.isnan(self.beam)):
            warnings.warn("Beam cannot be plotted as it is not available in the header.")
            return ax

        # beam dimensions
        bmaj, bmin, bpa = self.beam
        
        # add patch to ax
        if filled:
            beam = patches.Ellipse(xy=coords, width=bmin, height=bmaj, fc=beamcolor,
                                   ec=beamcolor, angle=-bpa, alpha=1, zorder=10)
        else:
            beam = patches.Ellipse(xy=coords, width=bmin, height=bmaj, fc='None',
                                    ec=beamcolor, linestyle="solid", 
                                    angle=-bpa, alpha=1, zorder=10)
        ax.add_patch(beam)
        return ax
        
    def trim(self, template=None, xlim=None, ylim=None, vlim=None, vsys=None, 
             nskip=None, vskip=None, inplace=False):
        """
        Trim the image data based on specified x and y axis limits.

        Parameters:
        template (Spatialmap/Channelmap): The xlim and ylim will be assigned as the the min/max values 
                                          of the axes of this image.
        xlim (iterable of length 2, optional): The limits for the x-axis to trim the data.
                                               If None, no trimming is performed on the x-axis.
        ylim (iterable of length 2, optional): The limits for the y-axis to trim the data.
                                               If None, no trimming is performed on the y-axis.
        inplace (bool, optional): If True, the trimming is performed in place and the original 
                                  object is modified. If False, a copy of the object is trimmed 
                                  and returned. Default is False.

        Returns:
        image: The trimmed image object. If `inplace` is True, the original object is returned 
               after modification. If `inplace` is False, a new object with the trimmed data is returned.

        Raises:
        ValueError: If any of `xlim` or `ylim` are provided and are not iterables of length 2.

        Notes:
        The method uses the `_trim_data` function to perform the actual data trimming. The reference 
        coordinates are updated based on the new trimmed axes.
        """
        if template is not None:
            xlim = template.xaxis.min(), template.xaxis.max()
            ylim = template.yaxis.min(), template.yaxis.max()

        image = self if inplace else self.copy()

        new_data, new_vaxis, new_yaxis, new_xaxis = _trim_data(self.data, 
                                                               yaxis=self.yaxis,
                                                               xaxis=self.xaxis,
                                                               vaxis=self.vaxis,
                                                               dy=self.dy,
                                                               dx=self.dx,
                                                               dv=self.dv,
                                                               ylim=ylim,
                                                               xlim=xlim,
                                                               vlim=vlim,
                                                               nskip=nskip,
                                                               vskip=vskip,
                                                               vsys=vsys)

        # get new parameters:
        new_shape = new_data.shape
        new_nx = new_xaxis.size 
        new_ny = new_yaxis.size 
        new_nchan = new_vaxis.size
        new_vrange = [new_vaxis.min(), new_vaxis.max()]
        new_refnx = new_nx//2 + 1
        new_refny = new_ny//2 + 1
        new_dv = new_vaxis[1] - new_vaxis[0]

        # calculate new reference coordinate:
        org_ref_skycoord = SkyCoord(self.refcoord)
        org_refcoord_x = org_ref_skycoord.ra
        org_refcoord_y = org_ref_skycoord.dec

        start_x = org_refcoord_x + u.Quantity(new_xaxis.min(), self.unit)
        end_x = org_refcoord_x + u.Quantity(new_xaxis.max(), self.unit)
        new_x = (start_x+end_x+self.dx*u.Unit(self.unit))/2

        start_y = org_refcoord_y + u.Quantity(new_yaxis.min(), self.unit)
        end_y = org_refcoord_y + u.Quantity(new_yaxis.max(), self.unit)
        new_y = (start_y+end_y+self.dy*u.Unit(self.unit))/2

        new_refcoord = SkyCoord(ra=new_x, dec=new_y).to_string("hmsdms")

        # update object:
        image.data = new_data
        image.overwrite_header(shape=new_shape, nx=new_nx, ny=new_ny, 
                               refnx=new_refnx, refny=new_refny, 
                               refcoord=new_refcoord, nchan=new_nchan,
                               vrange=new_vrange, dv=new_dv)
        return image
        
    def conv_bunit(self, bunit, inplace=True):
        """
        This method converts the brightness unit of the image into the desired unit.
        Parameters:
            bunit (str): the new unit.
            inplace (bool): True to update the current data with the new unit. 
                            False to create a new image having data with the new unit.
        Returns:
            A new image with the desired unit.
        """
        # string to astropy units 
        bunit = _to_apu(bunit)
        oldunit = _to_apu(self.bunit)
        
        # equivalencies
        equiv_bt = u.equivalencies.brightness_temperature(frequency=self.restfreq*u.Hz, 
                                                          beam_area=self.beam_area())
        equiv_pix = u.pixel_scale(u.Quantity(np.abs(self.dx), self.unit)**2/u.pixel)
        equiv_ba = u.beam_angular_area(self.beam_area())
        equiv = [equiv_bt, equiv_pix, equiv_ba]

        # factors
        factor_bt = (1*u.Jy/u.beam).to(u.K, equivalencies=equiv_bt) / (u.Jy/u.beam)
        factor_pix = (1*u.pixel).to(u.rad**2, equivalencies=equiv_pix) / (u.pixel)
        factor_ba = (1*u.beam).to(u.rad**2, equivalencies=equiv_ba) / (u.beam)
        factor_pix2bm = factor_pix / factor_ba
        factor_Jypix2K = factor_pix2bm / factor_bt
        factor_Jysr2K = factor_bt*factor_ba
        factors = [factor_bt, factor_pix, factor_ba, factor_pix2bm, 
                   factor_Jypix2K, factor_Jysr2K]
        
        # convert
        if isinstance(self.data, u.Quantity):
            olddata = self.data.copy()
        else:
            olddata = u.Quantity(self.data, oldunit)
        newdata = _convert_Bunit(olddata, newunit=bunit, 
                                 equivalencies=equiv, factors=factors)
        
        if newdata is None:
            raise UnitConversionError(f"Failure to convert intensity unit to {_apu_to_headerstr(bunit)}")

        # return and set values
        if not isinstance(self.data, u.Quantity):
            newdata = newdata.value
            
        newimage = self if inplace else self.copy()
        newimage.data = newdata
        newimage.header["bunit"] = _apu_to_headerstr(bunit)
        newimage.__updateparams()
        return newimage
    
    def extract_channels(self, chans=None, vrange=None):
        """
        Extract specific channel maps from the Datacube.
        Parameters:
            chans (list(int)): the range of channels to be extracted.
                               This parameter uses 1-based indexing following the CASA convention.
            vrange (list(float)): the velocity range of the channels to be extracted.
        Returns:
            list(Spatialmap): a list of extracted channel maps.
        """
        # initialize parameters
        maps = []
        vaxis = self.vaxis.copy()
        
        # convert 1-based indexing to 0-based indexing used in Python:
        if chans is not None and hasattr(chans, '__iter__'):
            chans = np.array(chans) - 1  
            
        # convert between vrange and nchans
        if vrange is not None:
            if hasattr(chans, '__iter__') and len(vrange) == 2:
                idx = np.where((vrange[0]<=vaxis)&(vaxis<=vrange[1]))[0]
                chans = [idx[0], idx[-1]]
            elif isinstance(vrange, (int, float)) or (hasattr(chans, '__iter__') and len(vrange)==1):
                if isinstance(vrange, (list, np.ndarray, float)):
                    vrange = vrange[0]
                chans = int(np.where(vaxis == vrange)[0][0])
        if chans is None and vrange is None:
            chans = [0, vaxis.size-1]
        elif isinstance(chans, (int, float)) or (chans is not None and len(chans) == 1):
            if hasattr(chans, '__iter__'):
                chans = chans[0]
            chans = [chans, chans] 
            
        # create new header info
        new_header = copy.deepcopy(self.header)
        new_header["dv"] = None
        new_header["nchan"] = 1
        new_header["shape"] = (1, 1, self.ny, self.nx)
        new_header["imagetype"] = "spatialmap"
        
        # start extracting and adding maps
        for i in range(chans[0], chans[1]+1, 1):
            new_header["vrange"] = [vaxis[i], vaxis[i]]
            maps.append(Spatialmap(header=copy.deepcopy(new_header), 
                                   data=self.data[0, i].reshape(1, 1, self.ny, self.nx)))
            
        # return maps as list or as inidividual object.
        if len(maps) == 1:
            return maps[0]
        return maps
    
    def set_threshold(self, threshold=None, minimum=True, inplace=True):
        """
        To mask the intensities above or below this threshold.
        Parameters:
            threshold (float): the value of the threshold. None to use three times the rms noise level.
            minimum (bool): True to remove intensities below the threshold. False to remove intensities above the threshold.
            inplace (bool): True to modify the data in-place. False to return a new image.
        Returns:
            Image with the masked data.
        """
        if threshold is None:
            threshold = 3*self.noise()
        if inplace:
            if minimum:
                self.data = np.where(self.data<threshold, np.nan, self.data)
            else:
                self.data = np.where(self.data<threshold, self.data, np.nan)
            return self
        if minimum:
            return Datacube(header=self.header, data=np.where(self.data<threshold, np.nan, self.data))
        return Datacube(header=self.header, data=np.where(self.data<threshold, self.data, np.nan))
    
    def beam_area(self, unit=None):
        """
        To calculate the beam area of the image.
        Parameters:
            unit (float): the unit of the beam area to be returned. 
                          Default (None) is to use same unit as the positional axes.
        Returns:
            The beam area.
        """
        bmaj = u.Quantity(self.bmaj, self.unit)
        bmin = u.Quantity(self.bmin, self.unit)
        area = np.pi*bmaj*bmin/(4*np.log(2))
        if unit is not None:
            area = area.to_value(unit)
        return area
    
    def pixel_area(self, unit=None):
        """
        To calculate the pixel area of the image.
        Parameters:
            unit (float): the unit of the pixel area to be returned.
                          Default (None) is to use the same unit as the positional axes.
        Returns:
            The pixel area.
        """
        width = u.Quantity(np.abs(self.dx), self.unit)
        height = u.Quantity(np.abs(self.dy), self.unit)
        area = width*height
        if unit is not None:
            area = area.to_value(unit)
        return area
    
    def conv_unit(self, unit, distance=None, inplace=True):
        """
        This method converts the axis unit of the image into the desired unit.
        Parameters:
            unit (str): the new axis unit.
            inplace (bool): True to update the current axes with the new unit. 
                            False to create a new image having axes with the new unit.
        Returns:
            A new image with axes of the desired unit.
        """
        newbeam = (u.Quantity(self.beam[0], self.unit).to_value(unit),
                   u.Quantity(self.beam[0], self.unit).to_value(unit),
                   self.bpa)
        if inplace:
            self.header["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
            self.header["dy"] = u.Quantity(self.dy, self.unit).to_value(unit)
            self.header["beam"] = newbeam
            self.header["unit"] = _apu_to_headerstr(_to_apu(unit))
            self.__updateparams()
            return self
        newheader = copy.deepcopy(self.header)
        newheader["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
        newheader["dy"] = u.Quantity(self.dy, self.unit).to_value(unit)
        newheader["beam"] = newbeam
        newheader["unit"] = _apu_to_headerstr(_to_apu(unit))
        return Datacube(header=newheader, data=self.data)
    
    def set_data(self, data, inplace=False):
        """
        This method allows users to assign a new dataset to the image.
        Parameters:
            data (ndarray): the new data to be set
            inplace (bool): True to modify the data of this image. False to only return a copy.
        Return: 
            An image with the specified dataset.
        """
        if data.shape != self.shape:
            data = data.reshape(self.shape)
        if inplace:
            self.data = data
            return self
        newimg = self.copy()
        newimg.data = data
        return newimg
    
    def noise(self, sigma=3., plthist=False, shownoise=False, printstats=False, bins='auto', 
              gaussfit=False, curvecolor="crimson", curvelw=0.6, fixmean=True, histtype='step', 
              linewidth=0.6, returnmask=False, **kwargs):
        """
        This method estimates the 1 sigma background noise level using sigma clipping.
        Parameters:
            sigma (float): sigma parameter of sigma clip
            plthist (bool): True to show background noise distribution as histogram (useful for checking if Gaussian)
            shownoise (bool): True to background noise spatial distribution
            printstats (bool): True to print information regarding noise statistics
            bins (int): The bin size of the histogram. Applicable if plthist or gaussfit
            gaussfit (bool): True to perform 1D gaussian fitting on noise distribution
            curvecolor (str): the color of the best-fit curve to be plotted, if applicable
            curvelw (float): the line width of the best-fit curve to be plotted, if applicable
            fixmean (bool): True to fix fitting parameter of mean to 0, if guassfit 
            histtype (str): the type of histogram
            linewidth (float): the line width of the histogram line borders
            returnmask (bool): True to return the mask (pixels that are considered 'noise')
        Returns:
            rms (float): the rms noise level
            mask (ndarray): the noise mask (pixels that are considered noise), if returnmask=True
        """
        bunit = self.bunit.replace(".", " ")
        
        clipped_data = sigma_clip(self.data, sigma=sigma, maxiters=10000, masked=False, axis=(1, 2, 3))
        mean = np.nanmean(clipped_data)
        rms = np.sqrt(np.nanmean(clipped_data**2))
        std = np.nanstd(clipped_data)
    
        if not rms:
            raise Exception("Sigma clipping failed. It is likely that most noise has been masked.")
        
        if printstats:
            print(15*"#" + "Noise Statistics" + 15*"#")
            print(f"Mean: {mean:.6e} [{bunit}]")
            print(f"RMS: {rms:.6e} [{bunit}]")
            print(f"StdDev: {std:.6e} [{bunit}]")
            print(46*"#")
            print()
        
        if gaussfit:
            if fixmean:
                def gauss(x, sigma, amp):
                    return amp*np.exp(-x**2/(2*sigma**2))
            else:
                def gauss(x, sigma, amp, mean):
                    return amp*np.exp(-(x-mean)**2/(2*sigma**2))
            flatdata = clipped_data[~np.isnan(clipped_data)]
            ydata, edges = np.histogram(flatdata, bins=bins)
            xdata = (edges[1:] + edges[:-1]) / 2  # set as midpoint of the bins
            p0 = [std, np.max(ydata)] if fixmean else [std, np.max(ydata), mean]
            popt, pcov = curve_fit(f=gauss, xdata=xdata, ydata=ydata, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            fitx = np.linspace(xdata.min(), xdata.max(), 1000)
            fity = gauss(fitx, popt[0], popt[1]) if fixmean else gauss(fitx, popt[0], popt[1], popt[2])
            
            print(15*"#" + "Gaussian Fitting Results" + 15*"#")
            if not fixmean:
                print(f"Mean: {popt[2]:.4e} +/- {perr[2]:.4e} [{bunit}]")
            print(f"StdDev: {popt[0]:.4e} +/- {perr[0]:.4e} [{bunit}]")
            print(f"Amplitude: {np.round(popt[1])} +/- {np.round(perr[1])} [pix]")
            print(54*"#")
            print()
            
        if plthist:
            q = _to_apu(self.bunit)
            flatdata = clipped_data[~np.isnan(clipped_data)]
            ax = plt_1ddata(flatdata, hist=True, 
                            xlabel=f"Intensity ({q:latex_inline})", ylabel="Pixels",
                            bins=bins, plot=False, xlim=[flatdata.min(), flatdata.max()], 
                            linewidth=linewidth, histtype=histtype, **kwargs)
            if gaussfit:
                ax.plot(fitx, fity, color=curvecolor, lw=curvelw)
                
        if shownoise:
            bg_image = self.set_data(clipped_data, inplace=False)
            bg_image.imview(title="Background Noise Distribution", **kwargs)
            
        if plthist or shownoise:
            plt.show()
        
        if returnmask:
            mask = ~np.isnan(clipped_data)
            return rms, mask
        return rms
    
    def view_region(self, region, color="skyblue", lw=1., ls="--", plot=True, **kwargs):
        """
        This method allows users to plot the specified region on the image.
        Parameters:
            region (Region): the region to be plotted
            color (str): the color of the line representing the region
            lw (float): the width of the line representing the region
            ls (str): the type of the line representing the region
            plot (bool): True to show the plot. False to only return the plot.
        Returns:
            The channel map with the annotated region.
        """
        grid = self.imview(plot=False, **kwargs)

        if isinstance(region, Region):
            region = self.__readregion(region)
            if region.shape == "circle":
                xsmooth = np.linspace(region.center[0]-region.radius, region.center[0]+region.radius, 1000)
                inside_circle = (np.square(xsmooth-region.center[0]) <= region.radius**2)
                ysmooth = np.sqrt(region.radius**2-np.square(xsmooth[inside_circle]-region.center[0]))
                
                # Plot the upper and lower semicircles
                for i in range(self.__pltnchan):
                    ax = grid[i]
                    ax.plot(xsmooth[inside_circle], region.center[1]+ysmooth, color=color, lw=lw, linestyle=ls)
                    ax.plot(xsmooth[inside_circle], region.center[1]-ysmooth, color=color, lw=lw, linestyle=ls)
            elif region.shape == "ellipse":
                theta = np.linspace(0, 2*np.pi, 1000)
                angle = np.deg2rad(-region.pa-270)

                # Ellipse equation before rotation
                x_ellipse = region.center[0]+region.semimajor*np.cos(theta)
                y_ellipse = region.center[1]+region.semiminor*np.sin(theta)

                # Applying rotation for position angle
                x_rotated = region.center[0]+(x_ellipse-region.center[0])*np.cos(angle)-(y_ellipse-region.center[1])*np.sin(angle)
                y_rotated = region.center[1]+(x_ellipse-region.center[0])*np.sin(angle)+(y_ellipse-region.center[1])*np.cos(angle)
                
                # Plot the ellipse
                for i in range(self.__pltnchan):
                    grid[i].plot(x_rotated, y_rotated, color=color, lw=lw, linestyle=ls)
            elif region.shape == "line":
                # Plot line
                for i in range(self.__pltnchan):
                    grid[i].plot([region.start[0], region.end[0]], [region.start[1], region.end[1]], 
                                 color=color, lw=lw, linestyle=ls)
            elif region.shape == "box":
                # make patch
                pa = region.pa
                pa_rad = np.deg2rad(pa)
                center = region.center
                width, height = region.width, region.height
                dx, dy = -width/2, -height/2
                x = center[0] + (dx*np.cos(pa_rad) + dy*np.sin(pa_rad))
                y = center[1] - (dx*np.sin(pa_rad) - dy*np.cos(pa_rad))
                
                # add patch on all plots
                for i in range(self.__pltnchan):
                    rect_patch = patches.Rectangle((x, y), width, height, angle=-region.pa,
                                               linewidth=lw, linestyle=ls, edgecolor=color, 
                                               facecolor='none')
                    grid[i].add_patch(rect_patch)
        elif hasattr(region, "__iter__"):
            region_list = region 
            for region in region_list:
                region = self.__readregion(region)
                if region.shape == "circle":
                    xsmooth = np.linspace(region.center[0]-region.radius, region.center[0]+region.radius, 1000)
                    inside_circle = (np.square(xsmooth-region.center[0]) <= region.radius**2)
                    ysmooth = np.sqrt(region.radius**2-np.square(xsmooth[inside_circle]-region.center[0]))
                    
                    # Plot the upper and lower semicircles
                    for i in range(self.__pltnchan):
                        ax = grid[i]
                        ax.plot(xsmooth[inside_circle], region.center[1]+ysmooth, color=color, lw=lw, linestyle=ls)
                        ax.plot(xsmooth[inside_circle], region.center[1]-ysmooth, color=color, lw=lw, linestyle=ls)
                elif region.shape == "ellipse":
                    theta = np.linspace(0, 2*np.pi, 1000)
                    angle = np.deg2rad(-region.pa-270)

                    # Ellipse equation before rotation
                    x_ellipse = region.center[0]+region.semimajor*np.cos(theta)
                    y_ellipse = region.center[1]+region.semiminor*np.sin(theta)

                    # Applying rotation for position angle
                    x_rotated = region.center[0]+(x_ellipse-region.center[0])*np.cos(angle)-(y_ellipse-region.center[1])*np.sin(angle)
                    y_rotated = region.center[1]+(x_ellipse-region.center[0])*np.sin(angle)+(y_ellipse-region.center[1])*np.cos(angle)
                    
                    # Plot the ellipse
                    for i in range(self.__pltnchan):
                        grid[i].plot(x_rotated, y_rotated, color=color, lw=lw, linestyle=ls)
                elif region.shape == "line":
                    # Plot line
                    for i in range(self.__pltnchan):
                        grid[i].plot([region.start[0], region.end[0]], [region.start[1], region.end[1]], 
                                     color=color, lw=lw, linestyle=ls)
                elif region.shape == "box":
                    # make patch
                    pa = region.pa
                    pa_rad = np.deg2rad(pa)
                    center = region.center
                    width, height = region.width, region.height
                    dx, dy = -width/2, -height/2
                    x = center[0] + (dx*np.cos(pa_rad) + dy*np.sin(pa_rad))
                    y = center[1] - (dx*np.sin(pa_rad) - dy*np.cos(pa_rad))
                    
                    # add patch on all plots
                    for i in range(self.__pltnchan):
                        rect_patch = patches.Rectangle((x, y), width, height, angle=-region.pa,
                                                   linewidth=lw, linestyle=ls, edgecolor=color, 
                                                   facecolor='none')
                        grid[i].add_patch(rect_patch)

        if plot:
            plt.show()

        return grid
    
    def mask_region(self, region, vrange=None, exclude=False, preview=True, inplace=False, **kwargs):
        """
        Mask the specified region.
        Parameters:
            region (Region): the region object inside/outside of which the image will be masked.
            vrange (list): the [min, max] velocity channels to be masked.
            exclude (bool): True to exclude pixels outside the region. False to exclude those inside.
            preview (bool): True to visualize masked image. 
            inplace (bool): True to modify the image in-place. False to return a new image.
        Returns:
            The masked image.
        """
        if vrange is None:
            vrange = []
        if region.shape == "line":
            raise Exception(f"Region shape cannot be '{region.shape}' ")
        region = self.__readregion(region)
        data = self.data.copy()
        xx, yy = self.get_xyaxes(grid=True)
        if region.shape == "circle":
            radius = region.radius
            x, y = region.center
            mask = ((xx-x)**2+(yy-y)**2 <= radius**2)
        elif region.shape == "ellipse":
            angle = np.deg2rad(region.pa+270)
            x, y = region.center
            a = region.semimajor 
            b = region.semiminor
            xx_prime = (xx-x)*np.cos(angle)-(yy-y)*np.sin(angle)
            yy_prime = (xx-x)*np.sin(angle)+(yy-y)*np.cos(angle)
            mask = np.square(xx_prime/a) + np.square(yy_prime/b) <= 1
        elif region.shape == "box":
            # get width and height
            width, height = region.width, region.height
            
            # shift center
            xx -= region.center[0]
            yy -= region.center[1]

            # rotate
            pa_rad = np.deg2rad(region.pa)
            xx_rot = xx*np.cos(pa_rad)-yy*np.sin(pa_rad)
            yy_rot = xx*np.sin(pa_rad)+yy*np.cos(pa_rad)

            # create masks
            xmask = (-width/2 <= xx_rot) & (xx_rot <= width/2)
            ymask = (-height/2 <= yy_rot) & (yy_rot <= height/2)
            mask = xmask & ymask
        if exclude:
            mask = ~mask
        mask = np.broadcast_to(mask, data.shape[1:])
        if len(vrange) == 2:
            vmask = (vrange[0] <= self.vaxis) & (self.vaxis <= vrange[1])
            vmask = np.broadcast_to(vmask[:, np.newaxis, np.newaxis], 
                                    (self.nchan, self.ny, self.nx)) 
            mask = mask | ~vmask
        masked_data = np.where(mask, data[0], np.nan)
        newshape =  (1, masked_data.shape[0],  masked_data.shape[1], masked_data.shape[2])
        masked_data = masked_data.reshape(newshape)
        masked_image = Datacube(header=self.header, data=masked_data)
        if preview: 
            masked_image.view_region(region, **kwargs)
        if inplace:
            self.data = masked_data
        return masked_image
    
    def __readregion(self, region):
        region = copy.deepcopy(region)
        center = region.header["center"]
        isJ2000 = (isinstance(center, str) and not center.isnumeric()) \
                    or (len(center)==2 and isinstance(center[0], str) and isinstance(center[1], str)) 
        if isJ2000:
            newcenter = _icrs2relative(center, ref=self.refcoord, unit=region.header["unit"])
            region.center = region.header["center"] = (newcenter[0].value, newcenter[1].value)
            if region.shape == "line":
                newstart = _icrs2relative(region.start, ref=self.refcoord, unit=region.header["unit"])
                newend = _icrs2relative(region.end, ref=self.refcoord, unit=region.header["unit"])
                region.start = region.header["start"] = (newstart[0].value, newstart[1].value)
                region.end = region.header["end"] = (newend[0].value, newend[1].value)
                region.length = region.header["length"] = u.Quantity(region.length, region.header["unit"]).to_value(self.unit)
            elif region.shape == "ellipse":
                newsemimajor = u.Quantity(region.semimajor, region.header["unit"]).to_value(self.unit)
                newsemiminor = u.Quantity(region.semiminor, region.header["unit"]).to_value(self.unit)
                region.semimajor = region.header["semimajor"] = newsemimajor
                region.semiminor = region.header["semiminor"] = newsemiminor
            elif region.shape == "circle":
                newradius = u.Quantity(region.radius, region.header["unit"]).to_value(self.unit)
                region.radius = region.semimajor = region.semiminor = region.header["radius"] = newradius
            elif region.shape == "box":
                height = u.Quantity(region.height, region.header["unit"]).to_value(self.unit)
                region.height = region.header["height"] = height
                width = u.Quantity(region.width, region.header["unit"]).to_value(self.unit)
                region.width = region.header["width"] = width
        elif not region.header["relative"]:
            refx, refy = _icrs2relative(self.refcoord, unit=self.unit)
            centerx, centery = u.Quantity(center[0], region.header["unit"]), u.Quantity(center[1], region.header["unit"])
            newcenter = u.Quantity(centerx-refx, self.unit).value, u.Quantity(centery-refy, self.unit).value
            region.center = region.header["center"] = newcenter
            if region.shape == "ellipse":
                newsemimajor = u.Quantity(region.semimajor, region.header["unit"]).to_value(self.unit)
                newsemiminor = u.Quantity(region.semiminor, region.header["unit"]).to_value(self.unit)
                region.semimajor = region.header["semimajor"] = newsemimajor
                region.semiminor = region.header["semiminor"] = newsemiminor
            elif region.shape == "circle":
                newradius = u.Quantity(region.radius, region.header["unit"]).to_value(self.unit)
                region.radius = region.semimajor = region.semiminor = region.header["radius"] = newradius
            elif region.shape == "line":
                start, end = region.start, region.end
                startx, starty = u.Quantity(start[0], region.header["unit"]), u.Quantity(start[1], region.header["unit"])
                newstart = u.Quantity(startx-refx, self.unit).value, u.Quantity(starty-refy, self.unit).value
                region.start = region.header["start"] = newstart
                endx, endy = u.Quantity(end[0], region.header["unit"]), u.Quantity(end[1], region.header["unit"])
                newend = u.Quantity(endx-refx, self.unit).value, u.Quantity(endy-refy, self.unit).value
                region.end = region.header["end"] = newend
                region.length = region.header["length"] = u.Quantity(region.length, region.header["unit"]).to_value(self.unit)
            elif region.shape == "box":
                height = u.Quantity(region.height, region.header["unit"]).to_value(self.unit)
                region.height = region.header["height"] = height
                width = u.Quantity(region.width, region.header["unit"]).to_value(self.unit)
                region.width = region.header["width"] = width
        region.relative = region.header["relative"] = True
        return region
        
    def pvextractor(self, region, vrange=None, width=1, preview=True, **kwargs):
        """
        This method extracts the pv diagram along the specified pv slice.
        Parameters:
            region (Region): the region object that is the pv slice.
            vrange (list): the range of radial velocities to be extracted.
            width (int): Width of slice for averaging pixels perpendicular to the slice. 
                         Must be an odd positive integer or valid quantity.
            preview (bool): True to preview the resulting image.
        Returns:
            The extracted PVdiagram object.
        """
        # get vaxis
        vaxis = self.vaxis.copy()
        
        # get xaxis
        xaxis = self.xaxis.copy()
        
        # raise exception if width is an even number of pixels.
        if width % 2 == 0:
            raise ValueError("The parameter 'width' must be an odd positive integer.")
        if not isinstance(width, int):
            width = int(width)
            
        # initialize parameters
        vrange = self.vaxis[[0, -1]] if vrange is None else vrange
        region = self.__readregion(region)
        center, pa, length = region.center, region.pa, region.length
        
        # shift and then rotate image
        pa_prime = 90.-pa
        shifted_img = self.imshift(center, inplace=False, printcc=False) if center != (0, 0) else self.copy()
        rotated_img = shifted_img.rotate(angle=-pa_prime, ccw=False, inplace=False) if pa_prime != 0 else shifted_img
        
        # trim data
        xmask = (-length/2 <= self.xaxis) & (self.xaxis <= length/2)
        vmask = (vrange[0] <= self.vaxis) & (self.vaxis <= vrange[1])
        pv_xaxis = self.xaxis[xmask]
        pv_vaxis = self.vaxis[vmask]
        pv_data = rotated_img.data[:, :, :, xmask][:, vmask, :, :]
        if width == 1:
            ymask = (self.yaxis == np.abs(self.yaxis).min())
            pv_data = pv_data[:, :, ymask, :]
        else:
            add_idx = int(width//2)
            idx = np.where(self.yaxis == np.abs(self.xaxis).min())[0][0]
            yidx1, yidx2 = idx-add_idx, idx+add_idx+1  # plus 1 to avoid OBOB
            pv_data = pv_data[:, :, yidx1:yidx2, :]
            pv_data = np.nanmean(pv_data, axis=2)
        pv_data = np.fliplr(pv_data.reshape(pv_vaxis.size, pv_xaxis.size))
        pv_data = pv_data[None, :, :]

        # export as pv data
        newheader = copy.deepcopy(self.header)
        newheader["shape"] = pv_data.shape
        newheader["imagetype"] = "pvdiagram"
        newheader["vrange"] = [pv_vaxis.min(), pv_vaxis.max()]
        newheader["dv"] = np.round(pv_vaxis[1]-pv_vaxis[0], 7)
        newheader["nchan"] = pv_vaxis.size
        newheader["dx"] = np.round(pv_xaxis[0]-pv_xaxis[1], 7)
        newheader["nx"] = pv_xaxis.size
        newheader["refnx"] = pv_xaxis.size//2 + 1
        newheader["dy"] = None
        newheader["ny"] = None
        newheader["refny"] = None
        newheader["refcoord"] = _relative2icrs(center, ref=self.refcoord, unit=self.unit)
        
        # new image
        pv = PVdiagram(header=newheader, data=pv_data)
        pv.pa = pa
        
        if preview:
            pv.imview(**kwargs)
        return pv
    
    def get_spectrum(self, region=None, vlim=None, ylabel=None, yunit=None, 
                     xlabel=None, xunit=None, preview=False, **kwargs):
        """
        Get the spectral profile of the data cube.
        Parameters:
            region (Region): the region in which pixels will be averaged in each channel.
            xlabel (str): the label on the xaxis. Default is to use r'Radio Velocity (km/s)'
            ylabel (str): the label on the yaxis. Default is to use f'Intensity (unit)'
            returndata (bool): True to return the generated data. False to return the plot.
        Returns:
            vaxis (ndarray, if returndata=True): the velocity axis
            intensity (ndarray, if returndata=True): the intensity data at each velocity
            ax (matplotlib.axes.Axes, if returndata=False): the generated plot
        """
        # initialize parameters
        if xlabel is None:
            xlabel = r"Radio Velocity"
        if xunit is None:
            xunit = self.specunit
        if ylabel is None:
            ylabel = "Intensity"
        if yunit is None:
            yunit = self.bunit

        # trim image
        trimmed_image = self if region is None else self.mask_region(region, preview=False)

        # if vlim is not None:
        if vlim is None:
            vaxis = self.vaxis
            intensity = np.nanmean(trimmed_image.data, axis=(0, 2, 3))
        else:
            if not hasattr(vlim, "__iter__") or len(vlim) != 2:
                raise ValueError("'vlim' must be an iterable of length 2")
            minvel = min(vlim) - 0.1 * self.dv
            maxvel = max(vlim) + 0.1 * self.dv
            vaxis = self.vaxis
            vmask = (minvel <= vaxis) & (vaxis <= maxvel)
            vaxis = vaxis[vmask]

            # generate the data
            intensity = np.nanmean(trimmed_image.data, axis=(0, 2, 3))[vmask]

        # header
        header = {"filepath": "",
                  "imagetype": "Plot2D",
                  "size": vaxis.size,
                  "xlabel": xlabel,
                  "xunit": xunit,
                  "ylabel": ylabel,
                  "yunit": yunit,
                  "scale": ("linear", "linear"),
                  "date": str(dt.datetime.now()).replace(" ", "T"),
                  "origin": "Generated by Astroviz."}

        # plot the data
        plot = Plot2D(x=vaxis, y=intensity, header=header)

        # preview data if necessary
        if preview:
            plot.imview(**kwargs)
        
        # return the objects
        return plot
        
    def imsmooth(self, bmaj=None, bmin=None, bpa=0, width=None, unit=None, kernel="gauss",
                 fft=True, preview=True, inplace=False, **kwargs):
        """
        Perform convolution on the image.
        Parameters:
            bmaj (float): the major axis of the new beam.
            bmin (float): the minor axis of the new beam. None to set it to the same as the major axis.
            bpa (float): the position angle of the elliptical beam
            unit (float): the unit of bmaj/bmin. None to use default unit of axes.
            kernel (str): the type of convolution to be used .This parameter is case-insensitive.
                          "gauss", "gaussian", "g": Gaussian convolution
                          "box", "b": Box shape (square).
            fft (bool): True to apply fast fourier transform.
            preview (bool): True to visualize convolved image.
            inplace (bool): True to modify current image in-place. False to return new convolved image.
        Returns: 
            The final convolved image (Spatialmap)
        """
        unit = self.unit if unit is None else unit
        kernel = kernel.lower()        
        if kernel in ["gauss", "gaussian", "g"]:
            bmin = bmaj if bmin is None else bmin
            bmaj = u.Quantity(bmaj, unit).to(u.arcsec).value
            bmin = u.Quantity(bmin, unit).to(u.arcsec).value
            xsigma = bmaj / (np.abs(self.dx)*2*np.sqrt(2*np.log(2)))
            ysigma = bmin / (np.abs(self.dy)*2*np.sqrt(2*np.log(2)))
            k = Gaussian2DKernel(x_stddev=xsigma, y_stddev=ysigma, theta=np.deg2rad(-(90-bpa)))
        elif kernel in ["box", "b"]:
            width = bmaj if width is None else width
            width = u.Quantity(width, unit) / u.Quantity(np.abs(self.dx), self.unit)
            k = Box2DKernel(width=width)
            newbeam = None
        else:
            raise ValueError("'kernel' must be 'gauss' or 'box'.")
        
        print("Convolving...")
        if fft: 
            newimage = np.array([[convolve_fft(self.data[0, i], kernel=k) for i in range(self.nchan)]])
        else:
            newimage = np.array([[convolve(self.data[0, i], kernel=k) for i in range(self.nchan)]])
        
        newbeam = [bmaj, bmin, bpa]
        new_header = copy.deepcopy(self.header)
        new_header["beam"] = newbeam
        
        if inplace:
            self.data[0, 0] = newimage
            self.header = new_header
            self.__updateparams()
        
        convolved_image = Datacube(data=newimage, header=new_header)
        
        if preview:
            convolved_image.imview(**kwargs)
        
        if inplace:
            return self
        return convolved_image

    def pad(self, new_xlim=None, new_ylim=None, 
            new_imsize=None, left_pad=None, right_pad=None, 
            top_pad=None, bottom_pad=None, mode="constant",
            fill_value=np.nan, inplace=False):
        """
        Adds padding to the image data along specified axes and adjusts the reference coordinates accordingly.

        Parameters:
            new_xlim (array-like, optional): New limits for the x-axis (RA), specified as a two-element array [min, max].
                                             The method will calculate the required padding to achieve these limits.
                                             Default is None.
            
            new_ylim (array-like, optional): New limits for the y-axis (Dec), specified as a two-element array [min, max].
                                             The method will calculate the required padding to achieve these limits.
                                             Default is None.
            
            new_imsize (tuple of int, optional): New image size as a tuple (nx, ny), where nx is the new size for the x-axis
                                                 and ny is the new size for the y-axis. The method will calculate the padding
                                                 required to reach these dimensions. Default is None.
            
            left_pad (int, optional): Number of pixels to pad on the left side of the x-axis. Defaults to None, which implies
                                      no padding.
            
            right_pad (int, optional): Number of pixels to pad on the right side of the x-axis. Defaults to None, which implies
                                       no padding.
            
            top_pad (int, optional): Number of pixels to pad on the top side of the y-axis. Defaults to None, which implies
                                     no padding.
            
            bottom_pad (int, optional): Number of pixels to pad on the bottom side of the y-axis. Defaults to None, which
                                        implies no padding.
            
            fill_value (float, optional): Value to fill the padding region. Default is NaN.
            
            inplace (bool, optional): If True, modify the image data in place and return self. If False, return a new instance
                                      with the modified data. Default is False.
        
        Returns:
            self or new instance: The modified image data with added padding. Returns self if inplace=True; otherwise,
                                  returns a new instance with the padded data.

        Raises:
            ValueError: If any padding value is negative or if no valid padding, limits, or image size are specified.

        Warnings:
            If the padding difference for a new image size is an odd number, a warning is raised, indicating potential 
            uneven padding.

        Example usage:
            # Assuming `image` is an instance with the method `pad`

            # Add padding to achieve new axis limits
            new_image = image.pad(new_xlim=[10, 50], new_ylim=[-20, 30])

            # Add padding to achieve a new image size
            new_image = image.pad(new_imsize=(100, 200))

            # Add specific padding to each side
            new_image = image.pad(left_pad=5, right_pad=5, top_pad=10, bottom_pad=10)

            # Add padding in place
            image.pad(left_pad=5, right_pad=5, inplace=True)

        Notes:
            This method modifies the reference coordinates (RA, Dec) of the image data to account for the added padding.
            The new reference coordinates are recalculated based on the new position of the reference pixels.
        """
        # get attributes
        xaxis = self.xaxis
        yaxis = self.yaxis
        dx = self.dx 
        dy = self.dy
        
        # check whether parameters are specified:
        params = (new_xlim, new_ylim, new_imsize, left_pad, right_pad, top_pad, bottom_pad)
        if all(param is None for param in params):
            raise ValueError("You must specify at least one of the following: " + \
                             "new_xlim, new_ylim, new_imsize, or any of the padding " + \
                             "values (left_pad, right_pad, top_pad, bottom_pad).")
        
        if new_imsize is not None:
            x_diff = new_imsize[0] - xaxis.size
            if x_diff % 2:  # if difference is odd number
                warnings.warn("Padding difference for x-axis is an odd number; " + \
                              "this may lead to uneven padding.")
                left_pad = x_diff // 2 + 1
            else:
                left_pad = x_diff // 2
            right_pad = x_diff // 2
            
            y_diff = new_imsize[1] - yaxis.size
            if y_diff % 2:  # if difference is odd number
                warnings.warn("Padding difference for y-axis is an odd number; " + \
                              "this may lead to uneven padding.")
                bottom_pad = y_diff // 2 + 1
            else:
                bottom_pad = y_diff // 2 
            top_pad = y_diff // 2
        
        if new_xlim is not None:
            new_xlim = np.array(new_xlim)
            left_pad = (new_xlim.max() - xaxis.max())/-dx
            left_pad = _true_round_whole(left_pad)
            right_pad = (xaxis.min() - new_xlim.min())/-dx
            right_pad = _true_round_whole(right_pad)
            
        if new_ylim is not None:
            new_ylim = np.array(new_ylim)
            top_pad = (new_ylim.max() - yaxis.max())/dy
            top_pad = _true_round_whole(top_pad)
            bottom_pad = (yaxis.min() - new_ylim.min())/dy
            bottom_pad = _true_round_whole(bottom_pad)
        
        # set to default values (0) if not specified
        if left_pad is None:
            left_pad = 0
        if right_pad is None:
            right_pad = 0
        if top_pad is None:
            top_pad = 0
        if bottom_pad is None:
            bottom_pad = 0
            
        # raise value error if number of pad is negative
        if any(num < 0 for num in (top_pad, bottom_pad, left_pad, right_pad)):
            raise ValueError("Padding values cannot be negative.")
            
        # add padding to new data
        pad_width = ((0, 0), (bottom_pad, top_pad), (left_pad, right_pad))
        new_data = np.pad(self.data[0],  # three-dimensional
                          pad_width=pad_width, 
                          mode="constant", 
                          constant_values=fill_value)
        new_data = new_data[np.newaxis, :, :, :]  # add stokes axis 
        
        # calculate new cell sizes and reference pixels
        new_nx = new_data.shape[3]
        new_refnx = new_nx // 2 + 1
        new_ny = new_data.shape[2]
        new_refny = new_ny // 2 + 1
        
        # calculate new reference coordinate
        current_refcoord = SkyCoord(self.refcoord)
        current_ra = current_refcoord.ra
        current_dec = current_refcoord.dec

        new_ra = current_ra + (right_pad-left_pad+1)*dx*u.Unit(self.unit)
        new_dec = current_dec + (top_pad-bottom_pad+1)*dy*u.Unit(self.unit)
        new_refcoord = SkyCoord(ra=new_ra, dec=new_dec).to_string("hmsdms")
        
        # update stuff
        image = self if inplace else self.copy()
        image.data = new_data
        image.overwrite_header(nx=new_nx, refnx=new_refnx, 
                               ny=new_ny, refny=new_refny,
                               refcoord=new_refcoord,
                               shape=new_data.shape,
                               )
        return image

    def imregrid(self, template, interpolation="linear", parallel=True, inplace=False):
        """
        Regrids the image to match the grid of a template image.

        This method adjusts the image to have the same coordinate grid as the template image by shifting, trimming, and/or padding
        the image data, and then regridding the data using interpolation.

        Parameters:
            template (Spatialmap or Datacube): The template image whose grid will be matched. Must be an instance of Spatialmap or Datacube.
            
            interpolation (str, optional): The interpolation method to use for regridding. Options include "linear", "nearest", and "cubic". 
                                           Default is "linear".
            
            inplace (bool, optional): If True, modifies the image in place and returns self. If False, returns a new instance with the regridded data.
                                      Default is False.

        Returns:
            self or new instance: The regridded image. Returns self if inplace=True; otherwise, returns a new instance with the regridded data.

        Raises:
            ValueError: If the template is not an instance of Spatialmap or Datacube.
            Exception: If inconsistent padding and trimming conditions are detected.

        Example usage:
            # Assuming `image` is an instance with the method `imregrid` and `template` is a Spatialmap or Datacube instance

            # Regrid the image to match the template
            new_image = image.imregrid(template)

            # Regrid the image with cubic interpolation and modify in place
            image.imregrid(template, interpolation="cubic", inplace=True)
        
        Notes:
            - The method first shifts the image to the new reference coordinate of the template.
            - The image is then trimmed or padded to match the limits and size of the template.
            - Finally, the image data is regridded using the specified interpolation method to match the template grid.
        """
        image = self if inplace else self.copy()

        if not isinstance(template, (Spatialmap, Datacube)):
            raise ValueError("Template must be an instance of 'Spatialmap' or 'Datacube'.")

        new_dx = template.dx
        new_dy = template.dy 

        if self.unit != template.unit:
            conversion_factor = (1*u.Unit(template.unit)).to_value(self.unit)
            new_dx *= conversion_factor
            new_dy *= conversion_factor

        # first shift the image to new coordinate
        image.imshift(template, inplace=True)

        # change cell size of data through interpolation
        image = _change_cell_size(image=image,
                                  dx=new_dx,
                                  dy=new_dy,
                                  interpolation=interpolation,
                                  inplace=True)

        # trim/pad limits of the image
        image = _match_limits(image=image, template_image=template, inplace=True)
        
        return image
    
    def normalize(self, valrange=None, template=None, inplace=False):
        """
        Normalize the data of the data cube to a specified range.
        Parameters:
            valrange (list/tuple): The range [min, max] to which the data will be normalized. Default is [0, 1].
            inplace (bool): If True, modify the data in-place. 
                            If False, return a new Spatialmap instance. Default is False.
        Returns:
            The normalized image, either as a new instance or the same instance based on the 'inplace' parameter.
        """
        if template is None:
            if valrange is None:
                valrange = [0, 1]
        else:
            valrange = [np.nanmin(template), np.nanmax(template)]

        # Extracting min and max from the range
        min_val, max_val = valrange

        # Normalization formula: (data - min(data)) / (max(data) - min(data)) * (max_val - min_val) + min_val
        data_min, data_max = np.nanmin(self.data), np.nanmax(self.data)
        normalized_data = (self.data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val

        if inplace:
            self.data = normalized_data
            return self
        return Datacube(header=self.header, data=normalized_data)
        
    def imshift(self, coord, unit=None, printcc=True, order=3,
                mode="constant", cval=np.nan, prefilter=True, 
                parallel=True, inplace=True):
        """
        Shifts the image to the desired coordinate.

        This method shifts the image data to a specified coordinate, either in J2000 format or in relative units.
        The shifting operation can be performed in parallel to improve performance.

        Parameters:
            coord (str/tuple/list/None): The target coordinate to shift the image to.
                                         If a string, it is interpreted as a J2000 coordinate.
                                         If a tuple or list, it is interpreted as relative coordinates.
                                         The format depends on the type:
                                         - J2000 (str): Example: '12h30m00s +12d30m00s'
                                         - Relative (tuple/list): Example: [30, 40] with units specified by `unit`.

            unit (str, optional): The unit of the coordinate if `coord` is not in J2000 format. Default is None,
                                  which uses the unit attribute of the instance.

            printcc (bool, optional): If True, prints the new center coordinate after shifting. Default is True.

            order (int, optional): The order of the spline interpolation used in the shifting process. Default is 3.

            mode (str, optional): Points outside the boundaries of the input are filled according to the given mode.
                                  Default is 'constant'. Modes match the `scipy.ndimage.shift` options.

            cval (float, optional): The value to use for points outside the boundaries of the input when mode is 'constant'.
                                    Default is NaN.

            prefilter (bool, optional): Determines if the input should be pre-filtered before applying the shift. Default is True.

            parallel (bool, optional): If True, performs the shifting operation in parallel across multiple channels.
                                       Default is True.

            inplace (bool, optional): If True, modifies the image data in-place. If False, returns a new instance
                                      with the shifted data. Default is True.

        Returns:
            Image: The shifted image instance. If `inplace` is True, returns self after modification.
                   Otherwise, returns a new instance with the shifted data.

        Raises:
            ValueError: If the provided coordinate format is invalid.

        Example usage:
            # Assuming `image` is an instance with the method `imshift`

            # Shift image to a new J2000 coordinate
            shifted_image = image.imshift('12h30m00s +12d30m00s')

            # Shift image to a new relative coordinate with specified units
            shifted_image = image.imshift([30, 40], unit='arcsec')

            # Shift image in-place
            image.imshift('12h30m00s +12d30m00s', inplace=True)

            # Shift image with parallel processing disabled
            shifted_image = image.imshift([30, 40], unit='arcsec', parallel=False)

        Notes:
            - This method converts J2000 coordinates to arcseconds and calculates the pixel shift required.
            - If the image contains NaN values, they will be replaced with zeros during the shifting process.
            - The new reference coordinate is updated in the image header.
        """
        unit = self.unit if unit is None else unit
        if isinstance(coord, (Spatialmap, Datacube, PVdiagram)):
            coord = coord.header["refcoord"]
        image = self if inplace else self.copy()

        # convert J2000 to arcsec
        isJ2000 = (isinstance(coord, str) and not coord.isnumeric()) \
                    or (len(coord)==2 and isinstance(coord[0], str) and isinstance(coord[1], str)) 
        if isJ2000:
            shift = _icrs2relative(coord=coord, ref=self.refcoord, unit="arcsec")
            shiftx, shifty = -shift[0], -shift[1]
            if len(coord) == 2:
                new_refcoord = SkyCoord(ra=coord[0], dec=coord[1], unit=(u.hourangle, u.deg), 
                                        frame='icrs').to_string("hmsdms")
            else:
                new_refcoord = SkyCoord(coord, unit=(u.hourangle, u.deg), 
                                        frame='icrs').to_string("hmsdms")
            image.refcoord = image.header["refcoord"] = new_refcoord
        else:
            shiftx, shifty = -u.Quantity(coord[0], unit), -u.Quantity(coord[1], unit)
            new_refcoord = _relative2icrs(coord, ref=self.refcoord, unit=unit)
            image.refcoord = image.header["refcoord"] = new_refcoord
        
        if printcc:
            print(f"New center [J2000]: {new_refcoord}")
        
        dx = u.Quantity(np.abs(self.dx), self.unit)
        dy = u.Quantity(np.abs(self.dy), self.unit)
        pixelx = shiftx / dx  # allow sub-pixel shifts
        pixely = shifty / dy  # allow sub-pixel shifts

        # warn users that nan will be replaced with zero:
        if np.any(np.isnan(image.data)):
            warnings.warn("NaN values were detected and will be replaced with zeros after 'imshift'.")
            image.data = np.nan_to_num(image.data)

        shift = np.array([self.refny-1, self.refnx-1])-[self.refny-pixely, self.refnx+pixelx]

        if parallel and self.nchan > 5:  # make sure parallelism makes sense (there are enough channels)
            # warn user that they may not have enough CPUs for parallelism
            if os.cpu_count() == 1:
                warnings.warn("This device only has 1 CPU, which is not suitable for parallelism.")

            # parallel processing
            with ProcessPoolExecutor() as executor:
                results = executor.map(ndimage.shift,  # input a callable function
                                       image.data[0],
                                       (shift,) * self.nchan,  # parameter: shift
                                       (None,) * self.nchan,  # parameter: output
                                       (order,) * self.nchan,  # parameter: order
                                       (mode,) * self.nchan,  # parameter: mode
                                       (cval,) * self.nchan,  # parameter: cval
                                       (prefilter,) * self.nchan,  # parameter: prefilter
                                       )
            image.data[0] = np.array(list(results))
        else:
            # use list comprehension, which is faster than assigning one by one
            image.data[0] = np.array([ndimage.shift(image.data[0, i], shift=shift, cval=cval, 
                                                    output=None, order=order, mode=mode, 
                                                    prefilter=prefilter) for i in range(self.nchan)])
        image.__updateparams()
        return image
    
    def peakshift(self, inplace=True, **kwargs):
        """
        Shift the maximum value of the image to the center of the image.
        Parameter:
            inplace (bool): True to modify the current image in-place. False to return a new image.
        Returns:
            The shifted image.
        """
        mom8_data = np.nanmax(self.data, axis=1)[0]  # maximum intensity of spectrum
        indices = np.unravel_index(np.nanargmax(mom8_data), mom8_data.shape)
        xx, yy = self.get_xyaxes(grid=True, unit=None)
        coord = xx[indices], yy[indices]
        if self._peakshifted:
            print("The peak is already shifted to the center.")
            return self.copy()
        else:
            shifted_image = self.imshift(coord=coord, printcc=True, unit=self.unit, inplace=inplace, **kwargs)
            if inplace:
                self._peakshifted = True
            else:
                shifted_image._peakshifted = True
            print(f"Shifted to {coord} [{self.unit}]")
            return shifted_image
    
    def line_info(self, **kwargs):
        """
        This method searches for the molecular line data from the Splatalogue database
        """
        if np.isnan(self.restfreq):
            raise Exception("Failed to find molecular line as rest frequency cannot be read.")
        return search_molecular_line(self.restfreq, unit="Hz", **kwargs)
    
    def get_hduheader(self):
        """
        To retrieve the header of the current FITS image. This method accesses the header 
        information of the original FITS file, and then modifies it to reflect the current
        status of this image object.

        Returns:
            The FITS header of the current image object (astropy.io.fits.header.Header).
        """
        self.__updateparams()
        return _get_hduheader(self)

    def get_wcs(self):
        """
        Get the world coordinate system of the image (astropy object.)
        """
        return WCS(self.get_hduheader())

    def get_hdu(self):
        """
        Get the primary HDU (astropy object) of the image.
        """
        return fits.PrimaryHDU(data=self.data, header=self.get_hduheader())
    
    def overwrite_header(self, new_vals=None, **kwargs):
        """
        Method to overwrite the existing keys of the header with new values.
        Parameters:
            new_vals (dict): a dictionary containing keys and values to be overwritten.
            **kwargs (dict): keyword arguments that will be overwritten in the header
        Return:
            self.header (dict): the updated header 
        """
        if new_vals is None and len(kwargs) == 0:
            raise ValueError("Header cannot be overwritten. Need to input a dictionary or keyword arguments.")
        if new_vals is not None:
            if isinstance(new_vals, dict):
                for key, value in new_vals.items():
                    if key in self.header:
                        self.header[key] = value
                    else:
                        warnings.warn(f"'{key}' is not a valid keyword of the header and will be ignored.")
            else:
                raise TypeError("Please input a new dictionary as the header.")
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                if key in self.header:
                    self.header[key] = value
                else:
                    warnings.warn(f"'{key}' is not a valid keyword of the header and will be ignored.")
        self.__updateparams()
        return self.header
    
    def exportfits(self, outname, overwrite=False):
        """
        Save the current image to a FITS file.

        This method exports the image data and header to a FITS file. If the specified 
        file name does not end with '.fits', it is appended. When 'overwrite' is False 
        and the file already exists, a number is appended to the file name to avoid 
        overwriting (e.g., 'filename(1).fits').

        Parameters:
            outname (str): The name of the output FITS file.
            overwrite (bool): If True, allows overwriting an existing file with the 
                              same name. If False, the file name is modified to 
                              prevent overwriting existing files.
        Returns:
            None
        """
        # get header
        hdu_header = self.get_hduheader()
        
        # add file name extension if not in user input
        if not outname.endswith(".fits"):
            outname += ".fits"
        
        # if not overwrite, add (1), (2), (3), etc. to file name before '.fits'
        if not overwrite:
            outname = _prevent_overwriting(outname)
        
        # Write to a FITS file
        hdu = fits.PrimaryHDU(data=self.data, header=hdu_header)
        hdu.writeto(outname, overwrite=overwrite)
        print(f"File saved as '{outname}'.")

    def to_casa(self, *args, **kwargs):
        """
        Converts the data cube object into CASA image format. 
        Wraps the 'importfits' function of casatasks.

        Parameters:
            outname (str): The output name for the CASA image file. Must end with ".image".
            whichrep (int, optional): The FITS representation to convert. Defaults to 0.
            whichhdu (int, optional): The HDU (Header/Data Unit) number to convert. Defaults to -1.
            zeroblanks (bool, optional): Replace undefined values with zeros. Defaults to True.
            overwrite (bool, optional): Overwrite the output file if it already exists. Defaults to False.
            defaultaxes (bool, optional): Use default axes for the output CASA image. Defaults to False.
            defaultaxesvalues (str, optional): Default axes values, provided as a string. Defaults to '[]'.
            beam (str, optional): Beam parameters, provided as a string. Defaults to '[]'.

        Raises:
            ValueError: If 'outname' is not a string.

        Returns:
            None

        Example:
            image.to_casa("output_image")
        """
        to_casa(self, *args, **kwargs)


class Spatialmap:
    """
    A class for handling FITS 2D spatial maps (e.g., continuum maps and moment maps) 
    in astronomical imaging.

    This class provides functionality to load, process, and manipulate 2D FITS images. 
    It supports operations like arithmetic calculations between data cubes, rotation, 
    normalization, regridding, and more. The class can handle FITS files directly and acts
    like a numpy array.

    Note:
        The class performs several checks on initialization to ensure that the provided data
        is in the correct format. It can handle FITS files with different configurations and
        is designed to be flexible for various data shapes and sizes.
    """
    def __init__(self, fitsfile=None, header=None, data=None, hduindex=0, 
                 spatialunit="arcsec", quiet=False):
        if fitsfile is not None:
            fits = importfits(fitsfile, hduindex=hduindex, spatialunit=spatialunit, quiet=False)
            self.header = fits.header
            self.data = fits.data
        elif header is not None:
            self.header = header
            self.data = data
        if self.header["imagetype"] != "spatialmap":
            raise TypeError("The given FITS file is not a spatial map.")
        self.__updateparams()
        self._peakshifted = False
        if isinstance(self.data, u.quantity.Quantity):
            self.value = Spatialmap(header=self.header, data=self.data.value)
    
    # magic methods to define operators
    def __add__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return Spatialmap(header=self.header, data=self.data+other.data)
        return Spatialmap(header=self.header, data=self.data+other)
    
    def __radd__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return Spatialmap(header=self.header, data=other.data+self.data)
        return Spatialmap(header=self.header, data=other+self.data)
        
    def __sub__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return Spatialmap(header=self.header, data=self.data-other.data)
        return Spatialmap(header=self.header, data=self.data-other)
    
    def __rsub__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return Spatialmap(header=self.header, data=other.data-self.data)
        return Spatialmap(header=self.header, data=other-self.data)
        
    def __mul__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data*other.data)
        return Spatialmap(header=self.header, data=self.data*other)
    
    def __rmul__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=other.data*self.data)
        return Spatialmap(header=self.header, data=other*self.data)
    
    def __pow__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data**other.data)
        return Spatialmap(header=self.header, data=self.data**other)
        
    def __rpow__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=other.data**self.data)
        return Spatialmap(header=self.header, data=other.data**self.data)
        
    def __truediv__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data/other.data)
        return Spatialmap(header=self.header, data=self.data/other)
    
    def __rtruediv__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=other.data/self.data)
        return Spatialmap(header=self.header, data=other/self.data)
        
    def __floordiv__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data//other.data)
        return Spatialmap(header=self.header, data=self.data//other)
    
    def __rfloordiv__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=other.data//self.data)
        return Spatialmap(header=self.header, data=other//self.data)
    
    def __mod__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data%other.data)
        return Spatialmap(header=self.header, data=self.data%other)
    
    def __rmod__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=other.data%self.data)
        return Spatialmap(header=self.header, data=other%self.data)
    
    def __lt__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data<other.data)
        return Spatialmap(header=self.header, data=self.data<other)
    
    def __le__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data<=other.data)
        return Spatialmap(header=self.header, data=self.data<=other)
    
    def __eq__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data==other.data)
        return Spatialmap(header=self.header, data=self.data==other)
        
    def __ne__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data!=other.data)
        return Spatialmap(header=self.header, data=self.data!=other)

    def __gt__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data>other.data)
        return Spatialmap(header=self.header, data=self.data>other)
        
    def __ge__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return Spatialmap(header=self.header, data=self.data>=other.data)
        return Spatialmap(header=self.header, data=self.data>=other)

    def __abs__(self):
        return Spatialmap(header=self.header, data=np.abs(self.data))
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return Spatialmap(header=self.header, data=-self.data)
    
    def __invert__(self):
        return Spatialmap(header=self.header, data=~self.data)
    
    def __getitem__(self, indices):
        try:
            try:
                return Spatialmap(header=self.header, data=self.data[indices])
            except:
                warnings.warn("Returning value after reshaping image data to 2 dimensions.")
                return self.data.copy[:, indices[0], indices[1]]
        except:
            return self.data[indices]
    
    def __setitem__(self, indices, value):
        newdata = self.data.copy()
        newdata[indices] = value
        return Spatialmap(header=self.header, data=newdata)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Extract the Spatialmap object from inputs
        inputs = [x.data if isinstance(x, Spatialmap) else x for x in inputs]
        
        if ufunc == np.round:
            # Process the round operation
            if 'decimals' in kwargs:
                decimals = kwargs['decimals']
            elif len(inputs) > 1:
                decimals = inputs[1]
            else:
                decimals = 0  # Default value for decimals
            return Spatialmap(header=self.header, data=np.round(self.data, decimals))
        
        # Apply the numpy ufunc to the data
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Return a new Spatialmap instance with the result if the ufunc operation was successful
        if method == '__call__' and isinstance(result, np.ndarray):
            return Spatialmap(header=self.header, data=result)
        else:
            return result
        
    def __array__(self, *args, **kwargs):
        return np.array(self.data, *args, **kwargs)

    def min(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the minimum value of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's min function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's min function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the minimum.

        Returns:
            scalar or array: Minimum value of the data array. If the data array is multi-dimensional, 
                             returns an array of minimum values along the specified axis.
        """
        if ignore_nan:
            return np.nanmin(self.data, *args, **kwargs)
        return np.min(self.data, *args, **kwargs)

    def max(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the maximum value of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's max function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's max function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the maximum.

        Returns:
            scalar or array: Maximum value of the data array. If the data array is multi-dimensional, 
                             returns an array of maximum values along the specified axis.
        """
        if ignore_nan:
            return np.nanmax(self.data, *args, **kwargs)
        return np.max(self.data, *args, **kwargs)

    def mean(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the mean value of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's mean function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's mean function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the mean.

        Returns:
            scalar or array: Mean value of the data array. If the data array is multi-dimensional, 
                             returns an array of mean values along the specified axis.
        """
        if ignore_nan:
            return np.nanmean(self.data, *args, **kwargs)
        return np.mean(self.data, *args, **kwargs)

    def sum(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the sum of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's sum function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's sum function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the sum.

        Returns:
            scalar or array: Sum of the data array. If the data array is multi-dimensional, 
                             returns an array of sum values along the specified axis.
        """
        if ignore_nan:
            return np.nansum(self.data, *args, **kwargs)
        return np.sum(self.data, *args, **kwargs)

    def flux(self, region=None, rms=None, rms_unit=None,
             print_results=True, **kwargs):
        """
        Calculate the total flux within a specified region of the image, taking into account the noise level.

        Args:
            region (Region, optional): The region within the image to calculate the flux. If None, the whole image is used.
                This should be an instance of a `Region` class or any other object that defines a region.
            rms (float or Quantity, optional): The root mean square (rms) noise level. If None, the rms noise is estimated from the image.
            rms_unit (str, optional): The unit of the rms noise level. If None, the unit is assumed to be the same as the image unit.
            print_results (bool, optional): If True, the results are printed. Default is True.
            **kwargs: Additional keyword arguments passed to the region masking function.

        Returns:
            tuple: A tuple containing:
                total_flux (float): The total flux within the specified region in mJy.
                flux_error (float): The error in the flux measurement in mJy.

        Notes:
            The function first converts the image to units of mJy/pixel. If the rms noise level is not provided, 
            it is estimated from the image. If a region is specified, the image is masked to this region.
            The function then calculates the total flux by summing the pixel values within the region.
            The error in the flux measurement is calculated as the square root of the number of pixels 
            in the region multiplied by the rms noise level converted to mJy/pixel.

            The results can be optionally printed, displaying the total flux, the error in the flux, 
            the number of pixels in the beam, and the number of pixels in the specified region.

        Examples:
            >>> flux, error = image.flux(region=my_region, rms=0.5)
            >>> flux, error = image.flux(print_results=False)
        """
        # convert image to mJy/pixel
        try:
            img_mJyppix = self.conv_bunit("mJy/pixel", inplace=False)
        except UnitConversionError:
            raise Exception("Image must be a continuum map or a single channel map.")

        # estimate noise from image if rms is not specified:
        estimate_rms = (rms is None)
        if estimate_rms:
            rms = self.noise()
            rms_unit = self.bunit
        elif isinstance(rms, u.Quantity):  # get the unit and value if it is an astropy 'Quantity' instance
            rms_unit = _apu_to_headerstr(rms.unit)
            rms = rms.value

        # create image for the rms to convert unit and then get the converted rms:
        rms_image = self.copy()
        if rms_unit is not None and rms_image.bunit != rms_unit:
            rms_image.set_header(bunit=rms_unit)  # set to correct unit
        rms_image.data = rms  # image value is the rms
        rms_image.conv_bunit("mJy/pixel", inplace=True)
        rms_mJyppix = rms_image.data  # rms converted to mJy/pixel

        # mask region 
        if region is not None:
            img_mJyppix = img_mJyppix.mask_region(region, **kwargs, preview=False, inplace=False)

        # calculate the total flux and the error from the rms
        total_flux = img_mJyppix.sum()
        pix_in_aperature = np.sum(~np.isnan(img_mJyppix.data))  # count number of pixels in aperature
        flux_error = np.sqrt(pix_in_aperature) * rms_mJyppix

        # print the measured results
        if print_results:
            print("Total Flux Measurement".center(40, "#"))
            print(f"Flux: {total_flux:.4f} +/- {flux_error:.4f} [mJy]")
            print(f"Number of pixels in aperature: {pix_in_aperature}")
            if estimate_rms:
                print(f"Estimated RMS: {rms_mJyppix} [mJy]")
            print(40*"#", end="\n\n")

        return total_flux, flux_error
    
    def to(self, unit, *args, **kwargs):
        """
        This method converts the intensity unit of original image to the specified unit.
        """
        return Spatialmap(header=self.header, data=self.data.to(unit, *args, **kwargs))
    
    def to_value(self, unit, *args, **kwargs):
        """
        Duplicate of astropy.unit.Quantity's 'to_value' method.
        """
        image = self.copy()
        image.data = image.data.to_value(unit, *args, **kwargs)
        image.bunit = image.header["bunit"] = _apu_to_headerstr(_to_apu(unit))
        return image
    
    def copy(self):
        """
        This method creates a copy of the original image.
        """
        return copy.deepcopy(self)
        
    def __updateparams(self):
        # make axes
        self.spatialunit = self.unit = self.axisunit = self.header["unit"]
        nx = self.nx = self.header["nx"]
        dx = self.dx = self.header["dx"]
        refnx = self.refnx = self.header["refnx"]
        ny = self.ny = self.header["ny"]
        dy = self.dy = self.header["dy"]
        refny = self.refny = self.header["refny"]
        self.xaxis, self.yaxis = self.get_xyaxes()
        self.shape = self.header["shape"]
        self.size = self.data.size
        self.restfreq = self.header["restfreq"]
        self.bmaj, self.bmin, self.bpa = self.beam = self.header["beam"]
        self.resolution = np.sqrt(self.beam[0]*self.beam[1]) if self.beam is not None else None
        self.refcoord = self.header["refcoord"]
        if isinstance(self.data, u.Quantity):
            self.bunit = self.header["bunit"] = _apu_to_headerstr(self.data.unit)
        else:
            self.bunit = self.header["bunit"]
        xmin, xmax = self.xaxis[[0, -1]]
        ymin, ymax = self.yaxis[[0, -1]]
        self.imextent = [xmin-0.5*self.dx, xmax+0.5*self.dx, 
                         ymin-0.5*self.dy, ymax+0.5*self.dy]
        self.widestfov = max(self.xaxis[0], self.yaxis[-1])
        
    def get_xyaxes(self, grid=False, unit=None):
        """
        Obtain the 1-dimensional or 2-dimensional x- and y-axes of the image.
        Parameters:
            grid (bool): True to return the x and y grids 
            unit (str): The angular unit of the positional axes. None to use the default unit (self.unit).
        Returns:
            xaxis (ndarray, if grid=False): the x-axis
            yaxis (ndarray, if grid=False): the y-axis
            xx (ndarray, if grid=True): the x grid
            yy (ndarray, if grid=True): the y grid
        """
        # create axes in the same unit as self.unit
        xaxis = self.dx * (np.arange(self.nx)-self.refnx+1)
        yaxis = self.dy * (np.arange(self.ny)-self.refny+1)
        
        # convert to the specified unit if necessary
        if unit is not None:
            xaxis = u.Quantity(xaxis, self.unit).to(unit).value
            yaxis = u.Quantity(yaxis, self.unit).to(unit).value
        
        # make 2D grid if grid is set to True and return the arrays
        if grid:
            xx, yy = np.meshgrid(xaxis, yaxis)
            return xx, yy
        return xaxis, yaxis
    
    def rotate(self, angle=0, ccw=True, unit="deg", inplace=False):
        """
        Rotate the image by the specified degree.
        Parameters:
            angle (float): angle by which image will be rotated
            ccw (bool): True to rotate counterclockwise. False to rotate clockwise.
            unit (str): unit of angle
            inplace (bool): whether to replace data of this image with the rotated data.
        Return: 
            The rotated image (Spatialmap)
        """
        if unit not in ["deg", "degree", "degrees"]:
            angle = u.Quantity(angle, unit).to(u.deg).value
        if ccw:
            angle = -angle
        
        # rotate data
        newdata = ndimage.rotate(self.data, angle, order=1, axes=(2, 3))
        
        # take the center of the rotated image
        y, x = newdata.shape[2:]
        nx, ny = self.nx, self.ny
        startx, starty = x//2 - nx//2, y//2 - ny//2
        newdata = newdata[:, :, starty:starty+ny, startx:startx+nx]
        
        if inplace:
            self.data = newdata
            return self
        return Spatialmap(header=self.header, data=newdata.reshape(self.shape))
        
    def imshift(self, coord, unit=None, printcc=True, order=3,
                mode="constant", cval=np.nan, prefilter=True,
                inplace=True, **kwargs):
        """
        Shifts the image to the desired coordinate.

        This method shifts the image data to a specified coordinate, either in J2000 format or in relative units.

        Parameters:
            coord (str/tuple/list): The target coordinate to shift the image to.
                                     If a string, it is interpreted as a J2000 coordinate.
                                     If a tuple or list, it is interpreted as relative coordinates.
                                     The format depends on the type:
                                     - J2000 (str): Example: '12h30m00s +12d30m00s'
                                     - Relative (tuple/list): Example: [30, 40] with units specified by `unit`.

            unit (str, optional): The unit of the coordinate if `coord` is not in J2000 format. Default is None,
                                  which uses the unit attribute of the instance.

            printcc (bool, optional): If True, prints the new center coordinate after shifting. Default is True.

            order (int, optional): The order of the spline interpolation used in the shifting process. Default is 3.

            mode (str, optional): Points outside the boundaries of the input are filled according to the given mode.
                                  Default is 'constant'. Modes match the `scipy.ndimage.shift` options.

            cval (float, optional): The value to use for points outside the boundaries of the input when mode is 'constant'.
                                    Default is NaN.

            prefilter (bool, optional): Determines if the input should be pre-filtered before applying the shift. Default is True.

            inplace (bool, optional): If True, modifies the image data in-place. If False, returns a new instance
                                      with the shifted data. Default is True.

        Returns:
            Image: The shifted image instance. If `inplace` is True, returns self after modification.
                   Otherwise, returns a new instance with the shifted data.

        Raises:
            ValueError: If the provided coordinate format is invalid.

        Example usage:
            # Assuming `image` is an instance with the method `imshift`

            # Shift image to a new J2000 coordinate
            shifted_image = image.imshift('12h30m00s +12d30m00s')

            # Shift image to a new relative coordinate with specified units
            shifted_image = image.imshift([30, 40], unit='arcsec')

            # Shift image in-place
            image.imshift('12h30m00s +12d30m00s', inplace=True)

            # Shift image with parallel processing disabled
            shifted_image = image.imshift([30, 40], unit='arcsec')

        Notes:
            - This method converts J2000 coordinates to arcseconds and calculates the pixel shift required.
            - If the image contains NaN values, they will be replaced with zeros during the shifting process.
            - The new reference coordinate is updated in the image header.
        """
        unit = self.unit if unit is None else unit
        if isinstance(coord, (Spatialmap, Datacube, PVdiagram)):
            coord = coord.header["refcoord"]
        image = self if inplace else self.copy()
        # convert J2000 to arcsec
        isJ2000 = (isinstance(coord, str) and not coord.isnumeric()) \
                    or (len(coord)==2 and isinstance(coord[0], str) and isinstance(coord[1], str)) 
        if isJ2000:
            shift = _icrs2relative(coord=coord, ref=self.refcoord, unit="arcsec")
            shiftx, shifty = -shift[0], -shift[1]
            if len(coord) == 2:
                new_refcoord = SkyCoord(coord[0], coord[1], unit=(u.hourangle, u.deg), frame='icrs').to_string("hmsdms")
            else:
                new_refcoord = SkyCoord(coord, unit=(u.hourangle, u.deg), frame='icrs').to_string("hmsdms")
            image.refcoord = image.header["refcoord"] = new_refcoord
        else:
            shiftx, shifty = -u.Quantity(coord[0], unit), -u.Quantity(coord[1], unit)
            new_refcoord = _relative2icrs(coord, ref=self.refcoord, unit=unit)
            image.refcoord = image.header["refcoord"] = new_refcoord
        
        if np.abs(shiftx) > np.abs(u.Quantity(self.dx, self.unit)) or np.abs(shifty) > np.abs(u.Quantity(self.dx, self.unit)):
            image._peakshifted = False
        
        if printcc:
            print(f"New center [J2000]: {new_refcoord}")
        
        dx = u.Quantity(np.abs(self.dx), self.unit)
        dy = u.Quantity(np.abs(self.dy), self.unit)
        pixelx = shiftx / dx
        pixely = shifty / dy

        # warn users about NaN values
        if np.any(np.isnan(image.data)):
            warnings.warn("NaN values were detected and will be replaced with zeros after 'imshift'.")
            image.data = np.nan_to_num(image.data)

        image.data[0, 0] = ndimage.shift(image.data[0, 0], 
                                         shift=np.array([image.refny-1, image.refnx-1])-[image.refny-pixely, image.refnx+pixelx], 
                                         cval=cval, 
                                         order=order,
                                         mode=mode, 
                                         prefilter=prefilter,
                                         **kwargs)        
        image.__updateparams()
        return image
    
    def peakshift(self, inplace=True, **kwargs):
        """
        Shift the maximum value of the image to the center of the image.
        Parameter:
            inplace (bool): True to modify the current image in-place. False to return a new image.
        Returns:
            The shifted image.
        """
        indices = np.unravel_index(np.nanargmax(self.data[0, 0]), self.data[0, 0].shape)
        xx, yy = self.get_xyaxes(grid=True, unit=None)
        coord = xx[indices], yy[indices]
        if self._peakshifted:
            print("The peak is already shifted to the center.")
            return self.copy()
        else:
            shifted_image = self.imshift(coord=coord, printcc=True, 
                                         unit=self.unit, inplace=inplace, **kwargs)
            if inplace:
                self._peakshifted = True
            else:
                shifted_image._peakshifted = True
            print(f"Shifted to {coord} [{self.unit}]")
            return shifted_image
        
    def max_pixel(self, index=False):
        """
        Method to get the pixel with the maximum value.
        Parameters:
            index: True to return the index of the maximum value. False to return the unit of the 
        Returns:
            The coordinate/index with the minimum value (tuple)
        """
        data = self.data
        peakpix = np.unravel_index(np.nanargmax(self.data), self.shape)
        if index:
            xval = peakpix[3]
            yval = peakpix[2]
        else:
            xval = self.xaxis[peakpix[3]]
            yval = self.yaxis[peakpix[2]]
        return (xval, yval)
    
    def min_pixel(self, index=False):
        """
        Method to get the pixel with the maximum value.
        Parameters:
            index: True to return the index of the maximum value. False to return the unit of the 
        Returns:
            The coordinate/index with the minimum value (tuple)
        """
        data = self.data
        peakpix = np.unravel_index(np.nanargmin(self.data), self.shape)
        if index:
            xval = peakpix[3]
            yval = peakpix[2]
        else:
            xval = self.xaxis[peakpix[3]]
            yval = self.yaxis[peakpix[2]]
        return (xval, yval)
    
    def mask_region(self, region, exclude=False, preview=True, inplace=False, **kwargs):
        """
        Mask the specified region.
        Parameters:
            region (Region): the region object inside/outside of which the image will be masked.
            exclude (bool): True to exclude pixels outside the region. False to exclude those inside.
            preview (bool): True to visualize masked image. 
            inplace (bool): True to modify the image in-place. False to return a new image.
        Returns:
            The masked image.
        """
        if region.shape == "line":
            raise Exception(f"Region shape cannot be '{region.shape}' ")
        region = self.__readregion(region)
        data = self.data.copy()
        xx, yy = self.get_xyaxes(grid=True)
        if region.shape == "circle":
            radius = region.radius
            x, y = region.center
            mask = ((xx-x)**2+(yy-y)**2 <= radius**2)
        elif region.shape == "ellipse":
            angle = np.deg2rad(region.pa+270)
            x, y = region.center
            a = region.semimajor 
            b = region.semiminor
            xx_prime = (xx-x)*np.cos(angle)-(yy-y)*np.sin(angle)
            yy_prime = (xx-x)*np.sin(angle)+(yy-y)*np.cos(angle)
            mask = np.square(xx_prime/a) + np.square(yy_prime/b) <= 1
        elif region.shape == "box":
            # get width and height
            width, height = region.width, region.height
            
            # shift center
            xx -= region.center[0]
            yy -= region.center[1]

            # rotate
            pa_rad = np.deg2rad(region.pa)
            xx_rot = xx*np.cos(pa_rad)-yy*np.sin(pa_rad)
            yy_rot = xx*np.sin(pa_rad)+yy*np.cos(pa_rad)

            # create masks
            xmask = (-width/2 <= xx_rot) & (xx_rot <= width/2)
            ymask = (-height/2 <= yy_rot) & (yy_rot <= height/2)
            mask = xmask & ymask
            
        if exclude:
            mask = ~mask

        masked_data = np.where(mask, data[0, 0], np.nan)
        masked_data = masked_data.reshape(1, 1, masked_data.shape[0], masked_data.shape[1])
        masked_image = Spatialmap(header=self.header, data=masked_data)
        if preview: 
            masked_image.view_region(region, **kwargs)
        if inplace:
            self.data = masked_data
        return masked_image
    
    def view_region(self, region, color="skyblue", lw=1., ls="--", plot=True, **kwargs):
        """
        This method allows users to plot the specified region on the image.
        Parameters:
            region (Region): the region to be plotted
            color (str): the color of the line representing the region
            lw (float): the width of the line representing the region
            ls (str): the type of the line representing the region
            plot (bool): True to show the plot. False to only return the plot.
        Returns:
            The intensity map with the annotated region.
        """
        ax = self.imview(plot=False, **kwargs)

        if isinstance(region, Region):
            region = self.__readregion(region)
            if region.shape == "circle":
                xsmooth = np.linspace(region.center[0]-region.radius, region.center[0]+region.radius, 1000)
                inside_circle = (np.square(xsmooth-region.center[0]) <= region.radius**2)
                ysmooth = np.sqrt(region.radius**2-np.square(xsmooth[inside_circle]-region.center[0]))

                # Plot the upper and lower semicircles
                ax.plot(xsmooth[inside_circle], region.center[1]+ysmooth, color=color, lw=lw, linestyle=ls)
                ax.plot(xsmooth[inside_circle], region.center[1]-ysmooth, color=color, lw=lw, linestyle=ls)
            elif region.shape == "ellipse":
                theta = np.linspace(0, 2*np.pi, 1000)
                angle = np.deg2rad(-region.pa-270)

                # Ellipse equation before rotation
                x_ellipse = region.center[0] + region.semimajor * np.cos(theta)
                y_ellipse = region.center[1] + region.semiminor * np.sin(theta)

                # Applying rotation for position angle
                x_rotated = region.center[0]+(x_ellipse-region.center[0])*np.cos(angle)-(y_ellipse-region.center[1])*np.sin(angle)
                y_rotated = region.center[1]+(x_ellipse-region.center[0])*np.sin(angle)+(y_ellipse-region.center[1])*np.cos(angle)

                # Plot the ellipse
                ax.plot(x_rotated, y_rotated, color=color, lw=lw, linestyle=ls)
            elif region.shape == "line":
                ax.plot([region.start[0], region.end[0]], [region.start[1], region.end[1]], 
                         color=color, lw=lw, linestyle=ls)
            elif region.shape == "box":
                # make patch
                pa = region.pa
                pa_rad = np.deg2rad(pa)
                center = region.center
                width, height = region.width, region.height
                dx, dy = -width/2, -height/2
                x = center[0] + (dx*np.cos(pa_rad) + dy*np.sin(pa_rad))
                y = center[1] - (dx*np.sin(pa_rad) - dy*np.cos(pa_rad))
                rect_patch = patches.Rectangle((x, y), width, height, angle=-region.pa,
                                               linewidth=lw, linestyle=ls, edgecolor=color, 
                                               facecolor='none')
                ax.add_patch(rect_patch)

        elif hasattr(region, "__iter__"):
            region_list = region
            for region in region_list:
                region = self.__readregion(region)
                if region.shape == "circle":
                    xsmooth = np.linspace(region.center[0]-region.radius, region.center[0]+region.radius, 1000)
                    inside_circle = (np.square(xsmooth-region.center[0]) <= region.radius**2)
                    ysmooth = np.sqrt(region.radius**2-np.square(xsmooth[inside_circle]-region.center[0]))

                    # Plot the upper and lower semicircles
                    ax.plot(xsmooth[inside_circle], region.center[1]+ysmooth, color=color, lw=lw, linestyle=ls)
                    ax.plot(xsmooth[inside_circle], region.center[1]-ysmooth, color=color, lw=lw, linestyle=ls)
                elif region.shape == "ellipse":
                    theta = np.linspace(0, 2*np.pi, 1000)
                    angle = np.deg2rad(-region.pa-270)

                    # Ellipse equation before rotation
                    x_ellipse = region.center[0] + region.semimajor * np.cos(theta)
                    y_ellipse = region.center[1] + region.semiminor * np.sin(theta)

                    # Applying rotation for position angle
                    x_rotated = region.center[0]+(x_ellipse-region.center[0])*np.cos(angle)-(y_ellipse-region.center[1])*np.sin(angle)
                    y_rotated = region.center[1]+(x_ellipse-region.center[0])*np.sin(angle)+(y_ellipse-region.center[1])*np.cos(angle)

                    # Plot the ellipse
                    ax.plot(x_rotated, y_rotated, color=color, lw=lw, linestyle=ls)
                elif region.shape == "line":
                    ax.plot([region.start[0], region.end[0]], [region.start[1], region.end[1]], 
                             color=color, lw=lw, linestyle=ls)
                elif region.shape == "box":
                    # make patch
                    pa = region.pa
                    pa_rad = np.deg2rad(pa)
                    center = region.center
                    width, height = region.width, region.height
                    dx, dy = -width/2, -height/2
                    x = center[0] + (dx*np.cos(pa_rad) + dy*np.sin(pa_rad))
                    y = center[1] - (dx*np.sin(pa_rad) - dy*np.cos(pa_rad))
                    rect_patch = patches.Rectangle((x, y), width, height, angle=-region.pa,
                                                   linewidth=lw, linestyle=ls, edgecolor=color, 
                                                   facecolor='none')
                    ax.add_patch(rect_patch)

        if plot:
            plt.show()

        return ax

    def pad(self, new_xlim=None, new_ylim=None, 
            new_imsize=None, left_pad=None, right_pad=None, 
            top_pad=None, bottom_pad=None, mode="constant",
            fill_value=np.nan, inplace=False):
        """
        Adds padding to the image data along specified axes and adjusts the reference coordinates accordingly.

        Parameters:
            new_xlim (array-like, optional): New limits for the x-axis (RA), specified as a two-element array [min, max].
                                             The method will calculate the required padding to achieve these limits.
                                             Default is None.
            
            new_ylim (array-like, optional): New limits for the y-axis (Dec), specified as a two-element array [min, max].
                                             The method will calculate the required padding to achieve these limits.
                                             Default is None.
            
            new_imsize (tuple of int, optional): New image size as a tuple (nx, ny), where nx is the new size for the x-axis
                                                 and ny is the new size for the y-axis. The method will calculate the padding
                                                 required to reach these dimensions. Default is None.
            
            left_pad (int, optional): Number of pixels to pad on the left side of the x-axis. Defaults to None, which implies
                                      no padding.
            
            right_pad (int, optional): Number of pixels to pad on the right side of the x-axis. Defaults to None, which implies
                                       no padding.
            
            top_pad (int, optional): Number of pixels to pad on the top side of the y-axis. Defaults to None, which implies
                                     no padding.
            
            bottom_pad (int, optional): Number of pixels to pad on the bottom side of the y-axis. Defaults to None, which
                                        implies no padding.
            
            fill_value (float, optional): Value to fill the padding region. Default is NaN.
            
            inplace (bool, optional): If True, modify the image data in place and return self. If False, return a new instance
                                      with the modified data. Default is False.
        
        Returns:
            self or new instance: The modified image data with added padding. Returns self if inplace=True; otherwise,
                                  returns a new instance with the padded data.

        Raises:
            ValueError: If any padding value is negative or if no valid padding, limits, or image size are specified.

        Warnings:
            If the padding difference for a new image size is an odd number, a warning is raised, indicating potential 
            uneven padding.

        Example usage:
            # Assuming `image` is an instance with the method `pad`

            # Add padding to achieve new axis limits
            new_image = image.pad(new_xlim=[10, 50], new_ylim=[-20, 30])

            # Add padding to achieve a new image size
            new_image = image.pad(new_imsize=(100, 200))

            # Add specific padding to each side
            new_image = image.pad(left_pad=5, right_pad=5, top_pad=10, bottom_pad=10)

            # Add padding in place
            image.pad(left_pad=5, right_pad=5, inplace=True)

        Notes:
            This method modifies the reference coordinates (RA, Dec) of the image data to account for the added padding.
            The new reference coordinates are recalculated based on the new position of the reference pixels.
        """
        # get attributes
        xaxis = self.xaxis
        yaxis = self.yaxis
        dx = self.dx 
        dy = self.dy
        
        # check whether parameters are specified:
        params = (new_xlim, new_ylim, new_imsize, left_pad, right_pad, top_pad, bottom_pad)
        if all(param is None for param in params):
            raise ValueError("You must specify at least one of the following: " + \
                             "new_xlim, new_ylim, new_imsize, or any of the padding " + \
                             "values (left_pad, right_pad, top_pad, bottom_pad).")
        
        if new_imsize is not None:
            x_diff = new_imsize[0] - xaxis.size
            if x_diff % 2:  # if difference is odd number
                warnings.warn("Padding difference for x-axis is an odd number; " + \
                              "this may lead to uneven padding.")
                left_pad = x_diff // 2 + 1
            else:
                left_pad = x_diff // 2
            right_pad = x_diff // 2
            
            y_diff = new_imsize[1] - yaxis.size
            if y_diff % 2:  # if difference is odd number
                warnings.warn("Padding difference for y-axis is an odd number; " + \
                              "this may lead to uneven padding.")
                bottom_pad = y_diff // 2 + 1
            else:
                bottom_pad = y_diff // 2 
            top_pad = y_diff // 2
        
        if new_xlim is not None:
            new_xlim = np.array(new_xlim)
            left_pad = (new_xlim.max() - xaxis.max())/-dx
            left_pad = _true_round_whole(left_pad)
            right_pad = (xaxis.min() - new_xlim.min())/-dx
            right_pad = _true_round_whole(right_pad)
            
        if new_ylim is not None:
            new_ylim = np.array(new_ylim)
            top_pad = (new_ylim.max() - yaxis.max())/dy
            top_pad = _true_round_whole(top_pad)
            bottom_pad = (yaxis.min() - new_ylim.min())/dy
            bottom_pad = _true_round_whole(bottom_pad)
        
        # set to default values (0) if not specified
        if left_pad is None:
            left_pad = 0
        if right_pad is None:
            right_pad = 0
        if top_pad is None:
            top_pad = 0
        if bottom_pad is None:
            bottom_pad = 0
            
        # raise value error if number of pad is negative
        if any(num < 0 for num in (top_pad, bottom_pad, left_pad, right_pad)):
            raise ValueError("Padding values cannot be negative.")
            
        # add padding to new data
        pad_width = ((0, 0), (bottom_pad, top_pad), (left_pad, right_pad))
        new_data = np.pad(self.data[0],  # three-dimensional
                          pad_width=pad_width, 
                          mode="constant", 
                          constant_values=fill_value)
        new_data = new_data[np.newaxis, :, :, :]  # add stokes axis 
        
        # calculate new cell sizes and reference pixels
        new_nx = new_data.shape[3]
        new_refnx = new_nx // 2 + 1
        new_ny = new_data.shape[2]
        new_refny = new_ny // 2 + 1
        
        # calculate new reference coordinate
        current_refcoord = SkyCoord(self.refcoord)
        current_ra = current_refcoord.ra
        current_dec = current_refcoord.dec

        new_ra = current_ra + (right_pad-left_pad+1)*dx*u.Unit(self.unit)
        new_dec = current_dec + (top_pad-bottom_pad+1)*dy*u.Unit(self.unit)
        new_refcoord = SkyCoord(ra=new_ra, dec=new_dec).to_string("hmsdms")
        
        # update stuff
        image = self if inplace else self.copy()
        image.data = new_data
        image.overwrite_header(nx=new_nx, refnx=new_refnx, 
                               ny=new_ny, refny=new_refny,
                               refcoord=new_refcoord,
                               shape=new_data.shape,
                               )
        return image

    def imregrid(self, template, interpolation="linear", inplace=False):
        """
        Regrids the image to match the grid of a template image.

        This method adjusts the image to have the same coordinate grid as the template image by shifting, trimming, and/or padding
        the image data, and then regridding the data using interpolation.

        Parameters:
            template (Spatialmap or Datacube): The template image whose grid will be matched. Must be an instance of Spatialmap or Datacube.
            
            interpolation (str, optional): The interpolation method to use for regridding. Options include "linear", "nearest", and "cubic". 
                                           Default is "linear".
            
            inplace (bool, optional): If True, modifies the image in place and returns self. If False, returns a new instance with the regridded data.
                                      Default is False.

        Returns:
            self or new instance: The regridded image. Returns self if inplace=True; otherwise, returns a new instance with the regridded data.

        Raises:
            ValueError: If the template is not an instance of Spatialmap or Datacube.
            Exception: If inconsistent padding and trimming conditions are detected.

        Example usage:
            # Assuming `image` is an instance with the method `imregrid` and `template` is a Spatialmap or Datacube instance

            # Regrid the image to match the template
            new_image = image.imregrid(template)

            # Regrid the image with cubic interpolation and modify in place
            image.imregrid(template, interpolation="cubic", inplace=True)
        
        Notes:
            - The method first shifts the image to the new reference coordinate of the template.
            - The image is then trimmed or padded to match the limits and size of the template.
            - Finally, the image data is regridded using the specified interpolation method to match the template grid.
        """
        image = self if inplace else self.copy()

        if not isinstance(template, (Spatialmap, Datacube)):
            raise ValueError("Template must be an instance of 'Spatialmap' or 'Datacube'.")

        new_dx = template.dx
        new_dy = template.dy 

        if self.unit != template.unit:
            conversion_factor = (1*u.Unit(template.unit)).to_value(self.unit)
            new_dx *= conversion_factor
            new_dy *= conversion_factor

        # first shift the image to new coordinate
        image.imshift(template, inplace=True)

        # change cell size of data through interpolation
        image = _change_cell_size(image=image,
                                  dx=new_dx,
                                  dy=new_dy,
                                  interpolation=interpolation,
                                  inplace=True)

        # trim/pad limits of the image
        image = _match_limits(image=image, template_image=template, inplace=True)
        
        return image
    
    def normalize(self, valrange=None, template=None, inplace=False):
        """
        Normalize the data of the spatial map to a specified range.
        Parameters:
            valrange (list/tuple): The range [min, max] to which the data will be normalized. Default is [0, 1].
            inplace (bool): If True, modify the data in-place. 
                            If False, return a new Spatialmap instance. Default is False.
        Returns:
            The normalized image, either as a new instance or the same instance based on the 'inplace' parameter.
        """
        if template is None:
            if valrange is None:
                valrange = [0, 1]
        else:
            valrange = [np.nanmin(template), np.nanmax(template)]
        
        # Extracting min and max from the range
        min_val, max_val = valrange

        # Normalization formula: (data - min(data)) / (max(data) - min(data)) * (max_val - min_val) + min_val
        data_min, data_max = np.nanmin(self.data), np.nanmax(self.data)
        normalized_data = (self.data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val

        if inplace:
            self.data = normalized_data
            return self
        return Spatialmap(header=self.header, data=normalized_data)
    
    def imsmooth(self, bmaj=None, bmin=None, bpa=0, width=None, unit=None, kernel="gauss",
                 fft=True, preview=True, inplace=False, **kwargs):
        """
        Perform convolution on the image.
        Parameters:
            bmaj (float): the major axis of the new beam.
            bmin (float): the minor axis of the new beam. None to set it to the same as the major axis.
            bpa (float): the position angle of the elliptical beam
            unit (float): the unit of bmaj/bmin. None to use default unit of axes.
            kernel (str): the type of convolution to be used. This parameter is case-insensitive.
                          "gauss", "gaussian", "g": Gaussian convolution
                          "box", "b": Box shape (square).
                            
            fft (bool): True to apply fast fourier transform.
            preview (bool): True to visualize convolved image.
            inplace (bool): True to modify current image in-place. False to return new convolved image.
        Returns: 
            The final convolved image (Spatialmap)
        """
        # initialize parameters
        unit = self.unit if unit is None else unit    # use the unit of this image as default
        kernel = kernel.lower()                       # make it case-insensitive
        
        # choose kernel and input parameters
        if kernel in ["gauss", "gaussian", "g"]:
            bmin = bmaj if bmin is None else bmin
            bmaj = u.Quantity(bmaj, unit).to(u.arcsec).value
            bmin = u.Quantity(bmin, unit).to(u.arcsec).value
            xsigma = bmaj / (np.abs(self.dx)*2*np.sqrt(2*np.log(2)))
            ysigma = bmin / (np.abs(self.dy)*2*np.sqrt(2*np.log(2)))
            k = Gaussian2DKernel(x_stddev=xsigma, y_stddev=ysigma, theta=np.deg2rad(-(90-bpa)))
        elif kernel in ["box", "b"]:
            width = bmaj if width is None else width
            width = u.Quantity(width, unit) / u.Quantity(np.abs(self.dx), self.unit)
            k = Box2DKernel(width=width)
            newbeam = None
        else:
            raise ValueError("'kernel' must be 'gauss' or 'box'.")
        
        # start convolving
        print("Convolving...")
        if fft: 
            newimage = convolve_fft(self.data[0, 0], kernel=k)
        else:
            newimage = convolve(self.data[0, 0], kernel=k)
        
        # update the beam dimensions
        newbeam = [bmaj, bmin, bpa]
        new_header = copy.deepcopy(self.header)
        new_header["beam"] = newbeam
        
        # modify the image in-place if necessary
        if inplace:
            self.data[0, 0] = newimage
            self.header = new_header
            self.__updateparams()
        
        # create new object with the new header and data
        convolved_image = Spatialmap(data=newimage.reshape((1, 1, self.ny, self.nx)), header=new_header)
        if preview:
            convolved_image.imview(title="Convolved image", **kwargs)
        
        # return new object
        if inplace:
            return self
        return convolved_image
        
    def imfit(self, region=None, threshold=None, estimates=None, preview=True, 
              plt_residual=False, shiftcenter=False, fov=None, **kwargs):
        """
        This method performs 2D Gaussian fitting on the image domain. 
        Parameters:
            region: the elliptical/circular region on which the fitting will be performed. 
                    Set to 'None' to use the entire region.
            threshold: the threshold above which the fitting will be performed. 
            estimates: the initial estimates of the parameters (x, y, amplitude, fwhmx, fwhmy, pa). 
                       None to use defaults (see notes below).
            preview (bool): True to preview the fitting results.
            plt_residual (bool): True to plot the residual map.
            shiftcenter (bool): True to shift the preview image to the Gaussian center.
        Returns (tuple):
            popt (list(float)): the list of best-fit parameters.
            perr (list(float)): the 1 sigma errors associated with the best-fit parameters.
            model_image (Spatialmap): the model image.
            residual_image (Spatialmap): the residual image.
        ------------
        Additional notes:
            The initial estimates are set to the following:
                - central position: the position of the peak value
                - amplitude: the maximum value 
                - FWHMs: Twice the shortest distance from the peak position where the intensity drops 
                         below half the maximum value. The FWHMs in the x and y directions are set to 
                         the same value.
                - PA: If the region is not specifed: set to 0 deg as the initial estimate.
                      Otherwise, it is set to the same pa as the region. 
                      The fitting bounds are set to -90 to 90 deg.
        """
        if region is not None:
            inp_region = copy.deepcopy(region)
            if inp_region.header["relative"]:
                inp_region.header["center"] = inp_region.center = _relative2icrs(inp_region.center, 
                                                                                 ref=self.refcoord)
                inp_region.header["relative"] = inp_region.relative = False
            region = self.__readregion(region)
            if region.shape == 'line':
                raise Exception(f"Region shape cannot be {region.shape}.")
            fitting_data = self.mask_region(region, preview=False).data[0, 0]
        else:
            fitting_data = self.data[0, 0]
        if threshold is not None:
            fitting_data = np.where(fitting_data<threshold, np.nan, fitting_data)
        xx, yy = self.get_xyaxes(grid=True)
        
        if estimates is None:
            maxindicies = np.unravel_index(np.nanargmax(fitting_data), shape=fitting_data.shape)
            centerx_est, centery_est = xx[maxindicies], yy[maxindicies]
            amp_est = fitting_data[maxindicies]
            halfindicies = np.where(fitting_data<=amp_est/2)
            halfx, halfy = xx[halfindicies], yy[halfindicies]
            fwhm_est = 2*np.sqrt((halfx-centerx_est)**2+(halfy-centery_est)**2).min()
            pa = 0. if region is None else region.pa
            estimates = [centerx_est, centery_est, amp_est, fwhm_est, fwhm_est, pa]
        
        # Perform the fitting
        sigma2fwhm = (2*np.sqrt(2*np.log(2)))
        g_init = models.Gaussian2D(x_mean=estimates[0],
                                   y_mean=estimates[1],
                                   amplitude=estimates[2],
                                   x_stddev=estimates[3]/sigma2fwhm, 
                                   y_stddev=estimates[4]/sigma2fwhm,
                                   theta=np.deg2rad(estimates[5]),
                                   )
        fit_g = fitting.LevMarLSQFitter()
        mask = ~np.isnan(fitting_data)
        g = fit_g(g_init, xx[mask], yy[mask], fitting_data[mask])
        popt = [g.x_mean.value, g.y_mean.value, g.amplitude.value, 
                g.x_stddev.value*sigma2fwhm, g.y_stddev.value*sigma2fwhm, np.rad2deg(-g.theta.value)]
        amp_err, x0_err, y0_err, xsig_err, ysig_err, theta_err = np.sqrt(np.diag(fit_g.fit_info['param_cov']))
        perr = [x0_err, y0_err, amp_err, xsig_err*sigma2fwhm, ysig_err*sigma2fwhm, np.rad2deg(theta_err)]
        
        # Calculate the fitted model
        fitted_model = g(xx, yy).reshape(1, 1, self.ny, self.nx)     
        model_image = Spatialmap(header=self.header, data=fitted_model)
        
        # Print results
        coord = _relative2icrs(coord=(popt[0], popt[1]), ref=self.refcoord, unit=self.unit)
        coord = SkyCoord(coord, unit=(u.hourangle, u.deg), frame='icrs')
        center_J2000 = coord.ra.to_string(unit=u.hourangle, precision=3) + \
                        " " + coord.dec.to_string(unit=u.deg, precision=2)
        total_flux = np.nansum(fitted_model)
        print(15*"#"+"2D Gaussian Fitting Results"+15*"#")
        if not shiftcenter:
            print(f"center: ({popt[0]:.4f} +/- {perr[0]:.4f}, {popt[1]:.4f} +/- {perr[1]:.4f}) [arcsec]")
        print("center (J2000): " + center_J2000)
        print(f"amplitude: {popt[2]:.4f} +/- {perr[2]:.4f} [{(self.bunit).replace('.', ' ')}]")
        print(f"total flux: {total_flux:.4f} [{(self.bunit).replace('.', ' ')}]")
        print(f"FWHM: {popt[3]:.4f} +/- {perr[3]:.4f} x {popt[4]:.4f} +/- {perr[4]:.4f} [arcsec]")

        # calculate deconvolved size:
        deconvolved_x = np.sqrt(popt[3]**2 - _get_beam_dim_along_pa(*self.beam, popt[5]+90)**2)
        deconvolved_y = np.sqrt(popt[4]**2 - _get_beam_dim_along_pa(*self.beam, popt[5])**2)
        print(f"Deconvolved size: {deconvolved_x:.4f} x {deconvolved_y:.4f} [arcsec]")

        print(f"P.A.: {popt[5]:.4f} +/- {perr[5]:.4f} [deg]")
        print(57*"#")
        print()
        regionfov = 0. if region is None else region.semimajor
        fov = np.max([popt[3], popt[4], regionfov])*2 if fov is None else fov
        if preview:
            # shift image to center
            if shiftcenter:
                centered_image = self.imshift(center_J2000, printcc=False, inplace=False)
            else:
                centered_image = self.copy()
            if region is not None:
                ax = centered_image.view_region(inp_region, fov=fov, title="Fitting result", 
                                      scalebaron=False, plot=False, **kwargs)
            else:
                ax = centered_image.imview(fov=fov, title="Fitting result", scalebaron=False, 
                                           plot=False, **kwargs)
            
            # parameters
            centerx, centery = (0., 0.) if shiftcenter else (popt[0], popt[1])
            fwhmx_plot, fwhmy_plot = popt[3], popt[4]
            pa_rad = -np.deg2rad(popt[5])

            # Add lines for major and minor axes
            ax.plot([centerx-fwhmx_plot*np.cos(pa_rad), centerx+fwhmx_plot*np.cos(pa_rad)],
                    [centery-fwhmx_plot*np.sin(pa_rad), centery+fwhmx_plot*np.sin(pa_rad)],
                    color='b', linestyle='--', lw=1.)  # Major axis
            ax.plot([centerx-fwhmy_plot*np.sin(pa_rad), centerx+fwhmy_plot*np.sin(pa_rad)],
                    [centery+fwhmy_plot*np.cos(pa_rad), centery-fwhmy_plot*np.cos(pa_rad)],
                    color='g', linestyle='--', lw=1.)  # Minor axis
            plt.show()
        
        # residual
        residual_data = fitting_data - fitted_model
        residual_image = Spatialmap(header=self.header, data=residual_data)
        effective_radius = np.sqrt(popt[3]*popt[4])
        if shiftcenter:
            residual_image = residual_image.imshift(center_J2000, printcc=False)
            model_image = model_image.imshift(center_J2000, printcc=False)
        if plt_residual:
            residual_image.imview(title="Residual Plot", scalebaron=False, fov=fov, **kwargs)
        return popt, perr, model_image, residual_image
    
    def __readregion(self, region):
        region = copy.deepcopy(region)
        center = region.header["center"]
        isJ2000 = (isinstance(center, str) and not center.isnumeric()) \
                    or (len(center)==2 and isinstance(center[0], str) and isinstance(center[1], str)) 
        if isJ2000:
            newcenter = _icrs2relative(center, ref=self.refcoord, unit=region.header["unit"])
            region.center = region.header["center"] = (newcenter[0].value, newcenter[1].value)
            if region.shape == "line":
                newstart = _icrs2relative(region.start, ref=self.refcoord, unit=region.header["unit"])
                newend = _icrs2relative(region.end, ref=self.refcoord, unit=region.header["unit"])
                region.start = region.header["start"] = (newstart[0].value, newstart[1].value)
                region.end = region.header["end"] = (newend[0].value, newend[1].value)
                region.length = region.header["length"] = u.Quantity(region.length, region.header["unit"]).to_value(self.unit)
            elif region.shape == "ellipse":
                newsemimajor = u.Quantity(region.semimajor, region.header["unit"]).to_value(self.unit)
                newsemiminor = u.Quantity(region.semiminor, region.header["unit"]).to_value(self.unit)
                region.semimajor = region.header["semimajor"] = newsemimajor
                region.semiminor = region.header["semiminor"] = newsemiminor
            elif region.shape == "circle":
                newradius = u.Quantity(region.radius, region.header["unit"]).to_value(self.unit)
                region.radius = region.semimajor = region.semiminor = region.header["radius"] = newradius
            elif region.shape == "box":
                height = u.Quantity(region.height, region.header["unit"]).to_value(self.unit)
                region.height = region.header["height"] = height
                width = u.Quantity(region.width, region.header["unit"]).to_value(self.unit)
                region.width = region.header["width"] = width
        elif not region.header["relative"]:
            refx, refy = _icrs2relative(self.refcoord, unit=self.unit)
            centerx, centery = u.Quantity(center[0], region.header["unit"]), u.Quantity(center[1], region.header["unit"])
            newcenter = u.Quantity(centerx-refx, self.unit).value, u.Quantity(centery-refy, self.unit).value
            region.center = region.header["center"] = newcenter
            if region.shape == "ellipse":
                newsemimajor = u.Quantity(region.semimajor, region.header["unit"]).to_value(self.unit)
                newsemiminor = u.Quantity(region.semiminor, region.header["unit"]).to_value(self.unit)
                region.semimajor = region.header["semimajor"] = newsemimajor
                region.semiminor = region.header["semiminor"] = newsemiminor
            elif region.shape == "circle":
                newradius = u.Quantity(region.radius, region.header["unit"]).to_value(self.unit)
                region.radius = region.semimajor = region.semiminor = region.header["radius"] = newradius
            elif region.shape == "line":
                start, end = region.start, region.end
                startx, starty = u.Quantity(start[0], region.header["unit"]), u.Quantity(start[1], region.header["unit"])
                newstart = u.Quantity(startx-refx, self.unit).value, u.Quantity(starty-refy, self.unit).value
                region.start = region.header["start"] = newstart
                endx, endy = u.Quantity(end[0], region.header["unit"]), u.Quantity(end[1], region.header["unit"])
                newend = u.Quantity(endx-refx, self.unit).value, u.Quantity(endy-refy, self.unit).value
                region.end = region.header["end"] = newend
                region.length = region.header["length"] = u.Quantity(region.length, region.header["unit"]).to_value(self.unit)
            elif region.shape == "box":
                height = u.Quantity(region.height, region.header["unit"]).to_value(self.unit)
                region.height = region.header["height"] = height
                width = u.Quantity(region.width, region.header["unit"]).to_value(self.unit)
                region.width = region.header["width"] = width
        region.relative = region.header["relative"] = True
        return region
    
    def set_threshold(self, threshold=None, minimum=True, inplace=False):
        """
        To mask the intensities above or below this threshold.
        Parameters:
            threshold (float): the value of the threshold. None to use three times the rms noise level.
            minimum (bool): True to remove intensities below the threshold. 
                            False to remove intensities above the threshold.
            inplace (bool): True to modify the data in-place. False to return a new image.
        Returns:
            Image with the masked data.
        """
        if threshold is None:
            threshold = 3*self.noise()
        if inplace:
            if minimum:
                self.data = np.where(self.data<threshold, np.nan, self.data)
            else:
                self.data = np.where(self.data<threshold, self.data, np.nan)
            return self
        if minimum:
            return Spatialmap(header=self.header, data=np.where(self.data<threshold, np.nan, self.data))
        return Spatialmap(header=self.header, data=np.where(self.data<threshold, self.data, np.nan))
    
    def beam_area(self, unit=None):
        """
        To calculate the beam area of the image.
        Parameters:
            unit (float): the unit of the beam area to be returned. 
                          Default (None) is to use same unit as the positional axes.
        Returns:
            The beam area.
        """
        bmaj = u.Quantity(self.bmaj, self.unit)
        bmin = u.Quantity(self.bmin, self.unit)
        area = np.pi*bmaj*bmin/(4*np.log(2))
        if unit is not None:
            area = area.to_value(unit)
        return area
    
    def pixel_area(self, unit=None):
        """
        To calculate the pixel area of the image.
        Parameters:
            unit (float): the unit of the pixel area to be returned.
                          None to use the same unit as the positional axes.
        Returns:
            The pixel area.
        """
        width = u.Quantity(np.abs(self.dx), self.unit)
        height = u.Quantity(np.abs(self.dy), self.unit)
        area = width*height
        if unit is not None:
            area = area.to_value(unit)
        return area
    
    def conv_unit(self, unit, inplace=True):
        """
        This method converts the axis unit of the image into the desired unit.
        Parameters:
            unit (str): the new axis unit.
            inplace (bool): True to update the current axes with the new unit. 
                            False to create a new image having axes with the new unit.
        Returns:
            A new image with axes of the desired unit.
        """
        newbeam = (u.Quantity(self.beam[0], self.unit).to_value(unit), 
                   u.Quantity(self.beam[1], self.unit).to_value(unit), 
                   self.bpa)
        if inplace:
            self.header["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
            self.header["dy"] = u.Quantity(self.dy, self.unit).to_value(unit)
            self.header["beam"] = newbeam
            self.header["unit"] = _apu_to_headerstr(_to_apu(unit))
            self.__updateparams()
            return self
        newheader = copy.deepcopy(self.header)
        newheader["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
        newheader["dy"] = u.Quantity(self.dy, self.unit).to_value(unit)
        newheader["beam"] = newbeam
        newheader["unit"] = _apu_to_headerstr(_to_apu(unit))
        return Spatialmap(header=newheader, data=self.data)
    
    def conv_bunit(self, bunit, inplace=True):
        """
        This method converts the brightness unit of the image into the desired unit.
        Parameters:
            bunit (str): the new unit.
            inplace (bool): True to update the current data with the new unit. 
                            False to create a new image having data with the new unit.
        Returns:
            A new image with the desired unit.
        """
        # string to astropy units 
        bunit = _to_apu(bunit)
        oldunit = _to_apu(self.bunit)
        
        # equivalencies
        equiv_bt = u.equivalencies.brightness_temperature(frequency=self.restfreq*u.Hz, 
                                                          beam_area=self.beam_area())
        equiv_pix = u.pixel_scale(u.Quantity(np.abs(self.dx), self.unit)**2/u.pixel)
        equiv_ba = u.beam_angular_area(self.beam_area())
        equiv = [equiv_bt, equiv_pix, equiv_ba]

        # factors
        factor_bt = (1*u.Jy/u.beam).to(u.K, equivalencies=equiv_bt) / (u.Jy/u.beam)
        factor_pix = (1*u.pixel).to(u.rad**2, equivalencies=equiv_pix) / (u.pixel)
        factor_ba = (1*u.beam).to(u.rad**2, equivalencies=equiv_ba) / (u.beam)
        factor_pix2bm = factor_pix / factor_ba
        factor_Jypix2K = factor_pix2bm / factor_bt
        factor_Jysr2K = factor_bt*factor_ba
        factors = [factor_bt, factor_pix, factor_ba, factor_pix2bm, 
                   factor_Jypix2K, factor_Jysr2K]
        
        # convert
        if isinstance(self.data, u.Quantity):
            olddata = self.data.copy()
        else:
            olddata = u.Quantity(self.data, oldunit)
        newdata = _convert_Bunit(olddata, newunit=bunit, 
                                 equivalencies=equiv, factors=factors)
        
        if newdata is None:
            raise UnitConversionError(f"Failure to convert intensity unit to {_apu_to_headerstr(bunit)}")

        # return and set values
        if not isinstance(self.data, u.Quantity):
            newdata = newdata.value
            
        newimage = self if inplace else self.copy()
        newimage.data = newdata
        newimage.header["bunit"] = _apu_to_headerstr(bunit)
        newimage.__updateparams()
        return newimage
            
    def imstat(self, region=None, exclude=False, preview=True):
        """
        A method to see the relevant statistics of a region. 
        Parameters:
            region (Region): The region of which the statistics will be viewed. 
                             None to use entire image as region.
            exclude (bool): True to indicate 'outside' the region. 
                            False to indicate 'inside' the region.
            preview (bool): True to view the region on the image.
        Returns:
            numpixels (int): the number of pixels in the region
            summation (float): the sum of all pixels in the region
            mean (float): the mean of all pixels in the region
            stddev (float): the standard deviation of all pixels in the region
            minimum (float): the minimum value of the region
            maximum (float): the maximum value of the region
            rms (float): the root-mean-square average of the pixels in the region
            sumsq (float): the sum of the squares of pixels in the region
        """
        if region is None:
            stat_image = copy.deepcopy(self) 
            fov = None
        else:
            region = self.__readregion(region)
            fov = np.sqrt(region.semimajor*region.semiminor)  # also applies for circle
            stat_image = copy.deepcopy(self).mask_region(region, exclude=exclude, preview=preview, fov=1.5*fov)
        statdata = stat_image.data[0, 0]
        
        numpixels = statdata[~np.isnan(statdata)].size
        summation = np.nansum(statdata)
        mean = np.nanmean(statdata)
        stddev = np.nanstd(statdata)
        minimum = np.nanmin(statdata)
        maximum = np.nanmax(statdata)
        rms = np.sqrt(np.nanmean(statdata**2))
        sumsq = np.nansum(statdata**2)
        params = (numpixels, summation, mean, stddev, minimum, maximum, rms, sumsq)
        
        # print info
        bunit = self.bunit.replace(".", " ")
        print(15*"#" + "Imstat Info" + 15*"#")
        print("NumPixels: %.0f [pix]"%numpixels)
        print(f"Sum: {summation:.6e} [{bunit}]")
        print(f"Mean: {mean:.6e} [{bunit}]")
        print(f"StdDev: {stddev:.6e} [{bunit}]")
        print(f"Min: {minimum:.6e} [{bunit}]")
        print(f"Max: {maximum:.6e} [{bunit}]")
        print(f"RMS: {rms:.6e} [{bunit}]")
        print(f"SumSq: {sumsq:.6e} [({bunit})^2]")
        print(41*"#")
        print()
        
        return params
    
    def get_xyprofile(self, region, width=1, xlabel=None, ylabel=None, 
                      returndata=False, **kwargs):
        """
        Method to get the spatial profile of the image along a line.
        Parameters:
            region (Region): the region representing the line.
            width (float): the averaging width [pix]. Default is to use 1 pixel.
            xlabel (str): the label on the xaxis. Defualt (None) is to use 'Offset (unit)'.
            ylabel (str): the label on the yaxis. Default (None) is to use the 'Intensity (unit)'.
            returndata (bool): True to return the data of the spatial profile. 
                               False to return the plot.
        Returns:
            xdata (np.ndarray, if returndata=False): the x coordinates of the profile data
            ydata (np.ndarray, if returndata=False): the y coordinates of the profile data
            ax (matplotlib.axes.Axes, if returndata=False): the image of the plot
        """
        # raise exception if width is an even number of pixels.
        if width % 2 == 0:
            raise ValueError("The parameter 'width' must be an odd positive integer.")
        
        # initialize parameters
        region = self.__readregion(region)
        center = region.center
        length = region.length
        pa = region.pa
        shifted_image = self.shift(center, inplace=False, printcc=False) if center != (0, 0) else self.copy()
        rotated_image = shifted_image.rotate(pa, ccw=False, inplace=False) if pa != 0 else shifted_image
        rot_data = rotated_image.data
        
        # crop image to fit length
        addidx = int(width // 2)
        argmin = np.where(self.xaxis==np.min(np.abs(self.xaxis)))[0][0]
        xidx1, xidx2 = argmin-addidx, argmin+addidx+1
        yindices = np.where(np.abs(self.yaxis) <= length/2)[0]
        yidx1, yidx2 = yindices[0], yindices[-1]+1
        xdata = self.yaxis[yidx1:yidx2]
        ydata = rot_data[:, :, xidx1:xidx2, :]
        ydata = ydata[:, :, :, yidx1:yidx2][0, 0, 0]
        
        # plot data
        if xlabel is None:
            xlabel = f"Offset ({self.unit})"
        if ylabel is None:
            ylabel = f"Intensity ({_unit_plt_str(_apu_to_str(_to_apu(self.bunit)))})"
        ax = plt_1ddata(xdata, ydata, xlabel=xlabel, ylabel=ylabel, **kwargs)
        
        # returning
        if returndata:
            return xdata, ydata
        return ax
        
    def set_data(self, data, inplace=False):
        """
        This method allows users to assign a new dataset to the image.
        Parameters:
            data (ndarray): the new data to be set
            inplace (bool): True to modify the data of this image. False to only return a copy.
        Return: 
            An image with the specified dataset.
        """
        if data.shape != self.shape:
            data = data.reshape(self.shape)
        if inplace:
            self.data = data
            return self
        newimg = self.copy()
        newimg.data = data
        return newimg
    
    def noise(self, sigma=3., plthist=False, shownoise=False, printstats=False, bins='auto', 
              gaussfit=False, curvecolor="crimson", curvelw=0.6, fixmean=True, histtype='step', 
              linewidth=0.6, returnmask=False, **kwargs):
        """
        This method estimates the 1 sigma background noise level using sigma clipping.
        Parameters:
            sigma (float): sigma parameter of sigma clip
            plthist (bool): True to show background noise distribution as histogram 
                            (useful for checking if Gaussian)
            shownoise (bool): True to background noise spatial distribution
            printstats (bool): True to print information regarding noise statistics
            bins (int): The bin size of the histogram. Applicable if plthist or gaussfit
            gaussfit (bool): True to perform 1D gaussian fitting on noise distribution
            curvecolor (str): the color of the best-fit curve to be plotted, if applicable
            curvelw (float): the line width of the best-fit curve to be plotted, if applicable
            fixmean (bool): True to fix fitting parameter of mean to 0, if guassfit 
            histtype (str): the type of histogram
            linewidth (float): the line width of the histogram line borders
            returnmask (bool): True to return the mask (pixels that are considered 'noise')
        Returns:
            rms (float): the rms noise level
            mask (ndarray): the noise mask (pixels that are considered noise), if returnmask=True
        """
        bunit = self.bunit.replace(".", " ")
        
        clipped_data = sigma_clip(self.data, sigma=sigma, maxiters=10000, masked=False, axis=(2, 3))
        bg_image = self.set_data(clipped_data, inplace=False)
        mean = np.nanmean(clipped_data)
        rms = np.sqrt(np.nanmean(clipped_data**2))
        std = np.nanstd(clipped_data)
        
        if not rms:
            raise Exception("Sigma clipping failed. It is likely that most noise has been masked.")
        
        if printstats:
            print(15*"#" + "Noise Statistics" + 15*"#")
            print(f"Mean: {mean:.6e} [{bunit}]")
            print(f"RMS: {rms:.6e} [{bunit}]")
            print(f"StdDev: {std:.6e} [{bunit}]")
            print(46*"#")
            print()
        
        if gaussfit:
            if fixmean:
                def gauss(x, sigma, amp):
                    return amp*np.exp(-x**2/(2*sigma**2))
            else:
                def gauss(x, sigma, amp, mean):
                    return amp*np.exp(-(x-mean)**2/(2*sigma**2))
            flatdata = clipped_data[~np.isnan(clipped_data)]
            ydata, edges = np.histogram(flatdata, bins=bins)
            xdata = (edges[1:] + edges[:-1]) / 2  # set as midpoint of the bins
            p0 = [std, np.max(ydata)] if fixmean else [std, np.max(ydata), mean]
            popt, pcov = curve_fit(f=gauss, xdata=xdata, ydata=ydata, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            fitx = np.linspace(xdata.min(), xdata.max(), 1000)
            fity = gauss(fitx, popt[0], popt[1]) if fixmean else gauss(fitx, popt[0], popt[1], popt[2])
            
            print(15*"#" + "Gaussian Fitting Results" + 15*"#")
            if not fixmean:
                print(f"Mean: {popt[2]:.4e} +/- {perr[2]:.4e} [{bunit}]")
            print(f"StdDev: {popt[0]:.4e} +/- {perr[0]:.4e} [{bunit}]")
            print(f"Amplitude: {np.round(popt[1])} +/- {np.round(perr[1])} [pix]")
            print(54*"#")
            print()
            
        if plthist:
            q = _to_apu(self.bunit)
            flatdata = clipped_data[~np.isnan(clipped_data)]
            ax = plt_1ddata(flatdata, hist=True, 
                            xlabel=f"Intensity ({q:latex_inline})", ylabel="Pixels",
                            bins=bins, plot=False, xlim=[flatdata.min(), flatdata.max()], 
                            linewidth=linewidth, histtype=histtype, **kwargs)
            if gaussfit:
                ax.plot(fitx, fity, color=curvecolor, lw=curvelw)
                
        if shownoise:
            bg_image.imview(title="Background Noise Distribution", **kwargs)
            
        if plthist or shownoise:
            plt.show()
        
        if returnmask:
            mask = ~np.isnan(clipped_data)
            return rms, mask
        return rms
            
    def imview(self, contourmap=None, title=None, fov=None,  coloron=True, contouron=None,
               vmin=None, vmax=None, percentile=None, scale="linear", gamma=1.5, crms=None, 
               clevels=np.arange(3, 21, 3), tickscale=None, scalebaron=True, distance=None, 
               cbarlabelon=True, cbarlabel=None, xlabelon=True, ylabelon=True, center=(0., 0.), 
               dpi=600, ha="left", va="top", titleloc=(0.1, 0.9), cmap=None, fontsize=10, 
               cbarwidth="5%", width=330, height=300, smooth=None, scalebarsize=None, 
               nancolor=None, beamon=True, beamcolor=None, contour_beamcolor=None, 
               contour_beamon=None, beamloc=(0.1225, 0.1225), contour_beamloc=None, 
               ccolors=None, clw=0.8, txtcolor=None, cbaron=True, cbarpad=0., tickson=True, 
               labelcolor="k", tickcolor="k", labelsize=10., ticksize=3., tickwidth=None, 
               title_fontsize=12, cbartick_length=3., cbartick_width=None, scalebar_fontsize=10, 
               axeslw=1., scalecolor=None, scalelw=1., cbarloc="right", xlim=None, ylim=None, 
               cbarticks=None, cbartick_direction="in", vcenter=None, vrange=None, 
               aspect_ratio=1, barloc=(0.85, 0.15), barlabelloc=(0.85, 0.075), linthresh=1, 
               beam_separation=0.04, linscale=1, decimals=2, ax=None, savefig=None, plot=True):
        """
        Method to plot the 2D spatial map (moment maps, continuum maps, etc.).
        Parameters:
            contourmap (Spatialmap): the image to be plotted as contour. Set to "None" to not add a contour.
            title (str): the title text to be plotted
            fov (float): the field of view of the image
            vmin (float): the minimum value of the color bar
            vmax (float): the maximum value of the color bar
            scale (str): the scale of the colormap (linear/gamma/log)
            gamma (float): the scaling exponent if scale is set to gamma
            crms (float): the rms level to be based on for the contour
            clevels (float): the contour levels 
            tickscale (float): the increment of each tick. None to use default setting of matplotlib.
            scalebaron (bool): True to add a scale bar
            distance (float): the distance to the object in parsec. 
                              Default is None (meaning a scale bar will not be added.)
            cbarlabelon (bool): True to add a colorbar label
            cbarlabel (str): the label of the colorbar. Set to "None" to add unit same as that of the header.
            xlabelon (bool): True to add an x-axis label
            ylabelon (bool): True to add a y-axis label
            dpi (int): the dots per inch value of the image
            ha (str): horizontal alignment of title
            va (str): vertical alignment of title
            titleloc (float): relative position of title (x, y)
            cmap (str): the colormap of the image
            fontsize (int): the font size of the title
            cbarwidth (str): the width of the colorbar
            width (float): the width of the image
            height (float): the height of the image
            imsmooth (float): the Gaussian filter parameter to be added to smooth out the contour
            scalebarsize (float): the scale of the size bar in arcsec
            nancolor (str): the color of nan values represented in the colormap
            beamcolor (str): the color of the ellipse representing the beam dimensions
            ccolors (str): the colors of the contour lines
            clw (float): the line width of the contour 
            txtcolor (str): the color of the title text
            cbaron (bool): True to add a colorbar
            cbarpad (float): the distance between the main image and the color bar
            tickson (bool): True to add ticks to the xaxis and yaxis
            labelcolor (str): the font color of the labels
            tickcolor (str): the color of the ticks
            labelsize (float): the font size of the axis labels
            cbartick_length (float): the length of the color bar ticks
            cbartick_width (float): the width of the color bar 
            beamon (bool): True to add an ellipse that represents the beam dimensions
            scalebar_fontsize (float): the font size of the scale bar label
            axeslw (float): the line width of the borders
            scalecolor (str): the color of the scale bar
            scalelw (float): the line width of the scale bar
            orientation (str): the orientation of the color bar
            savefig (dict): list of keyword arguments to be passed to 'plt.savefig'.
            plot (bool): True to show the plot
        Returns:
            The 2D image ax object.
        """
        # set parameters to default values
        if cbartick_width is None:
            cbartick_width = axeslw
        
        if tickwidth is None:
            tickwidth = axeslw
        
        if cmap is None:
            if _to_apu(self.bunit).is_equivalent(u.Hz) or \
               _to_apu(self.bunit).is_equivalent(u.km/u.s):
                cmap = "RdBu_r"
            else:
                cmap = "inferno"
            
        if ccolors is None:
            if not coloron:
                ccolors = "k"
            elif cmap == "inferno":
                ccolors = "w"
            else:
                ccolors = "k"
            
        if beamcolor is None:
            if cmap == "inferno":
                beamcolor = "skyblue" 
            elif cmap == 'RdBu_r':
                beamcolor = "gray"
            else:
                beamcolor = "magenta"

        if contour_beamcolor is None:
            contour_beamcolor = beamcolor

        if txtcolor is None:
            txtcolor = "w" if cmap == "inferno" else "k"
            
        if nancolor is None:
            nancolor = 'k' if cmap == "inferno" else 'w'
            
        if scalecolor is None:
            scalecolor = 'w' if cmap == "inferno" else "k"
        
        if cbarticks is None:
            cbarticks = []
            
        if fov is None:
            fov = self.widestfov # set field of view to widest of image by default
        
        if xlim is None:
            xlim = [center[0]+fov, center[0]-fov]
        else:
            xlim = xlim[:]  # create copy
        
        if xlim[1] > xlim[0]:
            xlim[0], xlim[1] = xlim[1], xlim[0]  # swap
        
        if ylim is None:
            ylim = [center[1]-fov, center[1]+fov]
        else:
            ylim = ylim[:]
            
        if ylim[0] > ylim[1]:
            ylim[0], ylim[1] = ylim[1], ylim[0]  # swap
            
        if scalebarsize is None:
            # default value of scale bar size
            xrange = np.abs(xlim[1]-xlim[0])
            order_of_magnitude = int(np.log10(xrange))
            scalebarsize = 10**order_of_magnitude / 10
            scalebarsize = np.clip(scalebarsize, 0.1, 10)
                
        if tickscale is not None:
            ticklist = np.arange(-fov, fov+tickscale, tickscale)
            xticklist = ticklist + center[0]
            yticklist = ticklist + center[1]
        
        # convert contour levels to numpy array if necessary
        if not isinstance(clevels, np.ndarray):
            clevels = np.array(clevels)
        
        # copy data and convert data to unitless array, if necessary
        data = self.data
        if isinstance(data, u.Quantity):
            data = data.value
        
        # create a copy of the contour map
        if not coloron and contouron is None:
            contouron = True

        if contourmap is None:
            contourmap = self
            if contouron is None:
                contouron = False  # default if contour map is not specified
            if contour_beamon is None:
                contour_beamon = not coloron
        else:
            if contouron is None:
                contouron = True  # default if contour map is specified
            if contour_beamon is None:
                if coloron:
                    contour_beamon = contouron and (self.beam != contourmap.beam)
                else:
                    contour_beamon = contouron

        if contouron and isinstance(contourmap, Spatialmap):
            contourmap = contourmap.copy()
            if contourmap.refcoord != self.refcoord:
                contourmap = contourmap.imshift(self.refcoord, printcc=False)
            if contourmap.unit != self.unit:
                contourmap = contourmap.conv_unit(self.unit)
            
        if vcenter is not None:
            if vrange is None:
                dist = max([np.nanmax(data)-vcenter, vcenter-np.nanmin(data)])
                vmin = vcenter - dist
                vmax = vcenter + dist
            else:
                vmin = vcenter - vrange/2
                vmax = vcenter + vrange/2
        elif vrange is not None:
            vcenter = (np.nanmax(data) + np.nanmin(data))/2
            vmin = vcenter - vrange/2
            vmax = vcenter + vrange/2
        
        # determine vmin and vmax using percentile if still not specified
        if vmin is None and vmax is None and percentile is not None:
            clipped_data = self.__trim_data(xlim=xlim, ylim=ylim)
            vmin, vmax = clip_percentile(data=clipped_data, area=percentile)
            
        # change default parameters if ax is None
        if ax is None:
            ncols, nrows = 1, 1
            fig_width_pt  = width*ncols
            fig_height_pt = height*nrows
            inches_per_pt = 1.0/72.27                     # Convert pt to inch
            fig_width     = fig_width_pt * inches_per_pt  # width in inches
            fig_height    = fig_height_pt * inches_per_pt # height in inches
            fig_size      = [fig_width, fig_height]
            params = {'axes.labelsize': fontsize,
                      'axes.titlesize': fontsize,
                      'font.size' : fontsize,
                      'legend.fontsize': fontsize,
                      'xtick.labelsize': labelsize,
                      'ytick.labelsize': labelsize,
                      'xtick.top': True,   # draw ticks on the top side
                      'xtick.major.top' : True,
                      'figure.figsize': fig_size,
                      'font.family': _fontfamily,
                      'mathtext.fontset': _mathtext_fontset,
                      'mathtext.tt': _mathtext_tt,
                      'axes.linewidth' : axeslw,
                      'xtick.major.width' : 1.0,
                      "xtick.direction": 'in',
                      "ytick.direction": 'in',
                      'ytick.major.width' : 1.0,
                      'xtick.minor.width' : 0.,
                      'ytick.minor.width' : 0.,
                      'xtick.major.size' : 6.,
                      'ytick.major.size' : 6.,
                      'xtick.minor.size' : 0.,
                      'ytick.minor.size' : 0.,
                      'figure.dpi': dpi,
                    }
            rcParams.update(params)

        # set color map
        my_cmap = copy.deepcopy(mpl.colormaps[cmap]) 
        my_cmap.set_bad(color=nancolor) 
        
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
            plt.subplots_adjust(wspace=0.4)
        
        if contouron and crms is None:
            try:
                crms = contourmap.noise()
                bunit = self.bunit.replace(".", " ")
                print(f"Estimated base contour level (rms): {crms:.4e} [{bunit}]")
            except Exception:
                contourmap = None
                print("Failed to estimate RMS noise level of contour map.")
                print("Please specify base contour level using 'crms' parameter.")
                print("The contour map will not be plotted.")
        
        # add image
        if coloron:
            is_logscale: bool = (scale.lower() in ("log", "logscale", "logarithm", "symlog"))
            ax, climage = _plt_cmap(self, ax, data[0, 0], imextent=self.imextent, 
                                    cmap=my_cmap, vmin=vmin, vmax=vmax, scale=scale,
                                    gamma=gamma, linthresh=linthresh, linscale=linscale)
            
        if cbarlabel is None:
            cbarlabel = "(" + _unit_plt_str(_apu_to_str(_to_apu(self.bunit))) + ")"

        if contouron:
            contour_data = contourmap.data[0, 0]
            if smooth is not None:
                contour_data = ndimage.gaussian_filter(contour_data, smooth)
            ax.contour(contour_data, extent=contourmap.imextent, 
                       levels=crms*clevels, colors=ccolors, linewidths=clw, origin='lower')
            
        if title is not None:
            xrange = xlim[1]-xlim[0]
            yrange = ylim[1]-ylim[0]
            titlex = xlim[0] + titleloc[0]*xrange
            titley = ylim[0] + titleloc[1]*yrange
            ax.text(x=titlex, y=titley, s=title, ha=ha, va=va, 
                    color=txtcolor, fontsize=title_fontsize)
        
        # set field of view
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if tickscale is None:
            # use the default ticks if ticklist is not specified
            if center == (0., 0.):
                default_ticks = ax.get_xticks().copy()
                ax.set_xticks(default_ticks)
                ax.set_yticks(-default_ticks)
            else:
                default_xticks = ax.get_xticks().copy()
                d_xticks = np.abs(default_xticks[0] - default_xticks[1])
                default_yticks = ax.get_yticks().copy()
                d_yticks = np.abs(default_yticks[0] - default_yticks[1])
                if d_xticks != d_yticks:
                    miny = np.min(default_yticks)
                    maxy = np.max(default_yticks)
                    new_yticks = np.arange(miny, maxy+d_xticks, d_xticks)
                    tick_mask = (ylim[0] < new_yticks) & (new_yticks < ylim[1])
                    new_yticks = new_yticks[tick_mask]
                    ax.set_xticks(default_xticks)
                    ax.set_yticks(new_yticks)
                else:
                    ax.set_xticks(default_xticks)
                    ax.set_yticks(default_yticks)
        else:
            ax.set_xticks(ticklist)
            ax.set_yticks(ticklist)
            
        if tickson:
            ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True,
                           length=ticksize, colors=tickcolor, labelrotation=0, labelcolor=labelcolor, 
                           labelsize=labelsize, width=tickwidth)
        else:
            ax.tick_params(which='both', direction='in',bottom=False, top=False, left=False, right=False,
                           colors=tickcolor, labelrotation=0, labelcolor=labelcolor, labelsize=labelsize,
                           width=tickwidth)
            
        # set labels
        if xlabelon:
            ax.set_xlabel(f"Relative RA ({self.unit})", fontsize=fontsize)
        if ylabelon:
            ax.set_ylabel(f"Relative Dec ({self.unit})", fontsize=fontsize)
        
        # add beam
        if beamon and coloron:
            ax = self._addbeam(ax, xlim=xlim, ylim=ylim, beamcolor=beamcolor, 
                               filled=True, beamloc=beamloc)
        else:
            if contour_beamloc is None:
                contour_beamloc = beamloc

        if contour_beamon:
            if contour_beamloc is None:
                contour_beamloc = _get_contour_beam_loc(self.beam, contourmap.beam,
                                                        color_beamloc=beamloc, xlim=xlim, 
                                                        beam_separation=beam_separation)
            ax = contourmap._addbeam(ax, xlim=xlim, ylim=ylim, 
                                     beamcolor=contour_beamcolor,
                                     filled=False,
                                     beamloc=contour_beamloc)

        # add scale bar
        if scalebaron and distance is not None:
            ax = self.__add_scalebar(ax, xlim=xlim, ylim=ylim, distance=distance, 
                                     size=scalebarsize, scalecolor=scalecolor, scalelw=scalelw, 
                                     fontsize=scalebar_fontsize, barloc=barloc, txtloc=barlabelloc)
        
        # reset field of view
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # plot color bar and set aspect ratio  
        if not coloron:
            cbaron = False
            climage = None
            is_logscale = False
        ax = _plt_cbar_and_set_aspect_ratio(ax=ax, 
                                            climage=climage,
                                            cbarlabelon=cbarlabelon,
                                            cbarlabel=cbarlabel,
                                            cbarloc=cbarloc,
                                            cbaron=cbaron,
                                            cbarwidth=cbarwidth, 
                                            cbarpad=cbarpad,
                                            cbarticks=cbarticks,
                                            fontsize=fontsize, 
                                            labelsize=labelsize, 
                                            cbartick_width=cbartick_width, 
                                            cbartick_length=cbartick_length, 
                                            cbartick_direction=cbartick_direction, 
                                            aspect_ratio=aspect_ratio, 
                                            is_logscale=is_logscale, 
                                            decimals=decimals)
        
        # save figure if parameters were specified
        if savefig:
            plt.savefig(**savefig)
        
        # plot image
        if plot:
            plt.show()
            
        return ax

    def add_contour(self, ax, crms=None, clevels=np.arange(3, 21, 3), 
                    beamon=True, beamcolor=None, beamloc=None, 
                    ccolors="limegreen", smooth=None, clw=0.8, 
                    beam_separation=0.04, zorder=None, plot=False):
        """
        Add contour of data to a matplotlib image.

        This function adds a contour plot to an existing matplotlib axis. It does not take 
        into account the center of the original image, so make sure the center of the 
        contour aligns with your desired center.

        Parameters:
            ax (matplotlib.axes._axes.Axes): The matplotlib axis instance on which the 
                                             contour will be added.
            crms (float, optional): The base contour level. If not specified, it will be 
                                    estimated using the `self.noise()` method.
            clevels (array-like, optional): The contour levels to be used. Defaults to 
                                            np.arange(3, 21, 3).
            beamon (bool, optional): If True, a beam will be added to the plot. Defaults to True.
            beamcolor (str or None, optional): Color of the beam. If None, it will be set 
                                               to the same as `ccolors`. Defaults to None.
            beamloc (tuple or None, optional): Location of the beam. If None, it will be 
                                               calculated. Defaults to None.
            ccolors (str, optional): Color of the contour lines. Defaults to "limegreen".
            smooth (float or None, optional): Standard deviation for Gaussian kernel to smooth 
                                              the data. If None, no smoothing is applied. 
                                              Defaults to None.
            clw (float, optional): Line width of the contour lines. Defaults to 0.8.
            beam_separation (float, optional): Separation between beams if multiple are present. 
                                               Defaults to 0.04.
            plot (bool, optional): If True, `plt.show()` will be called to display the plot. 
                                   Defaults to False.

        Returns:
            ax (matplotlib.axes._axes.Axes): The axis with the added contour.
        """
        # estimate rms noise level if it is not specified.
        if crms is None:
            try:
                crms = self.noise()
                bunit = self.bunit.replace(".", " ")
                print(f"Estimated base contour level (rms): {crms:.4e} [{bunit}]")
            except Exception:
                raise Exception("Failed to estimate RMS noise level of contour map. " + \
                    "Please specify base contour level using 'crms' parameter.")

        # set beam color to be the same as the contour colors
        if beamcolor is None:
            beamcolor = ccolors

        # get limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # plot contour
        contextent = [self.imextent[2], self.imextent[3], 
                     self.imextent[0], self.imextent[1]]
        contdata = self.data[0, 0, :, :].copy()

        # smooth parameter
        if smooth is not None:
            contdata = ndimage.gaussian_filter(contdata, smooth)

        ax.contour(contdata, colors=ccolors, origin='lower', extent=contextent, 
                   levels=crms*clevels, linewidths=clw, zorder=zorder)

        if beamon:
            if beamloc is None:
                # get information of rightmost beam (as reference):
                bottom_quarter_cutoff = 0.25*(ylim[1]-ylim[0]) + ylim[0]  # avoids non-beam ellipses 
                beams = [patch for patch in ax.get_children() if isinstance(patch, patches.Ellipse) and \
                         ylim[0] <= patch.center[1] <= bottom_quarter_cutoff]

                if len(beams) == 0:
                    beamloc = (0.1225, 0.1225)
                else:
                    normalized_locations = tuple(map(lambda beam: _normalize_location(beam.center, xlim, ylim), beams))
                    normalized_x = tuple(map(lambda coord: coord[0], normalized_locations))
                    rightmost_index = np.argmax(normalized_x)

                    ref_beam = beams[rightmost_index]
                    ref_beamloc = normalized_locations[rightmost_index]
                    ref_beam_dims = (ref_beam.get_height(), ref_beam.get_width(), -ref_beam.angle)
                    beamloc = _get_contour_beam_loc(color_beam_dims=ref_beam_dims,
                                                    contour_beam_dims=self.beam,
                                                    color_beamloc=ref_beamloc,
                                                    xlim=xlim,
                                                    beam_separation=beam_separation)
            # add beam to ax object
            ax = self._addbeam(ax, xlim, ylim, beamcolor, 
                               filled=False, beamloc=beamloc)

        if plot:
            plt.show()
            
        return ax
    
    def __trim_data(self, xlim, ylim):
        if list(xlim) == [self.widestfov, -self.widestfov] \
           and list(ylim) == [-self.widestfov, self.widestfov]:
            return self.data[0, 0]
        
        xmask = (min(xlim) <= self.xaxis) & (self.xaxis <= max(xlim))
        ymask = (min(ylim) <= self.yaxis) & (self.yaxis <= max(ylim))
        trimmed_data = self.data[0, 0][ymask, :][:, xmask]
        return trimmed_data

    def __add_scalebar(self, ax, xlim, ylim, distance, size=0.1, barloc=(0.85, 0.15), 
                       txtloc=(0.85, 0.075), scalecolor="w", scalelw=1., fontsize=10.):
        """
        Private method to add a scale bar
        Parameters:
            ax: image on which scale bar will be added
            distance: distance to object (parsec)
            scale: size (length) of scale bar (same unit as spatial axes)
            barloc: location of the bar (relative)
            txtcenter: the location of the text center (relative)
            scalecolor (str): the color of the scale bar
            scaelw (float): the line width of the scale bar
            fontsize (float): the font size of the text label
        Returns:
            ax: the image with the added scale bar
        """
        # conversion factor from angular unit to linear unit
        if self.unit == "arcsec":
            ang2au = distance
        else:
            ang2au = distance*u.Quantity(1, self.unit).to_value(u.arcsec)
            
        # distance in au
        dist_au = size*ang2au
        
        # convert absolute to relative
        barx, bary = _unnormalize_location(barloc, xlim, ylim)
        textx, texty = _unnormalize_location(txtloc, xlim, ylim)
        
        label = str(int(dist_au))+' au'
        ax.text(textx, texty, label, ha='center', va='bottom', 
                color=scalecolor, fontsize=fontsize)
        ax.plot([barx-size/2, barx+size/2], [bary, bary], 
                color=scalecolor, lw=scalelw)
        
        return ax
        
    def _addbeam(self, ax, xlim, ylim, beamcolor, filled=True, beamloc=(0.1225, 0.1225)):
        """
        This method adds an ellipse representing the beam size to the specified ax.
        """
        # coordinate of ellipse center 
        coords = _unnormalize_location(beamloc, xlim, ylim)

        if np.any(np.isnan(self.beam)):
            warnings.warn("Beam cannot be plotted as it is not available in the header.")
            return ax

        # beam dimensions
        bmaj, bmin, bpa = self.beam
        
        # add patch to ax
        if filled:
            beam = patches.Ellipse(xy=coords, width=bmin, height=bmaj, fc=beamcolor,
                                   ec=beamcolor, angle=-bpa, alpha=1, zorder=10)
        else:
            beam = patches.Ellipse(xy=coords, width=bmin, height=bmaj, fc='None',
                                    ec=beamcolor, linestyle="solid", 
                                    angle=-bpa, alpha=1, zorder=10)
        ax.add_patch(beam)
        return ax

    def trim(self, template=None, xlim=None, ylim=None, inplace=False):
        """
        Trim the image data based on specified x and y axis limits.

        Parameters:
        template (Spatialmap/Channelmap): The xlim and ylim will be assigned as the the min/max values 
                                          of the axes of this image.
        xlim (iterable of length 2, optional): The limits for the x-axis to trim the data.
                                               If None, no trimming is performed on the x-axis.
        ylim (iterable of length 2, optional): The limits for the y-axis to trim the data.
                                               If None, no trimming is performed on the y-axis.
        inplace (bool, optional): If True, the trimming is performed in place and the original 
                                  object is modified. If False, a copy of the object is trimmed 
                                  and returned. Default is False.

        Returns:
        image: The trimmed image object. If `inplace` is True, the original object is returned 
               after modification. If `inplace` is False, a new object with the trimmed data is returned.

        Raises:
        ValueError: If any of `xlim` or `ylim` are provided and are not iterables of length 2.

        Notes:
        The method uses the `_trim_data` function to perform the actual data trimming. The reference 
        coordinates are updated based on the new trimmed axes.
        """
        warnings.warn("The method 'trim' is still in testing.")
        if template is not None:
            xlim = template.xaxis.min(), template.xaxis.max()
            ylim = template.yaxis.min(), template.yaxis.max()

        image = self if inplace else self.copy()

        new_data, _, new_yaxis, new_xaxis = _trim_data(self.data, 
                                                       yaxis=self.yaxis,
                                                       xaxis=self.xaxis,
                                                       dy=self.dy,
                                                       dx=self.dx,
                                                       ylim=ylim,
                                                       xlim=xlim)

        # get new parameters:
        new_nx = new_xaxis.size 
        new_ny = new_yaxis.size 
        new_shape = new_data.shape
        new_refnx = new_nx//2 + 1
        new_refny = new_ny//2 + 1

        # calculate new reference coordinate:
        org_ref_skycoord = SkyCoord(self.refcoord)
        org_refcoord_x = org_ref_skycoord.ra
        org_refcoord_y = org_ref_skycoord.dec

        start_x = org_refcoord_x + u.Quantity(new_xaxis.min(), self.unit)
        end_x = org_refcoord_x + u.Quantity(new_xaxis.max(), self.unit)
        new_x = (start_x+end_x+self.dx*u.Unit(self.unit))/2

        start_y = org_refcoord_y + u.Quantity(new_yaxis.min(), self.unit)
        end_y = org_refcoord_y + u.Quantity(new_yaxis.max(), self.unit)
        new_y = (start_y+end_y+self.dy*u.Unit(self.unit))/2

        new_refcoord = SkyCoord(ra=new_x, dec=new_y).to_string("hmsdms")
        # update object:
        image.data = new_data
        image.overwrite_header(shape=new_shape, nx=new_nx, ny=new_ny, 
                               refnx=new_refnx, refny=new_refny, 
                               refcoord=new_refcoord)
        return image
    
    def line_info(self, **kwargs):
        """
        This method searches for the molecular line data from the Splatalogue database
        """
        if np.isnan(self.restfreq):
            raise Exception("Failed to find molecular line as rest frequency cannot be read.")
        return search_molecular_line(self.restfreq, unit="Hz", **kwargs)
    
    def get_hduheader(self):
        """
        To retrieve the header of the current FITS image. This method accesses the header 
        information of the original FITS file, and then modifies it to reflect the current
        status of this image object.

        Returns:
            The FITS header of the current image object (astropy.io.fits.header.Header).
        """
        self.__updateparams()
        return _get_hduheader(self)

    def get_hdu(self):
        """
        Get the primary HDU (astropy object) of the image.
        """
        return fits.PrimaryHDU(data=self.data, header=self.get_hduheader())

    def get_wcs(self):
        """
        Get the world coordinate system of the image (astropy object.)
        """
        return WCS(self.get_hduheader())
    
    def overwrite_header(self, new_vals=None, **kwargs):
        """
        Method to overwrite the existing keys of the header with new values.
        Parameters:
            new_vals (dict): a dictionary containing keys and values to be overwritten.
            **kwargs (dict): keyword arguments that will be overwritten in the header
        Return:
            self.header (dict): the updated header 
        """
        if new_vals is None and len(kwargs) == 0:
            raise ValueError("Header cannot be overwritten. Need to input a dictionary or keyword arguments.")
        if new_vals is not None:
            if isinstance(new_vals, dict):
                for key, value in new_vals.items():
                    if key in self.header:
                        self.header[key] = value
                    else:
                        print(f"'{key}' is not a valid keyword of the header and will be ignored.")
            else:
                raise TypeError("Please input a new dictionary as the header.")
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                if key in self.header:
                    self.header[key] = value
                else:
                    print(f"'{key}' is not a valid keyword of the header and will be ignored.")
        self.__updateparams()
        return self.header
                
    def exportfits(self, outname, overwrite=False):
        """
        Save the current image to a FITS file.

        This method exports the image data and header to a FITS file. If the specified 
        file name does not end with '.fits', it is appended. When 'overwrite' is False 
        and the file already exists, a number is appended to the file name to avoid 
        overwriting (e.g., 'filename(1).fits').

        Parameters:
            outname (str): The name of the output FITS file.
            overwrite (bool): If True, allows overwriting an existing file with the 
                              same name. If False, the file name is modified to 
                              prevent overwriting existing files.
        Returns:
            None
        """
        # get header
        hdu_header = self.get_hduheader()
        
        # add file name extension if not in user input
        if not outname.endswith(".fits"):
            outname += ".fits"
        
        # if not overwrite, add (1), (2), (3), etc. to file name before '.fits'
        if not overwrite:
            outname = _prevent_overwriting(outname)
        
        # Write to a FITS file
        hdu = fits.PrimaryHDU(data=self.data, header=hdu_header)
        hdu.writeto(outname, overwrite=overwrite)
        print(f"File saved as '{outname}'.")

    def to_casa(self, *args, **kwargs):
        """
        Converts the spatial map object into CASA image format. 
        Wraps the 'importfits' function of casatasks.

        Parameters:
            outname (str): The output name for the CASA image file. Must end with ".image".
            whichrep (int, optional): The FITS representation to convert. Defaults to 0.
            whichhdu (int, optional): The HDU (Header/Data Unit) number to convert. Defaults to -1.
            zeroblanks (bool, optional): Replace undefined values with zeros. Defaults to True.
            overwrite (bool, optional): Overwrite the output file if it already exists. Defaults to False.
            defaultaxes (bool, optional): Use default axes for the output CASA image. Defaults to False.
            defaultaxesvalues (str, optional): Default axes values, provided as a string. Defaults to '[]'.
            beam (str, optional): Beam parameters, provided as a string. Defaults to '[]'.

        Raises:
            ValueError: If 'outname' is not a string.

        Returns:
            None

        Example:
            image.to_casa("output_image")
        """
        to_casa(self, *args, **kwargs)


class PVdiagram:
    """
    A class for handling FITS position-velocity diagrams in astronomical imaging.

    This class provides functionality to load, process, and manipulate position-velocity diagrams. 
    It supports operations like arithmetic calculations between data cubes, rotation, 
    normalization, regridding, and more. The class can handle FITS files directly and acts
    like a numpy array.

    Note:
        The class performs several checks on initialization to ensure that the provided data
        is in the correct format. It can handle FITS files with different configurations and
        is designed to be flexible for various data shapes and sizes.
    """
    def __init__(self, fitsfile=None, header=None, data=None, hduindex=0, 
                 spatialunit="arcsec", specunit="km/s", quiet=False):
        if fitsfile is not None:
            fits = importfits(fitsfile, hduindex=hduindex, spatialunit=spatialunit, 
                              specunit=specunit, quiet=False)
            self.header = fits.header
            self.data = fits.data
        elif header is not None:
            self.header = header
            self.data = data
        if self.header["imagetype"] != "pvdiagram":
            raise TypeError("The given FITS file is not a PV diagram.")
        self.__updateparams()
        self.pa = None  # position angle at which the PV diagram was cut (to calculate offset resolution)
        
        if isinstance(self.data, u.quantity.Quantity):
            self.value = PVdiagram(header=self.header, data=self.data.value)
        
    def __updateparams(self):
        self.spatialunit = self.unit = self.axisunit = self.header["unit"]
        self.nx = self.header["nx"]
        self.dx = self.header["dx"]
        refnx = self.refnx = self.header["refnx"]
        self.ny = self.header["ny"]
        self.xaxis = self.get_xaxis()
        self.shape = self.header["shape"]
        self.size = self.data.size
        self.restfreq = self.header["restfreq"]
        self.bmaj, self.bmin, self.bpa = self.beam = self.header["beam"]
        self.resolution = np.sqrt(self.beam[0]*self.beam[1]) if self.beam is not None else None
        self.refcoord = self.header["refcoord"]
        if isinstance(self.data, u.Quantity):
            self.bunit = self.header["bunit"] = _apu_to_headerstr(self.data.unit)
        else:
            self.bunit = self.header["bunit"]
        self.specunit = self.header["specunit"]
        self.vrange = self.header["vrange"]
        self.dv = self.header["dv"]
        self.nv = self.nchan = self.header["nchan"]
        if self.specunit == "km/s":
            rounded_dv = round(self.header["dv"], 5)
            vmin, vmax = self.header["vrange"]
            rounded_vmin = round(vmin, 5)
            rounded_vmax = round(vmax, 5)
            if np.isclose(rounded_dv, self.header["dv"]):
                self.dv = self.header["dv"] = rounded_dv
            if np.isclose(vmin, rounded_vmin):
                vmin = rounded_vmin
            if np.isclose(vmax, rounded_vmax):
                vmax = rounded_vmax
            self.vrange = self.header["vrange"] = [vmin, vmax]
        self.vaxis = self.get_vaxis()
        vmin, vmax = self.vaxis[[0, -1]]
        xmin, xmax = self.xaxis[[0, -1]]
        self.imextent = [vmin-0.5*self.dx, vmax+0.5*self.dx, 
                         xmin-0.5*self.dx, xmax+0.5*self.dx]
        v1, v2 = self.vaxis[0]-self.dv*0.5, self.vaxis[-1]+self.dv*0.5
        x1 = self.xaxis[0]-self.dx*0.5 if self.xaxis[0] == self.xaxis.min() \
             else self.xaxis[0]+self.dx*0.5
        x2 = self.xaxis[-1]+self.dx*0.5 if self.xaxis[-1] == self.xaxis.max() \
             else self.xaxis[-1]-self.dx*0.5
        self.imextent = [v1, v2, x1, x2]
        self.maxxlim = self.xaxis[[0, -1]]
        self.maxvlim = self.vaxis[[0, -1]]
        
    # magic methods to define operators
    def __add__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return PVdiagram(header=self.header, data=self.data+other.data)
        return PVdiagram(header=self.header, data=self.data+other)
    
    def __radd__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return PVdiagram(header=self.header, data=other.data+self.data)
        return PVdiagram(header=self.header, data=other+self.data)
        
    def __sub__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return PVdiagram(header=self.header, data=self.data-other.data)
        return PVdiagram(header=self.header, data=self.data-other)
    
    def __rsub__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                warnings.warn("operation performed on two images with different units.")
            return PVdiagram(header=self.header, data=other.data-self.data)
        return PVdiagram(header=self.header, data=other-self.data)
        
    def __mul__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data*other.data)
        return PVdiagram(header=self.header, data=self.data*other)
    
    def __rmul__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=other.data*self.data)
        return PVdiagram(header=self.header, data=other*self.data)
    
    def __pow__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data**other.data)
        return PVdiagram(header=self.header, data=self.data**other)
        
    def __rpow__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=other.data**self.data)
        return PVdiagram(header=self.header, data=other**self.data)
        
    def __truediv__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data/other.data)
        return PVdiagram(header=self.header, data=self.data/other)
    
    def __rtruediv__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=other.data/self.data)
        return PVdiagram(header=self.header, data=other/self.data)
        
    def __floordiv__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data//other.data)
        return PVdiagram(header=self.header, data=self.data//other)
    
    def __rfloordiv__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=other.data//self.data)
        return PVdiagram(header=self.header, data=other//self.data)
    
    def __mod__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data%other.data)
        return PVdiagram(header=self.header, data=self.data%other)
    
    def __rmod__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=other.data%self.data)
        return PVdiagram(header=self.header, data=other%self.data)
    
    def __lt__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data<other.data)
        return PVdiagram(header=self.header, data=self.data<other)
    
    def __le__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data<=other.data)
        return PVdiagram(header=self.header, data=self.data<=other)
    
    def __eq__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data==other.data)
        return PVdiagram(header=self.header, data=self.data==other)
        
    def __ne__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data!=other.data)
        return PVdiagram(header=self.header, data=self.data!=other)

    def __gt__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data>other.data)
        return PVdiagram(header=self.header, data=self.data>other)
        
    def __ge__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    warnings.warn("operation performed on two images with significantly different beam sizes.")
            return PVdiagram(header=self.header, data=self.data>=other.data)
        return PVdiagram(header=self.header, data=self.data>=other)

    def __abs__(self):
        return PVdiagram(header=self.header, data=np.abs(self.data))
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return PVdiagram(header=self.header, data=-self.data)
    
    def __invert__(self):
        return PVdiagram(header=self.header, data=~self.data)
    
    def __getitem__(self, indices):
        try:
            try:
                return PVdiagram(header=self.header, data=self.data[indices])
            except:
                warnings.warn("Returning value after reshaping image data to 2 dimensions.")
                return self.data.copy[:, indices[0], indices[1]]
        except:
            return self.data[indices]
    
    def __setitem__(self, indices, value):
        newdata = self.data.copy()
        newdata[indices] = value
        return PVdiagram(header=self.header, data=newdata)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Extract the Spatialmap object from inputs
        inputs = [x.data if isinstance(x, PVdiagram) else x for x in inputs]
        
        if ufunc == np.round:
            # Process the round operation
            if 'decimals' in kwargs:
                decimals = kwargs['decimals']
            elif len(inputs) > 1:
                decimals = inputs[1]
            else:
                decimals = 0  # Default value for decimals
            return PVdiagram(header=self.header, data=np.round(self.data, decimals))
        
        # Apply the numpy ufunc to the data
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Return a new PVdiagram instance with the result if the ufunc operation was successful
        if method == '__call__' and isinstance(result, np.ndarray):
            return PVdiagram(header=self.header, data=result)
        else:
            return result
        
    def __array__(self, *args, **kwargs):
        return np.array(self.data, *args, **kwargs)

    def min(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the minimum value of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's min function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's min function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the minimum.

        Returns:
            scalar or array: Minimum value of the data array. If the data array is multi-dimensional, 
                             returns an array of minimum values along the specified axis.
        """
        if ignore_nan:
            return np.nanmin(self.data, *args, **kwargs)
        return np.min(self.data, *args, **kwargs)

    def max(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the maximum value of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's max function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's max function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the maximum.

        Returns:
            scalar or array: Maximum value of the data array. If the data array is multi-dimensional, 
                             returns an array of maximum values along the specified axis.
        """
        if ignore_nan:
            return np.nanmax(self.data, *args, **kwargs)
        return np.max(self.data, *args, **kwargs)

    def mean(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the mean value of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's mean function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's mean function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the mean.

        Returns:
            scalar or array: Mean value of the data array. If the data array is multi-dimensional, 
                             returns an array of mean values along the specified axis.
        """
        if ignore_nan:
            return np.nanmean(self.data, *args, **kwargs)
        return np.mean(self.data, *args, **kwargs)

    def sum(self, *args, ignore_nan=True, **kwargs):
        """
        Calculate the sum of the data array.

        Parameters:
            *args: tuple, optional
                Additional arguments passed to numpy's sum function.
            **kwargs: dict, optional
                Additional keyword arguments passed to numpy's sum function.
            ignore_nan (bool): Default is True. If True, ignores NaN values when computing the sum.

        Returns:
            scalar or array: Sum of the data array. If the data array is multi-dimensional, 
                             returns an array of sum values along the specified axis.
        """
        if ignore_nan:
            return np.nansum(self.data, *args, **kwargs)
        return np.sum(self.data, *args, **kwargs)
    
    def to(self, unit, *args, **kwargs):
        """
        This method converts the intensity unit of original image to the specified unit.
        """
        return PVdiagram(header=self.header, data=self.data.to(unit, *args, **kwargs))

    def to_value(self, unit, *args, **kwargs):
        """
        Duplicate of astropy.unit.Quantity's 'to_value' method.
        """
        image = self.copy()
        image.data = image.data.to_value(unit, *args, **kwargs)
        image.bunit = image.header["bunit"] = _apu_to_headerstr(_to_apu(unit))
        return image
        
    def copy(self):
        """
        This method makes a deep copy of the instance.
        """
        return copy.deepcopy(self)
    
    def get_xaxis(self, unit=None):
        """
        Get the spatial axis of the PV diagram.
        Parameters:
            unit (str): the unit of the spatial axis that will be returned.
                        Defualt (None) is to use the same unit as one read from the FITS file.
        Returns:
            The spatial axis (ndarray).
        """
        # create axes in original unit
        xaxis = self.dx * (np.arange(self.nx)-self.refnx+1)
        xaxis = np.round(xaxis, 6)
        
        # convert units to specified
        if unit is not None:
            xaxis = u.Quantity(xaxis, self.unit).to_value(unit)
        return xaxis
    
    def get_vaxis(self, specunit=None):
        """
        To get the vaxis of the data cube.
        Parameters:
            specunit (str): the unit of the spectral axis. 
                            Default is to use the same as the header unit.
        Returns:
            vaxis (ndarray): the spectral axis of the data cube.
        """
        vaxis = np.linspace(self.vrange[0], self.vrange[1], self.nchan)
        if specunit is None:
            if self.specunit == "km/s":
                vaxis = np.round(vaxis, 7)
            return vaxis
        try:
            # attempt direct conversion
            vaxis = u.Quantity(vaxis, self.specunit).to_value(specunit)
        except UnitConversionError:
            # if that fails, try using equivalencies
            equiv = u.doppler_radio(self.restfreq*u.Hz)
            vaxis = u.Quantity(vaxis, self.specunit).to_value(specunit, equivalencies=equiv)
        if _to_apu(specunit).is_equivalent(u.km/u.s):
            round_mask = np.isclose(vaxis, np.round(vaxis, 5))
            vaxis[round_mask] = np.round(vaxis, 5)
            return vaxis
        return vaxis
    
    def conv_specunit(self, specunit, inplace=True):
        """
        Convert the spectral axis into the desired unit.
        """
        image = self if inplace else self.copy()
        vaxis = image.get_vaxis(specunit=specunit)
        image.header["dv"] = vaxis[1]-vaxis[0]
        image.header["vrange"] = vaxis[[0, -1]].tolist()
        image.header["specunit"] = _apu_to_headerstr(_to_apu(specunit))
        image.__updateparams()
        return image
    
    def set_data(self, data, inplace=False):
        """
        This method allows users to assign a new dataset to the image.
        Parameters:
            data (ndarray): the new data to be set
            inplace (bool): True to modify the data of this image. False to only return a copy.
        Return: 
            An image with the specified dataset.
        """
        if data.shape != self.shape:
            data = data.reshape(self.shape)
        if inplace:
            self.data = data
            return self
        newimg = self.copy()
        newimg.data = data
        return newimg
    
    def set_threshold(self, threshold=None, minimum=True, inplace=True):
        """
        To mask the intensities above or below this threshold.
        Parameters:
            threshold (float): the value of the threshold. Default is to use three times the rms noise level.
            minimum (bool): True to remove intensities below the threshold. 
                            False to remove intensities above the threshold.
            inplace (bool): True to modify the data in-place. False to return a new image.
        Returns:
            Image with the masked data.
        """
        if threshold is None:
            threshold = 3*self.noise()
        if inplace:
            if minimum:
                self.data = np.where(self.data<threshold, np.nan, self.data)
            else:
                self.data = np.where(self.data<threshold, self.data, np.nan)
            return self
        if minimum:
            return PVdiagram(header=self.header, data=np.where(self.data<threshold, np.nan, self.data))
        return PVdiagram(header=self.header, data=np.where(self.data<threshold, self.data, np.nan))
    
    def noise(self, sigma=3., plthist=False, shownoise=False, printstats=False, bins='auto', 
              gaussfit=False, curvecolor="crimson", curvelw=0.6, fixmean=True, histtype='step', 
              linewidth=0.6, returnmask=False, **kwargs):
        """
        This method estimates the 1 sigma background noise level using sigma clipping.
        Parameters:
            sigma (float): sigma parameter of sigma clip
            plthist (bool): True to show background noise distribution as histogram 
                            (useful for checking if noise is Gaussian)
            shownoise (bool): True to background noise spatial distribution
            printstats (bool): True to print information regarding noise statistics
            bins (int): The bin size of the histogram. Applicable if plthist or gaussfit
            gaussfit (bool): True to perform 1D gaussian fitting on noise distribution
            curvecolor (str): the color of the best-fit curve to be plotted, if applicable
            curvelw (float): the line width of the best-fit curve to be plotted, if applicable
            fixmean (bool): True to fix fitting parameter of mean to 0, if guassfit 
            histtype (str): the type of histogram
            linewidth (float): the line width of the histogram line borders
            returnmask (bool): True to return the mask (pixels that are considered 'noise')
        Returns:
            rms (float): the rms noise level
            mask (ndarray): the noise mask (pixels that are considered noise), if returnmask=True
        """
        bunit = self.bunit.replace(".", " ")
        
        clipped_data = sigma_clip(self.data, sigma=sigma, maxiters=10000, masked=False, axis=(0, 1, 2))
        bg_image = self.set_data(clipped_data, inplace=False)
        mean = np.nanmean(clipped_data)
        rms = np.sqrt(np.nanmean(clipped_data**2))
        std = np.nanstd(clipped_data)
        
        if not rms:
            raise Exception("Sigma clipping failed. It is likely that most noise has been masked.")
        
        if printstats:
            print(15*"#" + "Noise Statistics" + 15*"#")
            print(f"Mean: {mean:.6e} [{bunit}]")
            print(f"RMS: {rms:.6e} [{bunit}]")
            print(f"StdDev: {std:.6e} [{bunit}]")
            print(46*"#")
            print()
        
        if gaussfit:
            if fixmean:
                def gauss(x, sigma, amp):
                    return amp*np.exp(-x**2/(2*sigma**2))
            else:
                def gauss(x, sigma, amp, mean):
                    return amp*np.exp(-(x-mean)**2/(2*sigma**2))
            flatdata = clipped_data[~np.isnan(clipped_data)]
            ydata, edges = np.histogram(flatdata, bins=bins)
            xdata = (edges[1:] + edges[:-1]) / 2  # set as midpoint of the bins
            p0 = [std, np.max(ydata)] if fixmean else [std, np.max(ydata), mean]
            popt, pcov = curve_fit(f=gauss, xdata=xdata, ydata=ydata, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            fitx = np.linspace(xdata.min(), xdata.max(), 1000)
            fity = gauss(fitx, popt[0], popt[1]) if fixmean else gauss(fitx, popt[0], popt[1], popt[2])
            
            print(15*"#" + "Gaussian Fitting Results" + 15*"#")
            if not fixmean:
                print(f"Mean: {popt[2]:.4e} +/- {perr[2]:.4e} [{bunit}]")
            print(f"StdDev: {popt[0]:.4e} +/- {perr[0]:.4e} [{bunit}]")
            print(f"Amplitude: {np.round(popt[1])} +/- {np.round(perr[1])} [pix]")
            print(54*"#")
            print()
            
        if plthist:
            q = _to_apu(self.bunit)
            flatdata = clipped_data[~np.isnan(clipped_data)]
            ax = plt_1ddata(flatdata, hist=True, 
                            xlabel=f"Intensity ({q:latex_inline})", ylabel="Pixels",
                            bins=bins, plot=False, xlim=[flatdata.min(), flatdata.max()], 
                            linewidth=linewidth, histtype=histtype, **kwargs)
            if gaussfit:
                ax.plot(fitx, fity, color=curvecolor, lw=curvelw)
                
        if shownoise:
            bg_image.imview(title="Background Noise Distribution", **kwargs)
            
        if plthist or shownoise:
            plt.show()
        
        if returnmask:
            mask = ~np.isnan(clipped_data)
            return rms, mask
        return rms

    def conv_unit(self, unit, inplace=True):
        """
        This method converts the axis unit of the image into the desired unit.
        Parameters:
            unit (str): the new axis unit.
            inplace (bool): True to update the current axes with the new unit. 
                            False to create a new image having axes with the new unit.
        Returns:
            A new image with axes of the desired unit.
        """
        newbeam = (u.Quantity(self.beam[0], self.unit).to_value(unit), 
                   u.Quantity(self.beam[1], self.unit).to_value(unit), 
                   self.bpa)
        if inplace:
            self.header["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
            self.header["beam"] = newbeam
            self.header["unit"] = _apu_to_headerstr(_to_apu(unit))
            self.__updateparams()
            return self
        newheader = copy.deepcopy(self.header)
        newheader["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
        newheader["beam"] = newbeam
        newheader["unit"] = _apu_to_headerstr(_to_apu(unit))
        return PVdiagram(header=newheader, data=self.data)
    
    def conv_bunit(self, bunit, inplace=True):
        """
        This method converts the brightness unit of the image into the desired unit.
        Parameters:
            bunit (str): the new unit.
            inplace (bool): True to update the current data with the new unit. 
                            False to create a new image having data with the new unit.
        Returns:
            A new image with the desired unit.
        """
        # string to astropy units 
        bunit = _to_apu(bunit)
        oldunit = _to_apu(self.bunit)
        
        # equivalencies
        equiv_bt = u.equivalencies.brightness_temperature(frequency=self.restfreq*u.Hz, 
                                                          beam_area=self.beam_area())
        equiv_pix = u.pixel_scale(u.Quantity(np.abs(self.dx), self.unit)**2/u.pixel)
        equiv_ba = u.beam_angular_area(self.beam_area())
        equiv = [equiv_bt, equiv_pix, equiv_ba]

        # factors
        factor_bt = (1*u.Jy/u.beam).to(u.K, equivalencies=equiv_bt) / (u.Jy/u.beam)
        factor_pix = (1*u.pixel).to(u.rad**2, equivalencies=equiv_pix) / (u.pixel)
        factor_ba = (1*u.beam).to(u.rad**2, equivalencies=equiv_ba) / (u.beam)
        factor_pix2bm = factor_pix / factor_ba
        factor_Jypix2K = factor_pix2bm / factor_bt
        factor_Jysr2K = factor_bt*factor_ba
        factors = [factor_bt, factor_pix, factor_ba, factor_pix2bm, 
                   factor_Jypix2K, factor_Jysr2K]
        
        # convert
        if isinstance(self.data, u.Quantity):
            olddata = self.data.copy()
        else:
            olddata = u.Quantity(self.data, oldunit)
        newdata = _convert_Bunit(olddata, newunit=bunit, 
                                 equivalencies=equiv, factors=factors)
        
        if newdata is None:
            raise UnitConversionError(f"Failed to convert intensity unit to {_apu_to_headerstr(bunit)}")

        # return and set values
        if not isinstance(self.data, u.Quantity):
            newdata = newdata.value
            
        newimage = self if inplace else self.copy()
        newimage.data = newdata
        newimage.header["bunit"] = _apu_to_headerstr(bunit)
        newimage.__updateparams()
        return newimage

    def beam_area(self, unit=None):
        """
        To calculate the beam area of the image.
        Parameters:
            unit (float): the unit of the beam area to be returned. 
                          Default (None) is to use same unit as the positional axes.
        Returns:
            The beam area.
        """
        bmaj = u.Quantity(self.bmaj, self.unit)
        bmin = u.Quantity(self.bmin, self.unit)
        area = np.pi*bmaj*bmin/(4*np.log(2))
        if unit is not None:
            area = area.to_value(unit)
        return area

    def get_representative_points(self, threshold=None, vlim=None, xlim=None):
        vaxis = self.vaxis   # shape: (nv,)
        xaxis = self.xaxis   # shape: (nx,)
        data = self.data[0].copy()  # shape: (nv, nx)

        # set values below threshold to NaN
        if threshold is not None:
            data[data<threshold] = np.nan 

        # set velocity and position limits
        if vlim is not None:
            vmin = min(vlim) - 0.5*abs(self.dv)
            vmax = max(vlim) + 0.5*abs(self.dv) 
            vmask = (vmin<=vaxis) & (vaxis<=vmax)
            vaxis = vaxis[vmask]
            data = data[vmask, :]

        if xlim is not None:
            xmin = min(xlim) - abs(0.5*self.dx)
            xmax = max(xlim) + abs(0.5*self.dx)
            xmask = (xmin<=xaxis) & (xaxis<=xmax)
            xaxis = xaxis[xmask]
            data = data[:, xmask]
            
        # mean_velocities = np.empty_like(xaxis)
        # for i in range(mean_velocities.shape[0]):
        #     mean_velocities[i] = np.average(vaxis, weights=data[:, i])
        
        # mean_positions = np.empty_like(vaxis):
        # for j in range(mean_positions.shape[0]):
        #     mean_positions[j] = np.average(xaxis, weights=data[j, :])

        # return (mean_velocities, xaxis), (vaxis, mean_positions)

        # Calculate mean velocities
        sum_weights = np.nansum(data, axis=0)
        mean_velocities = np.nansum(vaxis[:, np.newaxis] * data, axis=0) / sum_weights

        # Calculate mean positions
        sum_weights = np.nansum(data, axis=1)
        mean_positions = np.nansum(xaxis * data, axis=1) / sum_weights

        return (mean_velocities, xaxis), (vaxis, mean_positions)
    
    def __get_offset_resolution(self, pa=None):
        if pa is None:
            return np.sqrt(self.bmaj*self.bmin)  # simply take geometric mean if pa is not specified
        angle = np.deg2rad(pa-self.bpa)
        aa = np.square(np.sin(angle)/self.bmin)
        bb = np.square(np.cos(angle)/self.bmaj)
        return np.sqrt(1/(aa+bb))
        
    def imview(self, contourmap=None, cmap="inferno", vmin=None, vmax=None, nancolor="k", crms=None, 
               clevels=np.arange(3, 21, 3), ccolors="w", clw=1., dpi=600, cbaron=True, cbarloc="right", 
               cbarpad=0., vsys=None, percentile=None, xlim=None, vlim=None, xcenter=0., vlineon=True, 
               xlineon=True, cbarlabelon=True, cbarwidth='5%', cbarlabel=None, cbarticks=None, fontsize=10, 
               labelsize=10, width=330, height=300, plotres=True, xlabelon=True, vlabelon=True, xlabel=None, 
               vlabel=None, offset_as_hor=False, aspect_ratio=1., axeslw=1., tickson=True, tickwidth=None, 
               tickdirection="in", ticksize=3., tickcolor="k", cbartick_width=None, cbartick_length=3, 
               cbartick_direction="in", xticks=None, title_fontsize=12, vticks=None, title=None, 
               titleloc=(0.1, 0.9), ha="left", va="top", txtcolor="w", refline_color="w", pa=None, 
               refline_width=None, subtract_vsys=False, errbarloc=(0.1, 0.1225), errbarlw=None, 
               errbar_captick=None, errbar_capsize=1.5, scale="linear", gamma=1.5, smooth=None, 
               flip_offset=False, flip_contour_offset=False, decimals=2, ax=None, savefig=None, plot=True):
        """
        Display a Position-Velocity (PV) diagram.

        This method generates and plots a PV diagram, offering several customization options 
        for visual aspects like colormap, contour levels, axis labels, etc.

        Parameters:
            contourmap (PVdiagram, optional): A PVdiagram instance to use for contour mapping. Default is None.
            cmap (str): Colormap for the image data. Default is 'inferno'.
            nancolor (str): Color used for NaN values in the data. Default is 'k' (black).
            crms (float, optional): RMS noise level for contour mapping. Automatically estimated if None and contourmap is provided.
            clevels (numpy.ndarray): Contour levels for the contour mapping. Default is np.arange(3, 21, 3).
            clw (float): Line width of the contour lines. Default is 1.0.
            dpi (int): Dots per inch for the plot. Affects the quality of the image. Default is 500.
            cbaron (bool): Flag to display the color bar. Default is True.
            cbarloc (str): Location of the color bar. Default is 'right'.
            cbarpad (str): Padding for the color bar. Default is '0%'.
            vsys (float, optional): Systemic velocity. If provided, adjusts velocity axis accordingly. 
                                    Default is None (no systemic velocity).
            xlim (tuple, optional): Limits for the X-axis. Default is None (automatically determined).
            vlim (tuple, optional): Limits for the velocity axis. Default is None (automatically determined).
            xcenter (float): Center position for the X-axis. Default is 0.0.
            vlineon (bool): Flag to display a vertical line at systemic velocity. Default is True.
            xlineon (bool): Flag to display a horizontal line at the center position. Default is True.
            cbarlabelon (bool): Flag to display the label on the color bar. Default is True.
            cbarwidth (str): Width of the color bar. Default is '5%'.
            cbarlabel (str, optional): Label for the color bar. Default is None (automatically determined).
            fontsize (int): Font size for labels and ticks. Default is 18.
            labelsize (int): Size for axis labels. Default is 18.
            figsize (tuple): Size of the figure. Default is (11.69, 8.27).
            plotres (bool): Flag to display resolution elements. Default is True.
            xlabelon (bool): Flag to display the X-axis label. Default is True.
            vlabelon (bool): Flag to display the velocity axis label. Default is True.
            xlabel (str, optional): Label for the X-axis. Default is None (automatically determined).
            vlabel (str, optional): Label for the velocity axis. Default is None (automatically determined).
            offset_as_hor (bool): Treat offset as horizontal. Default is False.
            aspect_ratio (float): Aspect ratio of the plot. Default is 1.1.
            axeslw (float): Line width of the plot axes. Default is 1.3.
            tickwidth (float): Width of the ticks in the plot. Default is 1.3.
            tickdirection (str): Direction of the ticks ('in' or 'out'). Default is 'in'.
            ticksize (float): Size of the ticks. Default is 5.0.
            xticks (list, optional): Custom tick positions for X-axis. Default is None.
            vticks (list, optional): Custom tick positions for velocity axis. Default is None.
            title (str, optional): Title of the plot. Default is None.
            titlepos (float): Position of the title. Default is 0.85.
            txtcolor (str): Color of the text in the plot. Default is 'w' (white).
            refline_color (str): Color of the reference lines. Default is 'w' (white).
            refline_width (float, optional): Line width of the reference lines. Default is None (same as clw).
            savefig (dict): list of keyword arguments to be passed to 'plt.savefig'.
            plot (bool): Flag to execute the plotting. Default is True.

        Returns:
            matplotlib.axes.Axes: The Axes object of the plot if 'plot' is True, allowing further customization.
        """
        # initialize parameters:
        if tickwidth is None:
            tickwidth = axeslw

        if cbartick_width is None:
            cbartick_width = axeslw

        if cbarticks is None:
            cbarticks = []

        if not isinstance(clevels, np.ndarray):
            clevels = np.array(clevels)

        if isinstance(self.data, u.Quantity):
            colordata = self.data.value.copy()[0]
        else:
            colordata = self.data.copy()[0]

        if flip_offset:
            colordata = colordata[:, ::-1]

        if pa is None:
            pa = self.pa
            if self.pa is None:
                warnings.warn("Position angle is not specified nor known from the header. " + \
                               "The beam dimension in the error bar will be shown as the geometric " + \
                               "mean of the beam dimensions.")
        
        if subtract_vsys and vsys is None:
            raise ValueError("'vsys' parameter cannot be None if subtract_vsys = True.")

        if crms is None and contourmap is not None:
            try:
                crms = contourmap.noise()
                bunit = self.bunit.replace(".", " ")
                print(f"Estimated base contour level (rms): {crms:.4e} [{bunit}]")
            except Exception:
                contourmap = None
                print("Failed to estimate RMS noise level of contour map.")
                print("Please specify base contour level using 'crms' parameter.")

        if contourmap is not None:
            if isinstance(contourmap.data, u.Quantity):
                contmap = contourmap.value.copy()
            else:
                contmap = contourmap.copy()
            if flip_contour_offset:
                contmap.data = contmap.data[:, :, ::-1]

        if xlim is None:
            xlim = self.maxxlim

        if vlim is None:
            if subtract_vsys:
                vlim = np.array(self.maxvlim)-vsys
            else:
                vlim = self.maxvlim

        if vlabel is None:
            if subtract_vsys:
                vlabel = r"$v_{\rm obs}-v_{\rm sys}$ " + "(" + _unit_plt_str(_apu_to_str(_to_apu(self.specunit))) + ")"
            else:
                vlabel = "LSR velocity " + "(" + _unit_plt_str(_apu_to_str(_to_apu(self.specunit))) + ")"
        
        if xlabel is None:
            xlabel = f'Offset ({self.unit})'

        if cbarlabel is None:
            cbarlabel = "(" + _unit_plt_str(_apu_to_str(_to_apu(self.bunit))) + ")"

        if refline_width is None:
            refline_width = clw

        if errbarlw is None:
            errbarlw = axeslw

        if errbar_captick is None:
            errbar_captick = errbarlw

        vres, xres = self.dv, self.__get_offset_resolution(pa=pa)
        cmap = copy.deepcopy(mpl.colormaps[cmap]) 
        cmap.set_bad(color=nancolor)
                
        # change default matplotlib parameters
        if ax is None:
            ncols, nrows = 1, 1
            fig_width_pt  = width*ncols
            fig_height_pt = height*nrows
            inches_per_pt = 1.0/72.27                     # Convert pt to inch
            fig_width     = fig_width_pt * inches_per_pt  # width in inches
            fig_height    = fig_height_pt * inches_per_pt # height in inches
            fig_size      = [fig_width, fig_height]
            params = {'axes.labelsize': labelsize,
                      'axes.titlesize': labelsize,
                      'font.size': fontsize,
                      'legend.fontsize': labelsize,
                      'xtick.labelsize': labelsize,
                      'ytick.labelsize': labelsize,
                      'xtick.top': True,   # draw ticks on the top side
                      'xtick.major.top': True,
                      'figure.figsize': fig_size,
                      'figure.dpi': dpi,
                      'font.family': _fontfamily,
                      'mathtext.fontset': _mathtext_fontset,
                      'mathtext.tt': _mathtext_tt,
                      'axes.linewidth': axeslw,
                      'xtick.major.width': tickwidth,
                      'xtick.major.size': ticksize,
                      'xtick.direction': tickdirection,
                      'ytick.major.width': tickwidth,
                      'ytick.major.size': ticksize,
                      'ytick.direction': tickdirection,
                      }
            rcParams.update(params)

            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(111)
        
        if percentile is not None:
            trimmed_data = self.__trim_data(xlim=xlim, vlim=vlim)
            vmin, vmax = clip_percentile(data=trimmed_data, area=percentile)
        
        # get data for preparation
        imextent = copy.deepcopy(self.imextent)
        
        # check if color map is in log scale
        is_logscale: bool = (scale.lower() in ("log", "logscale", "logarithm"))
        
        # plot image
        if offset_as_hor:
            imextent = [imextent[2], imextent[3], imextent[0], imextent[1]]
            if subtract_vsys:
                imextent[3] -= vsys
                imextent[4] -= vsys
            ax, climage = _plt_cmap(image_obj=self, 
                                    ax=ax,
                                    two_dim_data=colordata,
                                    imextent=imextent, 
                                    cmap=cmap,
                                    vmin=vmin, 
                                    vmax=vmax,
                                    scale=scale, 
                                    gamma=gamma)
                
            if contourmap is not None:
                contextent = [contmap.imextent[2], contmap.imextent[3], contmap.imextent[0], contmap.imextent[1]]
                contdata = contmap.data[0, :, :]
                if smooth is not None:
                    contdata = ndimage.gaussian_filter(contdata, smooth)
                ax.contour(contdata, colors=ccolors, origin='lower', extent=contextent, 
                           levels=crms*clevels, linewidths=clw)
            if xlabelon:
                ax.set_xlabel(xlabel, fontsize=fontsize)
            if vlabelon:
                ax.set_ylabel(vlabel, fontsize=fontsize)
            ax.set_xlim(xlim)
            ax.set_ylim(vlim)
            if xticks is not None:
                ax.set_xticks(xticks)
            if vticks is not None:
                ax.set_vticks(vticks)
            if xlineon:
                ax.plot([xcenter, xcenter], [vlim[0], vlim[1]], color=refline_color, 
                        ls='dashed', lw=refline_width)
            if vlineon and vsys is not None:
                if subtract_vsys:
                    ax.plot([xlim[0], xlim[1]], [0, 0], color=refline_color, 
                            ls='dashed', lw=refline_width)
                else:
                    ax.plot([xlim[0], xlim[1]], [vsys, vsys], color=refline_color, 
                            ls='dashed', lw=refline_width)
        else:
            if subtract_vsys:
                imextent[0] -= vsys
                imextent[1] -= vsys
                
            ax, climage = _plt_cmap(image_obj=self, 
                                    ax=ax,
                                    two_dim_data=colordata.T,
                                    imextent=imextent, 
                                    cmap=cmap,
                                    vmin=vmin, 
                                    vmax=vmax,
                                    scale=scale, 
                                    gamma=gamma)
            
            if contourmap is not None:
                contextent = contmap.imextent
                contdata = contmap.data[0, :, :].T
                if smooth is not None:
                    contdata = ndimage.gaussian_filter(contdata, smooth)
                ax.contour(contdata, colors=ccolors, origin='lower', extent=contextent, 
                           levels=crms*clevels, linewidths=clw)
            if vlabelon:
                ax.set_xlabel(vlabel, fontsize=fontsize)
            if xlabelon:
                ax.set_ylabel(xlabel, fontsize=fontsize)
            ax.set_xlim(vlim)
            ax.set_ylim(xlim)
            if vticks is not None:
                ax.set_xticks(vticks)
            if xticks is not None:
                ax.set_vticks(xticks)
            if vlineon and vsys is not None:
                if subtract_vsys:
                    ax.plot([0, 0], [xlim[0], xlim[1]], color=refline_color, 
                        ls='dashed', lw=refline_width)
                else:
                    ax.plot([vsys, vsys], [xlim[0], xlim[1]], color=refline_color, 
                            ls='dashed', lw=refline_width)
            if xlineon:
                ax.plot([vlim[0], vlim[1]], [xcenter, xcenter], color=refline_color, 
                        ls='dashed', lw=refline_width)
        # tick parameters
        if tickson:
            ax.tick_params(which='both', direction=tickdirection, bottom=True, top=True, 
                           left=True, right=True, pad=9, labelsize=labelsize, color=tickcolor,
                           width=tickwidth)
        else:
            ax.tick_params(which='both', direction=tickdirection, bottom=False, top=False, 
                           left=False, right=False, pad=9, labelsize=labelsize)
        
        # define horizontal and vertical limits
        if offset_as_hor:
            horlim = xlim
            vertlim = vlim
        else:
            horlim = vlim
            vertlim = xlim
        
        hor_range = horlim[1] - horlim[0]  # horizontal range
        vert_range = vertlim[1] - vertlim[0]  # vertical range
            
        # set aspect ratio
        if cbarlabel is None:
            cbarlabel = "(" + _unit_plt_str(_apu_to_str(_to_apu(self.bunit))) + ")"
            
        # plot color bar and set aspect ratio (helper function)
        ax = _plt_cbar_and_set_aspect_ratio(ax=ax, 
                                            climage=climage, 
                                            cbarlabelon=cbarlabelon,
                                            cbarlabel=cbarlabel,
                                            cbarloc=cbarloc,
                                            cbaron=cbaron,
                                            cbarwidth=cbarwidth, 
                                            cbarpad=cbarpad,
                                            cbarticks=cbarticks,
                                            fontsize=fontsize, 
                                            labelsize=labelsize, 
                                            cbartick_width=cbartick_width, 
                                            cbartick_length=cbartick_length, 
                                            cbartick_direction=cbartick_direction, 
                                            aspect_ratio=aspect_ratio, 
                                            is_logscale=is_logscale,  # will change in the future 
                                            decimals=decimals)  # will change
        
        # plot resolution
        if plotres:
            res_x, res_y = (xres, vres) if offset_as_hor else (vres, xres)
            res_x_plt, res_y_plt = ax.transLimits.transform((res_x*0.5, res_y*0.5))-ax.transLimits.transform((0, 0))
            ax.errorbar(errbarloc[0], errbarloc[1], xerr=res_x_plt, yerr=res_y_plt, color=ccolors, 
                        capsize=errbar_capsize, capthick=errbar_captick, elinewidth=errbarlw, 
                        transform=ax.transAxes)
           
        # plot title, if necessary
        if title is not None:
            titlex = horlim[0] + titleloc[0]*hor_range
            titley = vertlim[0] + titleloc[1]*vert_range
            ax.text(x=titlex, y=titley, s=title, ha=ha, va=va, 
                    color=txtcolor, fontsize=title_fontsize)
        
        # save figure if parameters were specified
        if savefig:
            plt.savefig(**savefig)
            
        # plot image
        if plot:
            plt.show()
        
        return ax
    
    def __trim_data(self, vlim, xlim):
        if list(vlim) == [self.vaxis.min(), self.vaxis.max()] \
           and list(xlim) == [self.xaxis.min(), self.xaxis.max()]:
            return self.data[0]
        
        vmask = (min(vlim) <= self.vaxis) & (self.vaxis <= max(vlim))
        xmask = (min(xlim) <= self.xaxis) & (self.xaxis <= max(xlim))
        trimmed_data = self.data[0][vmask, :][:, xmask]
        return trimmed_data
    
    def line_info(self, **kwargs):
        """
        This method searches for the molecular line data from the Splatalogue database
        """
        if np.isnan(self.restfreq):
            raise Exception("Failed to find molecular line as rest frequency cannot be read.")
        return search_molecular_line(self.restfreq, unit="Hz", **kwargs)
    
    def get_hduheader(self):
        """
        To retrieve the header of the current FITS image. This method accesses the header 
        information of the original FITS file, and then modifies it to reflect the current
        status of this image object.

        Returns:
            The FITS header of the current image object (astropy.io.fits.header.Header).
        """
        self.__updateparams()
        return _get_hduheader(self)

    def get_hdu(self):
        """
        Get the primary HDU (astropy object) of the image.
        """
        return fits.PrimaryHDU(data=self.data, header=self.get_hduheader())

    def get_wcs(self):
        """
        Get the world coordinate system of the image (astropy object.)
        """
        return WCS(self.get_hduheader())
    
    def overwrite_header(self, new_vals=None, **kwargs):
        """
        Method to overwrite the existing keys of the header with new values.
        Parameters:
            new_vals (dict): a dictionary containing keys and values to be overwritten.
            **kwargs (dict): keyword arguments that will be overwritten in the header
        Return:
            self.header (dict): the updated header 
        """
        if new_vals is None and len(kwargs) == 0:
            raise ValueError("Header cannot be overwritten. Need to input a dictionary or keyword arguments.")
        if new_vals is not None:
            if isinstance(new_vals, dict):
                for key, value in new_vals.items():
                    if key in self.header:
                        self.header[key] = value
                    else:
                        print(f"'{key}' is not a valid keyword of the header and will be ignored.")
            else:
                raise TypeError("Please input a new dictionary as the header.")
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                if key in self.header:
                    self.header[key] = value
                else:
                    print(f"'{key}' is not a valid keyword of the header and will be ignored.")
        self.__updateparams()
        return self.header
    
    def exportfits(self, outname, overwrite=False):
        """
        Save the current image to a FITS file.

        This method exports the image data and header to a FITS file. If the specified 
        file name does not end with '.fits', it is appended. When 'overwrite' is False 
        and the file already exists, a number is appended to the file name to avoid 
        overwriting (e.g., 'filename(1).fits').

        Parameters:
            outname (str): The name of the output FITS file.
            overwrite (bool): If True, allows overwriting an existing file with the 
                              same name. If False, the file name is modified to 
                              prevent overwriting existing files.
        Returns:
            None
        """        
        # get header
        hdu_header = self.get_hduheader()
        
        # add file name extension if not in user input
        if not outname.endswith(".fits"):
            outname += ".fits"
        
        # if not overwrite, add (1), (2), (3), etc. to file name before '.fits'
        if not overwrite:
            outname = _prevent_overwriting(outname)
        
        # Write to a FITS file
        hdu = fits.PrimaryHDU(data=self.data, header=hdu_header)
        hdu.writeto(outname, overwrite=overwrite)
        print(f"File saved as '{outname}'.")

    def to_casa(self, *args, **kwargs):
        """
        Converts the PV diagram object into CASA image format. 
        Wraps the 'importfits' function of casatasks.

        Parameters:
            outname (str): The output name for the CASA image file. Must end with ".image".
            whichrep (int, optional): The FITS representation to convert. Defaults to 0.
            whichhdu (int, optional): The HDU (Header/Data Unit) number to convert. Defaults to -1.
            zeroblanks (bool, optional): Replace undefined values with zeros. Defaults to True.
            overwrite (bool, optional): Overwrite the output file if it already exists. Defaults to False.
            defaultaxes (bool, optional): Use default axes for the output CASA image. Defaults to False.
            defaultaxesvalues (str, optional): Default axes values, provided as a string. Defaults to '[]'.
            beam (str, optional): Beam parameters, provided as a string. Defaults to '[]'.

        Raises:
            ValueError: If 'outname' is not a string.

        Returns:
            None

        Example:
            image.to_casa("output_image")
        """
        to_casa(self, *args, **kwargs)


class Plot2D:
    """
    A class for handling two-dimensional plots such as SED plots, 
    spectral/spatial profiles, and intensity distribution diagrams.
    """
    def __init__(self, file=None, x=None, y=None, xerr=None, yerr=None,
                 header=None, bins=None, xlabel=None, xunit=None, 
                 ylabel=None, yunit=None, pandas=False, delimiter=None, 
                 xloc=None, yloc=None, comment="#", pd_header=None, 
                 scale="linear", quiet=False, **kwargs):
        """
        Constructor that initializes the Plot2D object with the provided data and parameters.

        Parameters:
        - file (str, optional): Path to the file containing data.
        - x (array-like, optional): Data for the x-axis.
        - y (array-like, optional): Data for the y-axis.
        - xerr (array-like, optional): Error bars for the x-axis.
        - yerr (array-like, optional): Error bars for the y-axis.
        - header (dict, optional): Dictionary containing header information.
        - bins (int or sequence, optional): Number of bins or bin edges for histograms.
        - xlabel (str, optional): Label for the x-axis.
        - xunit (str or astropy.units.Unit, optional): Unit for the x-axis data.
        - ylabel (str, optional): Label for the y-axis.
        - yunit (str or astropy.units.Unit, optional): Unit for the y-axis data.
        - pandas (bool, optional): Whether to use pandas for file reading.
        - delimiter (str, optional): Delimiter for reading the file.
        - xloc (int, optional): Column index for x data in the file.
        - yloc (int, optional): Column index for y data in the file.
        - comment (str, optional): Character used to indicate the start of a comment in the file.
        - pd_header (int, optional): Row number to use as the header (and column names) for pandas.
        - scale (str or tuple, optional): Scale type for the axes ('linear' or 'log').
        - **kwargs: Additional keyword arguments to be passed to the file reading function.

        Raises:
        - Exception: If no file, x, or y data is provided.
        - ValueError: If x or y is 0-sized.
        """
        
        # if nothing is specified, raise Exception
        if all(var is None for var in (file, x, y)):
            raise Exception("Please specify the file, x, and/or y.")
            
        # read file, if any
        if file:
            read_file_results = self.__read_file(file=file, 
                                                 pandas=pandas, 
                                                 delimiter=delimiter, 
                                                 xloc=xloc, 
                                                 yloc=yloc, 
                                                 comment=comment, 
                                                 header=pd_header, 
                                                 quiet=quiet,
                                                 **kwargs)
            xdata,  ydata, xlabel, ylabel, xunit, yunit = read_file_results
            x = xdata
        
        # initialize header
        init_header = {"filepath": "" if file is None else file,
                       "imagetype": "Plot2D",
                       "size": x.size,
                       "xlabel": "" if xlabel is None else xlabel,
                       "xunit": "" if not xunit else _apu_to_headerstr(_to_apu(xunit)),
                       "ylabel": "Counts" if ylabel is None else ylabel,
                       "yunit": "" if not yunit else _apu_to_headerstr(_to_apu(yunit)),
                       "scale": (scale.lower(), scale.lower()) if isinstance(scale, str) else scale,
                       "date": str(dt.datetime.now()).replace(" ", "T"),
                       "origin": "Generated by Astroviz."}
        
        immutable_keys = ("imagetype", "size")
        if header is not None and isinstance(header, dict):
            for key, value in header.items():
                if key in init_header:
                    if key in immutable_keys and header[key] != init_header[key]:
                        print(f"'{key}' is an immutable header parameter " + \
                               "and cannot be modified.")
                    else:
                        init_header[key] = value
                else:
                    print(f"'{key}' is not a valid header parameter" + \
                          " and will be ignored.")
                
        self.header = init_header
        
        # initialize x values
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.size < 1:
            raise ValueError("x should not be 0-sized.")
        if x.ndim > 1:
            warnings.warn("x is not 1-dimensional and will be flattened.")
            x = x.flatten()
        
        # convert into frequency graph if y is not provided
        if y is None:
            self.histogram = True
            if bins is None:
                bins = "auto"
            self.y, self.bins = np.histogram(x, bins=bins)
            self.yaxis = self.y
            self.data = x
            self.xaxis = self.x = (self.bins[:-1] + self.bins[1:]) / 2  # set midpoints as x
        else:
            self.histogram = False
            # if provided, initialize y values
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if y.size < 1:
                raise ValueError("y should not be 0-sized.")
            if y.ndim > 1:
                warnings.warn("y is not 1-dimensional and will be flattened.")
                y = y.flatten()
            
            # sort by x if x is not sorted
            if np.all(x[:-1] <= x[1:]):  # check if numpy array is sorted
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = y[sort_idx]
            
            # assign attributes
            self.x = self.xaxis = x
            self.y = self.yaxis = y
            self.data = (x, y)

        # parse error bars
        if xerr is not None:
            xerr = np.array(xerr)  # convert to numpy array
            if xerr.size != self.x.size:
                raise ValueError("Length of 'xerr' is different from length of 'x'.")
        if yerr is not None:
            yerr = np.array(yerr)  # convert to numpy array
            if yerr.size != self.y.size:
                raise ValueError("Length of 'yerr' is different from length of 'y'.")

        self.xerr = xerr  # set as attributes
        self.yerr = yerr

        if self.x.size != self.y.size:
            raise ValueError("Length of x is different from length of y.")
        
        # match attributes with header information
        self.__updateparams()
        
    def __updateparams(self):
        self.filepath = self.header["filepath"]
        self.size = self.header["size"]
        self.xunit = self.header["xunit"]
        self.xlabel = self.header["xlabel"]
        self.yunit = self.header["yunit"]
        self.ylabel = self.header["ylabel"]
        self.scale = self.header["scale"]
            
    def __len__(self):
        return self.size

    def max(self, axis="y"):
        """
        Public method to compute the maximum value along a specified axis.
        ------
        Parameters:
            axis (str): The axis along which to compute the maximum value.
                        It should be either 'x' or 'y' and is case-insensitive.
                        Default is 'y'.
        Returns:
            float: The maximum value along the specified axis, ignoring any NaN values.
        
        Raises:
            ValueError: If the provided axis is not 'x' or 'y'.
        """
        axis = axis.lower()  # make it case-insensitive
        if axis == "y":
            return np.nanmax(self.y)
        elif axis == "x":
            return np.nanmax(self.x)
        raise ValueError(f"Invalid axis '{axis}'. Expected 'x' or 'y'.")

    def min(self, axis="y"):
        """
        Public method to compute the minimum value along a specified axis.
        ------
        Parameters:
            axis (str): The axis along which to compute the maximum value.
                        It should be either 'x' or 'y' and is case-insensitive.
                        Default is 'y'.
        Returns:
            float: The minimum value along the specified axis, ignoring any NaN values.
        
        Raises:
            ValueError: If the provided axis is not 'x' or 'y'.
        """
        axis = axis.lower()
        if axis == "y":
            return np.nanmin(self.y)
        elif axis == "x":
            return np.nanmax(self.x)
        raise ValueError(f"Invalid axis '{axis}'. Expected 'x' or 'y'.")

    def copy(self):
        """
        This method creates a copy of the original image.
        """
        return copy.deepcopy(self)
    
    def concatenate(self, plot, equivalencies=None, inplace=False):
        """
        Public method to concatenate two datasets.
        ------
        Parameters:
            plot (Plot2D): plot to be concatenated
        Returns:
            concat_plot (Plot2D): plot that is the concatenation of two datasets.
        """
        concat_plot = self if inplace else self.copy()
        
        # check if the units of the plots are the same
        if self.xunit != plot.xunit:
            # try to convert the units
            if _to_apu(self.xunit).is_equivalent(_to_apu(plot.xunit), equivalencies=equivalencies):
                plot = plot.conv_xunit(self.xunit, equivalencies=equivalencies, inplace=False)
            else:
                # if that doesn't work, raise an error
                raise UnitConversionError("The x units of the two plots are not equivalent.")
            
        if self.yunit != plot.yunit:
            # try to convert the units
            if _to_apu(self.yunit).is_equivalent(_to_apu(plot.yunit), equivalencies=equivalencies):
                plot = plot.conv_yunit(self.yunit, equivalencies=equivalencies, inplace=False)
            else:
                # if that doesn't work, raise an error
                raise UnitConversionError("The y units of the two plots are not equivalent.")
            
        # add the axes if the units are the same
        new_x = np.concatenate(self.x, plot.x)
        new_y = np.concatenate(self.y, plot.y)
        
        # sort based on x
        xidx = np.argsort(new_x)
        new_x = new_x[xidx]
        new_y = new_y[xidx]
        
        # assign to new data
        concat_plot.x = new_x
        concat_plot.y = new_y
        
        return concat_plot
        
    def conv_xunit(self, xunit, equivalencies=None, inplace=True):
        """
        Public method to convert the unit of the x data.
        ------
        Parameters:
            xunit (str): the new unit
            equivalencies (astropy.units.equivalencies): equivalencies needed
        Returns:
            image (Plot2D): the plot with the units converted
        """
        image = self if inplace else self.copy()
        current_xunit = 1 * _to_apu(image.xunit)
        new_xunit = 1 * _to_apu(xunit)
        conversion_factor = current_xunit.to_value(new_xunit, equivalencies=equivalencies)
        image.x *= conversion_factor
        image.header["xunit"] = _apu_to_headerstr(new_xunit.unit)
        image.__updateparams()
        return image
    
    def conv_yunit(self, yunit, equivalencies=None, inplace=True):
        """
        Public method to convert the unit of the y data.
        ------
        Parameters:
            yunit (str): the new unit
            equivalencies (astropy.units.equivalencies): equivalencies needed
        Returns:
            image (Plot2D): the image with the units converted
        """
        image = self if inplace else self.copy()
        current_yunit = 1 * _to_apu(image.yunit)
        new_yunit = 1 * _to_apu(yunit)
        conversion_factor = current_xunit.to_value(new_yunit, equivalencies=equivalencies)
        image.y *= conversion_factor
        image.header["yunit"] = _apu_to_headerstr(new_yunit.unit)
        image.__updateparams()
        return image
    
    def set_xlabel(self, xlabel, inplace=True):
        image = self if inplace else self.copy()
        image.header["xlabel"] = xlabel
        image.__updateparams()
        return image
    
    def set_ylabel(self, ylabel, inplace=True):
        image = self if inplace else self.copy()
        image.header["ylabel"] = ylabel
        image.__updateparams()
        return image
    
    def set_xunit(self, xunit, inplace=True):
        image = self if inplace else self.copy()
        xunit = _apu_to_headerstr(_to_apu(xunit))
        image.header["xunit"] = xunit
        image.__updateparams()
        return image
    
    def set_yunit(self, yunit, inplace=True):
        image = self if inplace else self.copy()
        yunit = _apu_to_headerstr(_to_apu(yunit))
        image.header["yunit"] = yunit
        image.__updateparams()
        return image
    
    def stats_1d(self) -> None:
        """
        View the relevant 1D statistics of the data.
        """
        # remove nan values
        mask = (~np.isnan(self.x)) & (~np.isnan(self.y))
        xdata = self.x[mask]
        ydata = self.y[mask]
        
        # x-axis
        x_mean = np.mean(xdata)
        x_std = np.std(xdata)
        x_max = np.max(xdata)
        x_min = np.min(xdata)
        
        # y-axis
        y_mean = np.mean(ydata)
        y_std = np.std(ydata)
        y_max = np.max(ydata)
        y_min = np.min(ydata)
        
        # print values
        print("1D Statistics".center(25, "#"))
        print("X-axis")
        if self.xunit:
            print(f"Mean: {x_mean:.4f} [{self.xunit}]")
            print(f"SD: {x_std:.4f} [{self.xunit}]")
            print(f"Min: {x_min:.4f} [{self.xunit}]")
            print(f"Max: {x_max:.4f} [{self.xunit}]")
        else:
            print(f"Mean: {x_mean:.4f}")
            print(f"SD: {x_std:.4f}")
            print(f"Min: {x_min:.4f}")
            print(f"Max: {x_max:.4f}")
        print()  # skip a line
        
        print("Y-axis")
        if self.yunit:
            print(f"Mean: {y_mean:.4f} [{self.yunit}]")
            print(f"SD: {y_std:.4f} [{self.yunit}]")
            print(f"Min: {y_min:.4f} [{self.yunit}]")
            print(f"Max: {y_max:.4f} [{self.yunit}]")
        else:
            print(f"Mean: {y_mean:.4f}")
            print(f"SD: {y_std:.4f}")
            print(f"Min: {y_min:.4f}")
            print(f"Max: {y_max:.4f}")
        print(25*"#")
        
    def linear(self, x=True, y=True, inplace=True):
        """
        Convert the scale of the image from log to linear.
        ------
        Parameters:
            x (bool): True to convert the xaxis to linear scale
            y (bool): True to convert the yaxis to linear scale
            inplace (bool): True to modify the object in-place. False to return a new object.
        Returns:
            image (Plot2D): the new plot with scales converted.
        """
        image = self if inplace else self.copy()
        
        if x:
            if image.header["scale"][0] == "log":
                image.x = 10**(image.x)
                image.header["scale"] = ("linear", image.header["scale"][1])
            elif image.header["scale"][0] == "linear":
                warnings.warn("the x-axis of the plot is already in " + \
                      "linear scale and will not be converted.")
        if y:
            if image.header["scale"][1] == "log":
                image.y = 10**(image.y)
                image.header["scale"] = (image.header["scale"][0], "linear")
            elif image.header["scale"][1] == "linear":
                warnings.warn("the y-axis of the plot is already in linear scale " + \
                      "and will not be converted.")
        
        # update parameters to match header
        image.__updateparams()
        
        return image
        
    def logscale(self, x=True, y=True, inplace=True):
        """
        Convert the scale of the image from linear to log.
        ------
        Parameters:
            x (bool): True to convert the xaxis to log scale
            y (bool): True to convert the yaxis to log scale
            inplace (bool): True to modify the object in-place. False to return a new object.
        Returns:
            image (Plot2D): the new plot with scales converted
        """
        image = self if inplace else self.copy()
        
        if x:
            if image.header["scale"][0] == "linear":
                image.x = np.log10(image.x)
                image.header["scale"] = ("log", image.header["scale"][1])
            elif image.header["scale"][0] == "log":
                warnings.warn("the x-axis of the plot is already in log scale and will not be converted.")
        if y:
            if image.header["scale"][1] == "linear":
                image.y = np.log10(image.y)
                image.header["scale"] = (image.header["scale"][0], "log")
            elif image.header["scale"][1] == "log":
                warnings.warn("the y-axis of the plot is already in log scale and will not be converted.")
        
        # update parameters to match header
        image.__updateparams()
        
        return image
    
    def set_threshold(self, threshold, inplace=True):
        """
        Public method to remove data points with y values that lie 
        below the given threshold.
        threshold (float): the threshold
        inplace (bool): True to modify the plot in-place. 
                        False to return a new plot.
        """
        image = self if inplace else self.copy()
        mask = image.y > threshold
        image.y = image.y[mask]
        image.x = image.x[mask]
        return image
        
    def trim(self, xlim=None, ylim=None, exclude=False, inplace=False):
        """
        Private method to trim the data. 
        Parameters:
            xlim (list[float, ...] | list[list[float, float], ...]): range of x values
            ylim (list[float, ...] | list[list[float, float], ...]): range of y values
            exclude (bool): True to exclude specified ranges. False to include.
            inplace (bool): True to modify the plot in-place. False to return a new plot.
        Returns:
            image (Plot2D): the trimmed image.
        """
        warnings.warn("The method 'trim' is still in testing.")
        image = self if inplace else self.copy()
        if xlim is None and ylim is None:
            warnings.warn("'xlim' and 'ylim' are not specified." + \
                  "The original image will be returned.")
            return image
        else:
            # convert to numpy arrays
            xlim = np.array(xlim)
            ylim = np.array(ylim)
            
            mask = np.full(image.x.shape, False)  # initialize mask
            if xlim is not None:
                # mask in x direction
                if xlim.ndim == 1 and xlim.size == 2: 
                    mask = mask | ((xlim[0]<=image.x)&(image.x<=xlim[1]))
                elif xlim.ndim == 2:
                    for (x1, x2) in xlim:
                        mask = mask | ((x1<=image.x)&(image.x<=x2))
                else:
                    raise ValueError("Invalid range of x values provided.")
                # mask in y direction
                if ylim.ndim == 1 and ylim.size == 2:
                    mask = mask | ((ylim[0]<=image.y)&(image.y<=ylim[1]))
                elif ylim.ndim == 2:
                    for (y1, y2) in ylim:
                        mask = mask | ((y1<=image.y)&(image.y<=y2))
                else:
                    raise ValueError("Invalid range of y values provided.")
                    
            if exclude:
                mask = ~mask
                
            # trim
            image.x = image.x[mask]
            image.y = image.y[mask]
            
        return image
    
    def __read_file(self, file, pandas=False, delimiter=None, 
                    xloc=None, yloc=None, comment="#", header=None, 
                    quiet=False, **kwargs):
        
        if not os.path.exists(file):
            if not quiet:
                print(f"Given directory '{file}' does not exist as a relative directory. " + \
                       "Recursively finding file...")
            maybe_filename = _find_file(file)
        if maybe_filename is not None:
            file = maybe_filename
            if not quiet:
                print(f"Found a matching filename: '{file}'")
        else:
            raise FileNotFoundError(f"Filename '{file}' does not exist.")
        
        def get_str_between(s: str, start: str, end: str) -> str:
            start_idx: int = s.find(start)
            if start_idx == -1:
                return ""
            start_idx += len(start)
            end_idx = s.rfind(end)
            if end_idx == -1:
                return ""
            return s[start_idx:end_idx]
        
        if pandas:
            xunit = xlabel = yunit = ylabel = ""  # initalize values
            is_spec_profile = False
            with open(file, "r") as f:
                for line in f:
                    lower_line = line.lower()
                    if "spectral profile" in lower_line:
                        is_spec_profile = True
                    if line.startswith("#"):
                        if "xlabel" in lower_line:
                            xunit = get_str_between(line, "[", "]")
                            if is_spec_profile and xunit.lower() == "km/s" \
                               and "radio velocity" in lower_line:
                                xlabel = "radio velocity"
                            else:
                                xlabel = get_str_between(line, ": ", " [")
                        elif "ylabel" in lower_line:
                            yunit = get_str_between(line, "[", "]")
                            if is_spec_profile and yunit.lower() == "kelvin":
                                ylabel = "intensity"
                            else:
                                ylabel = get_str_between(line, ": ", " [")
                    else:
                        break
                    
            df = pd.read_csv(file,
                             delimiter=" " if delimiter is None else delimiter, 
                             comment=comment, 
                             header=header,
                             **kwargs)
            xdata = np.array(df.iloc[:, (0 if xloc is None else xloc)], dtype=float)
            ydata = np.array(df.iloc[:, (1 if yloc is None else yloc)], dtype=float)
        else:
            with open(file, "r") as f:
                lines = np.array(f.readlines())
            lines = np.char.strip(lines, "\n")
            mask = (np.char.find(lines, comment) == -1)
            lines_with_data = lines[mask]
            lines_with_comments = lines[~mask]
            
            if delimiter is None:
                # try different delimiters and see if one of them works!
                delims_to_try = (" ", "\t", ",", ", ", ":", "/")
                for delimiter in delims_to_try:
                    try:
                        lines_with_data = np.char.split(lines_with_data, delimiter)
                        lines_with_data = np.array(lines_with_data.tolist(), dtype=float)
                        xdata = lines_with_data[:, (0 if xloc is None else xloc)]
                        ydata = lines_with_data[:, (1 if yloc is None else yloc)]
                        break  # exit loop if there is no exception
                    except ValueError:
                        continue  # try next delimiter
                else:
                    raise ValueError("None of the delimiters worked " + \
                                     "to split the data correctly.")
            else:
                lines_with_data = np.char.split(lines_with_data, delim)
                lines_with_data = np.array(lines_with_data.tolist(), dtype=float)
                xdata = lines_with_data[:, 0]
                ydata = lines_with_data[:, 1]
            
            xunit = xlabel = yunit = ylabel = ""  # initialize values
            is_spec_profile = False
            for line in lines_with_comments:
                lower_line = line.lower()
                if "spectral profile" in lower_line:
                    is_spec_profile = True
                if line.startswith("#"):
                    if "xlabel" in lower_line:
                        xunit = get_str_between(line, "[", "]")
                        if is_spec_profile and xunit.lower() == "km/s" \
                           and "radio velocity" in lower_line:
                            xlabel = "radio velocity"
                        else:
                            xlabel = get_str_between(line, ": ", " [")
                    elif "ylabel" in lower_line:
                        yunit = get_str_between(line, "[", "]")
                        if is_spec_profile and yunit.lower() == "kelvin":
                            ylabel = "intensity"
                        else:
                            ylabel = get_str_between(line, ": ", " [")
                else:
                    break
        
        return xdata, ydata, xlabel, ylabel, xunit, yunit
    
    def SED(self, distance):
        """
        Derive the bolometric temperature and luminosity from the spectral energy distribution.
        Integration is used rather than fitting.
        Parameters:
            distance (float): distance to the target object
        Returns:
            mean_freq_GHz (float): the intensity-weighted mean frequency in GHz
            Tbol (float): the bolometric 
            Lbol
        """
        # get x, y axes, in order of x axis to correctly integrate
        sort_idx = np.argsort(self.x)
        x = self.x[sort_idx]
        y = self.y[sort_idx]
        
        # parse units
        xunit = _to_apu(self.xunit)
        yunit = _to_apu(self.yunit)
        
        # parse frequency axis
        if xunit.is_equivalent(u.Hz, equivalencies=u.spectral()):
            freq = x * (1*xunit).to_value(u.Hz, equivalencies=u.spectral())
        else:
            raise Exception("Incorrect x-axis unit. " + \
                            "Must be a unit of frequency (e.g., Hz).")
            
        # parse flux axis
        if yunit.is_equivalent(u.Jy):
            flux = y * (1*yunit).to_value(u.Jy)
        elif yunit.is_equivalent(u.Hz*u.Jy, equivalencies=u.spectral()):
            flux = y * (1*yunit).to_value(u.Hz*u.Jy)
            flux /= freq
        else:
            raise Exception("Incorrect y-axis unit. Must be a unit of flux (e.g., Jy).")
            
        # intensity-weighted mean frequency
        int_flux_dfreq = np.trapz(y=flux, x=freq)  # integrate flux wrt freq
        mean_freq = np.trapz(y=freq*flux, x=freq)/int_flux_dfreq  # Hz
        mean_freq_GHz = mean_freq / 1e9
        
        # use mean frequency to calculate bolometric temperature
        Tbol = (1.25e-11 * mean_freq)  # K
        
        # bolometric luminosity
        if isinstance(distance, u.Quantity):
            if not (distance.unit).is_equivalent(u.pc):
                raise Exception("Unit of distance provided is not a " + \
                                "unit of length (e.g., pc).")
        else:
            distance *= u.pc
        Lbol = (4*np.pi*(distance**2) * int_flux_dfreq*u.Jy*u.Hz).to_value(u.Lsun)
        
        # print results
        print("Spectral Energy Distribution".center(40, "#"))
        print(f"Mean frequency: {(mean_freq_GHz):.4f} [GHz]")
        print(f"Tbol: {Tbol:.2f} [K]")
        print(f"Lbol: {Lbol:.2f} [L_sun]")
        print(40*"#", end="\n\n")
        
        return (mean_freq_GHz, Tbol, Lbol)
        
    def fit_SED(self):
        raise Exception("Not implemented yet.")
        
    def imview(self, title=None, xlim=None, ylim=None, legendon=None, legendsize=6,
               legendloc="best", bbox_to_anchor=(0.6, 0.95), xticks=None, yticks=None,
               xlabelon=True, ylabelon=True, linewidth=None, linestyle="-", scale=("linear", "linear"),
               linecolor=None, model_linewidth=None, model_linestyle="-", xlabel=None, ylabel=None, 
               model_linecolor="tomato", ha="center", va="top", textcolor="k",
               threshold=None, fit_xlim=None, fit_ylim=None, figsize=(2.76, 2.76), 
               gauss_fit=False, components=1, p0=None, fixed_values=None, line_fit=False, 
               alternative="two-sided", title_loc=(0.1, 0.875), plot_threshold=True, 
               labelsize=7., curve_fit=None, threshold_color="gray", threshold_ls="dashed",
               threshold_lw=None, fontsize=None, plot_predicted=True, dpi=600, color=None,
               plot_type="line", axeslw=0.7, ticksize=3, labels=["Observation", "Model"], 
               aspect_ratio=1., interpolation=None, linspace_num=10000, residuals_on=True,
               capsize=1., elinewidth=None, fmt='none', marker=".", tick_direction="in",
               ecolor=None, markeredgewidth=None, top_ticks=True, bottom_ticks=True, 
               plot_ebars=None, left_ticks=True, right_ticks=True, savefig=None, 
               linear_ticks=True, with_multiple=False, ax=None, plot=True, **kwargs):
        """
        Plot the 2D data with various customization options.

        Parameters:
        - title (str, optional): Title of the plot.
        - xlim (list or tuple, optional): Limits for the x-axis.
        - ylim (list or tuple, optional): Limits for the y-axis.
        - legendon (bool, optional): Whether to display the legend.
        - legendsize (int, optional): Font size of the legend.
        - legendloc (str, optional): Location of the legend.
        - bbox_to_anchor (tuple, optional): Bounding box anchor for the legend.
        - xticks (list, optional): Tick values for the x-axis.
        - yticks (list, optional): Tick values for the y-axis.
        - xlabelon (bool, optional): Whether to display the x-axis label.
        - ylabelon (bool, optional): Whether to display the y-axis label.
        - linewidth (float, optional): Line width for the data plot.
        - linestyle (str, optional): Line style for the data plot.
        - linecolor (str, optional): Line color for the data plot.
        - model_linewidth (float, optional): Line width for the model plot.
        - model_linestyle (str, optional): Line style for the model plot.
        - model_linecolor (str, optional): Line color for the model plot.
        - ha (str, optional): Horizontal alignment for text.
        - va (str, optional): Vertical alignment for text.
        - textcolor (str, optional): Color for text.
        - threshold (float, optional): Threshold value for a horizontal line.
        - fit_xlim (list or tuple, optional): Limits for fitting on the x-axis.
        - fit_ylim (list or tuple, optional): Limits for fitting on the y-axis.
        - figsize (tuple, optional): Size of the figure.
        - gauss_fit (bool, optional): Whether to perform Gaussian fitting.
        - components (int, optional): Number of components for Gaussian fitting.
        - p0 (list, optional): Initial guess for fitting parameters.
        - fixed_values (dict, optional): Fixed values for fitting parameters.
        - line_fit (bool, optional): Whether to perform linear fitting.
        - alternative (str, optional): Alternative hypothesis for linear fitting.
        - title_loc (tuple, optional): Location for the title.
        - plot_threshold (bool, optional): Whether to plot the threshold line.
        - labelsize (float, optional): Font size for labels.
        - curve_fit (callable, optional): Function for custom curve fitting.
        - threshold_color (str, optional): Color for the threshold line.
        - threshold_ls (str, optional): Line style for the threshold line.
        - threshold_lw (float, optional): Line width for the threshold line.
        - fontsize (float, optional): Font size for the plot text.
        - plot_predicted (bool, optional): Whether to plot predicted values.
        - dpi (int, optional): Dots per inch for the figure.
        - plot_type (str, optional): Type of plot ('line' or 'scatter').
        - borderwidth (float, optional): Width of the plot border.
        - ticksize (float, optional): Size of the plot ticks.
        - labels (list, optional): Labels for the data and model.
        - plot (bool, optional): Whether to display the plot.
        - **kwargs: Additional keyword arguments for plotting.

        Returns:
        - ax (matplotlib.axes.Axes): The axes object of the plot.

        Raises:
        - ValueError: If 'xlim' or 'ylim' have a length greater than 2.
        """
        # get x and y axes
        xdata = self.x
        ydata = self.y
        xerr = self.xerr 
        yerr = self.yerr 

        if plot_ebars is None:
            plot_ebars = not (xerr is None and yerr is None)
        
        # remove bad values
        mask = (~np.isnan(xdata)) & (~np.isnan(ydata)) 
        if not np.all(mask):
            xdata = xdata[mask]
            ydata = ydata[mask]
            if xerr is not None:
                xerr = xerr[mask]
            if yerr is not None:
                yerr = yerr[mask]
            
        # initialize fontsize
        if fontsize is None:
            fontsize = labelsize

        if linewidth is None:
            linewidth = axeslw
        
        if threshold_lw is None:
            threshold_lw = linewidth

        if elinewidth is None:
            elinewidth = linewidth

        if model_linewidth is None:
            model_linewidth = linewidth

        if linecolor is None:
            if interpolation:
                linecolor = "cornflowerblue"
            else:
                linecolor = "k"

        if plot_ebars and ecolor is None:
            if color is None:
                ecolor = linecolor
            else:
                ecolor = color 
        if color is None:
            color = linecolor
            
        # set image parameters
        if ax is None:
            params = {'axes.labelsize': labelsize,
                      'axes.titlesize': labelsize,
                      'font.size': fontsize,
                      'figure.dpi': dpi,
                      'legend.fontsize': legendsize,
                      'xtick.labelsize': labelsize,
                      'ytick.labelsize': labelsize,
                      'font.family': _fontfamily,
                      "mathtext.fontset": _mathtext_fontset,
                      'mathtext.tt': _mathtext_tt,
                      'axes.linewidth': axeslw,
                      'xtick.major.width': axeslw,
                      'ytick.major.width': axeslw,
                      'figure.figsize': figsize,
                      'xtick.major.size': ticksize,
                      'ytick.major.size': ticksize,
                      }
            rcParams.update(params)
            
        # initialize xlim and ylim
        if not with_multiple:
            if xlim is None:
                xmin = np.nanmin(xdata)
                xmax = np.nanmax(xdata)
                xrange = xmax - xmin
                xlim = [xmin-0.1*xrange, xmax+0.1*xrange]
            elif len(xlim) > 2:
                raise ValueError("'xlim' cannot have a length greater than 2.")
            elif xlim[0] > xlim[1]:  # swap if xlim is invalid
                warnings.warn("'xlim' is not correctly specified. Swapping...")
                if isinstance(xlim, tuple):
                    xlim = list(xlim)
                xlim[0], xlim[1] = xlim[1], xlim[0]
                
            if ylim is None:
                ymin = np.nanmin(ydata)
                ymax = np.nanmax(ydata)
                yrange = ymax - ymin
                ylim = [ymin-0.1*yrange, ydata.max()+0.1*yrange]
            elif len(xlim) > 2:
                raise ValueError("'xlim' cannot have a length greater than 2.")
            elif ylim[0] > ylim[1]:  # swap if ylim is invalid
                warnings.warn("'ylim' is not correctly specified. Swapping...")
                if isinstance(ylim, tuple):
                    ylim = list(ylim)
                ylim[0], ylim[1] = ylim[1], ylim[0]
        
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)

        # interpolation:
        if isinstance(interpolation, str):
            if not interpolation.islower():
                interpolation = interpolation.lower()  # make it case-insensitive

            if interpolation in ("linear", "nearest", "nearest-up", "zero", "slinear", 
                               "quartic", "cubic", "previous", "next"):
                # create callable
                func: Callable = interpolate.interp1d(x=xdata, y=ydata, kind=interpolation)
                x_interp = np.linspace(xdata.min(), xdata.max(), linspace_num)
                y_interp = func(x_interp)

            else:
                raise ValueError(f"Invalid interpolation method: {interpolation}. Must be one of 'linear', 'nearest', " + \
                                  "'nearest-up', 'zero', 'slinear', 'quartic', 'cubic', 'previous', or 'next'.")
        else:
            # do nothing if user does not wish to interpolate data:
            x_interp = xdata
            y_interp = ydata

        # parse plot type -> make it case-insensitive
        if not plot_type.islower():  
            plot_type = plot_type.lower()

        # start plotting data
        ax = _customizable_scale(ax=ax, xdata=x_interp, ydata=y_interp,
                                 scale=scale, xticks=xticks, yticks=yticks,
                                 plot_type=plot_type, linewidth=linewidth,
                                 linestyle=linestyle, linecolor=linecolor,
                                 label=labels[0], marker=marker, color=color,
                                 plot_ebars=plot_ebars, linear_ticks=linear_ticks, 
                                 **kwargs)
        
        # plot error bars
        if xerr is not None or yerr is not None:
            if markeredgewidth is None:
                markeredgewidth = elinewidth
            ax.errorbar(x=xdata, y=ydata, xerr=xerr, yerr=yerr, 
                        fmt=fmt, marker=marker, ecolor=ecolor, elinewidth=elinewidth, 
                        capsize=capsize, markeredgewidth=markeredgewidth, 
                        label=labels[0])

        # set labels
        if xlabelon:
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=fontsize)
            elif self.xlabel:
                xlabel = self.xlabel.title()
                if self.xunit:
                    xunit = _unit_plt_str(_apu_to_str(_to_apu(self.xunit)))
                    xlabel += " (" + xunit +")" 
                ax.set_xlabel(xlabel, fontsize=fontsize)
        
        if ylabelon:
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=fontsize)
            elif self.ylabel:
                ylabel = self.ylabel.title()
                if self.yunit:
                    yunit = _unit_plt_str(_apu_to_str(_to_apu(self.yunit)))
                    ylabel += " (" + yunit +")" 
                ax.set_ylabel(ylabel, fontsize=fontsize)
            
        # plot threshold as horizontal line
        if plot_threshold and threshold is not None:
            ax.axhline(y=threshold, color=threshold_color, ls=threshold_ls, lw=threshold_lw)
        
        # perform fitting if needed
        if threshold is not None:
            if fit_ylim is None:
                fit_ylim = [ydata.min()-np.std(ydata), threshold]
            else:
                warnings.warn("'threshold' and 'fit_ylim' parameters were both given. " + \
                              "Only 'fit_ylim' will be effective.")
        
        if (gauss_fit or line_fit or curve_fit):
            if gauss_fit:
                # do Gaussian fitting
                popt, perr, func = self.gauss_fit(fit_xlim=fit_xlim, 
                                                  fit_ylim=fit_ylim, 
                                                  components=components, 
                                                  p0=p0, 
                                                  fixed_values=fixed_values)
            elif line_fit:
                popt, perr = self.linear_regression(alternative=alternative)
                func = lambda x, slope, y_int: slope*x + y_int
            elif curve_fit:
                popt, perr = self.curve_fit(function=curve_fit, p0=p0)
                func = curve_fit
                
            # predict and plot model
            smooth_x, predicted_y = self.__predict(popt, func, xlim=None, linspace_num=linspace_num)
            ax.plot(smooth_x, predicted_y, lw=model_linewidth, label=labels[1],
                    ls=model_linestyle, color=model_linecolor)
            
            if legendon is None:  # modify default value
                legendon = True

            # # plot residuals
            # if residuals_on:
            #     raise Exception("Not implemented yet.")
            #     # # calculate residuals
            #     # residuals = ydata - func(xdata, *popt)

            #     # # plot residuals
            #     # height = 
            #     # bbox_to_anchor = ()
            #     # residuals_ax = inset_axes(ax, 
            #     #                           width="100%", 
            #     #                           height=, 
            #     #                           loc="", 
            #     #                           borderpad=0.,
            #     #                           bbox_to_anchor=,
            #     #                           bbox_transform=ax.transAxes)

            
        if legendon is not None and legendon:
            ax.legend(frameon=False, loc=legendloc, bbox_to_anchor=bbox_to_anchor)
            
        # plot title
        if not with_multiple:
            if title is not None:
                # determine location on the xy axes
                xlim_range = xlim[1] - xlim[0]
                ylim_range = ylim[1] - ylim[0]
                title_x = xlim[0] + xlim_range * title_loc[0]
                title_y = ylim[0] + ylim_range * title_loc[1]
                ax.text(title_x, title_y, title, color=textcolor, fontsize=fontsize)
        
            # set xlim and ylim
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        
            # tick parameters:
            ax.tick_params(which='both', direction=tick_direction, bottom=bottom_ticks, 
                           top=top_ticks, left=left_ticks, right=right_ticks, 
                           colors="k", labelrotation=0, labelcolor="k", labelsize=labelsize)
            
            # set aspect ratio
            if aspect_ratio:
                horizontal_limit = ax.get_xlim()
                horizontal_range = horizontal_limit[1] - horizontal_limit[0]
                vertical_limit = ax.get_ylim()
                vertical_range = vertical_limit[1] - vertical_limit[0]
                real_aspect_ratio = abs(1./aspect_ratio*horizontal_range/vertical_range)
                ax.set_aspect(real_aspect_ratio)
            
            if savefig:
                plt.savefig(**savefig)

            if plot:
                plt.show()
        
        return ax
    
    def linear_regression(self, alternative="two-sided"):
        """
        Perform a linear regression on the x and y data of the plot.

        Parameters:
        - alternative (str, optional): Defines the alternative hypothesis.
          The default value is "two-sided". Other options are "less" and "greater".

        Returns:
        - output (tuple): A tuple containing two lists:
            - [slope, intercept]: The slope and intercept of the regression line.
            - [slope_err, intercept_err]: The standard error of the slope and intercept.

        Prints:
        - The slope and its error with units.
        - The intercept and its error with units.
        - The correlation coefficient (r) and its square (r^2).
        - The p-value for the hypothesis test.
        """
        result = linregress(x=self.x, y=self.y, alternative=alternative)
        slope = result.slope
        intercept = result.intercept
        rvalue = result.rvalue
        pvalue = result.pvalue
        slope_err = result.stderr
        intercept_err = result.intercept_stderr
        
        print("Linear Regression".center(40, "#"))
        if self.yunit and self.xunit:
            slope_unit = f"[{self.yunit}/{self.xunit}]"
        elif self.yunit:
            slope_unit = f"[{self.yunit}]"
        elif self.xunit:
            slope_unit = f"[1/{self.xunit}]"
        else:
            slope_unit = ""
        print(f"slope: {slope:.4f} +/- {slope_err:.4f} " + slope_unit)
        if self.yunit:
            print(f"intercept: {intercept:.4f} +/- {intercept_err:.4f} [{self.yunit}]")
        else:
            print(f"intercept: {intercept:.4f} +/- {intercept_err:.4f}")
        print(f"r: {rvalue:.4f}")
        print(f"r2: {(rvalue**2):.4f}")
        print(f"p-value ({alternative}): {pvalue:.4f}")
        print(40*"#", end="\n\n")
        
        output = ([slope, intercept], [slope_err, intercept_err])
        return output
    
    def curve_fit(self, function, p0=None, **kwargs):
        """
        Fit the data to a given callable function.
        Parameters:
            function (callable): the function that will be fitted
            p0 (list): a list of parameters to be inputted
            **kwargs: other keyword arguments for 'curve_fit' function of scipy
        Returns:
            popt: the best-fit parameters.
            perr: errors associated with the best-fit parameters.
        """
        mask = (~np.isnan(self.x))&(~np.isnan(self.y))
        
        
        
        # perform fitting based on a function
        try:
            popt, pcov = curve_fit(f=function, xdata=self.x, ydata=self.y, 
                                   p0=p0, **kwargs)
        except:
            print("Failed to fit the given function.")
            return None
        
        # find standard errors of the parameters:
        perr = np.sqrt(np.diag(pcov))
        
        # retreve parameter names by inspecting the callable:
        param_names = tuple(inspect.signature(function).parameters.keys())
        
        # print fitting results:
        print("Fitting Results".center(40, "#"))
        for i, param in enumerate(param_names):
            print(f"{param}: {popt[i]:.4f} +/- {perr[i]:.4f}")
        print(40*"#")
        
        return popt, perr

    def gauss_fit(self, fit_xlim=None, fit_ylim=None, components=1, p0=None, fixed_values=None):
        """
        Public method to fit a Gaussian function to the data.
        --------
        Parameters:
            fit_range (2d array): 
            components (int): number of gaussian components to be fitted
            p0 (list[float]): Initial parameters. 
                              Order is [x, amplitude, fwhm].
                              If there are multiple components, set the initial values to:
                              [[x1, a1, fwhm1], [x2, a2, fwhm2], [x3, a3, fwhm3]], etc.
                              Default is None (guess based on data).
                              If components > 1, default is all ones.
            fixed_values (list[None | float]): Values that will be fixed while fitting.
                                               None to not fix the value.
                                               If there are multiple components, set to:
                                               [[x1, a1, fwhm1], [x2, a2, fwhm2], [x3, a3, fwhm3]], etc.
        Returns:
            popt (list[float]): List of best-fit values.
            pcov (list[float]): SD associated with the best-fit values.
        """
        # remove invalid values while fitting
        mask = (~np.isnan(self.x)) & (~np.isnan(self.y))
        
        # remove values that lie outside the specified ranges
        if fit_xlim is not None:  
            fit_xlim = np.asarray(fit_xlim).flatten()
            xmask = np.array([False]*self.x.size)  # initialize
            for (x1, x2) in zip(fit_xlim[::2], fit_xlim[1::2]):
                xmask = xmask | ((x1<=self.x) & (self.x<=x2))
            mask = mask & xmask

        if fit_ylim is not None:
            fit_ylim = np.asarray(fit_ylim).flatten()
            ymask = np.array([False]*self.y.size)  # initialize
            for (y1, y2) in zip(fit_ylim[::2], fit_ylim[1::2]):
                ymask = ymask | ((y1<=self.y) & (self.y<=y2))
            mask = mask & ymask

        # apply mask
        masked_x = self.x[mask]
        masked_y = self.y[mask]

        # convert to array and reshape
        if fixed_values is not None:
            if not isinstance(fixed_values, np.ndarray):
                fixed_values = np.array(fixed_values, dtype=object)
            if fixed_values.shape != (components, 3):
                fixed_values = fixed_values.reshape((components, 3))
        elif fixed_values is None:
            fixed_values = np.array([[None]*3]*components)
        
        # check whether values are fixed:
        is_fixed = list(list(param is not None for param in comp) \
                        for comp in fixed_values.tolist())
        
        # reassign p0 to a parsable reshape
        if p0 is not None:
            if isinstance(p0, list) and not isinstance(p0[0], list):  # check if p0 is 1d list:
                p0 = [p0]  # convert to 2d list
            elif isinstance(p0, np.ndarray):
                # only acceptable if p0 is 2-dimensional
                if p0.ndim == 1:
                    p0 = p0[np.newaxis, :]
                elif p0.ndim > 2:
                    raise ValueError("p0 should not be more than 2 dimensions.")

            for comp in p0:   # add nan values to p0:
                if len(comp) != 3:
                    comp += [np.nan]*(3-len(comp))

            flattened_p0 = np.array(p0).flatten()  # flatten
            p0 = flattened_p0[~np.isnan(flattened_p0)]
        elif p0 is None and components == 1:
            guess_x0 = np.average(masked_x, weights=masked_y)
            sd_weights = masked_y / np.sum(masked_y)
            guess_std = np.sqrt(np.sum((masked_x - guess_x0)**2 * sd_weights))
            guess_fwhm = guess_std * 2.355  # std -> fwhm
            guess_a = np.max(masked_y)
            p0 = [guess_val for i, guess_val in enumerate((guess_x0, guess_a, guess_fwhm)) \
                  if not is_fixed[0][i]]  # Need to fix this line
            print(f"Initial Guess: {p0}")

        local_namespace = {"np": np}

        # fix values
        for i in range(components):
            params = ("x0", "a", "fwhm")
            valid_params = [param for j, param in enumerate(params) if not is_fixed[i][j]]
            valid_params = ", ".join(valid_params)
            fwhm_in_fn = fixed_values[i][2] if is_fixed[i][2] else 'fwhm'
            amp_in_fn = fixed_values[i][1] if is_fixed[i][1] else 'a'
            center_in_fn = fixed_values[i][0] if is_fixed[i][0] else 'x0'
            exec_code = f"def gauss_component{i+1}(x, {valid_params}): \n"
            exec_code += f"  sigma = {fwhm_in_fn} / 2.355 \n"
            exec_code += f"  return {amp_in_fn}*np.exp(-(x-{center_in_fn})**2/(2*sigma**2))"

            exec(exec_code, local_namespace)

        # define function that is the sum of all gaussian components
        exec_code = f"def gauss(x, "
        for i in range(components):
            params = (f"x0_{i+1}", f"a{i+1}", f"fwhm{i+1}")
            valid_params = [param for j, param in enumerate(params) if not is_fixed[i][j]]
            valid_params = ", ".join(valid_params)
            exec_code += valid_params
            if i != components - 1:
                exec_code += ", "

        exec_code += "): \n"

        for i in range(components):
            if i == 0:
                exec_code += "  return "
            params = (f"x0_{i+1}", f"a{i+1}", f"fwhm{i+1}")
            valid_params = [param for j, param in enumerate(params) if not is_fixed[i][j]]
            valid_params = ", ".join(valid_params)
            exec_code += f"gauss_component{i+1}(x, {valid_params})"
            if i != components - 1:
                exec_code += "+"
        
        exec(exec_code, local_namespace)
        gauss = local_namespace['gauss']
        
        # start fitting gaussian function
        try:
            popt, pcov = curve_fit(f=gauss, xdata=masked_x, ydata=masked_y, p0=p0)
        except:
            print("Fitting failed.")
            return None
        perr = np.sqrt(np.diag(pcov))

        # start printing results
        j = 0  # counter
        print("Gaussian Fitting Results".center(40, "#"))
        for i in range(components):
            print(f"Component {i+1}")
            if not is_fixed[i][0]:
                print(f"Center: {popt[j]:.4f} +/- {perr[j]:.4f}")
                j += 1
            if not is_fixed[i][1]:
                print(f"Amplitude: {popt[j]:.4f} +/- {perr[j]:.4f}")
                j += 1
            if not is_fixed[i][2]:
                print(f"FWHM: {popt[j]:.4f} +/- {perr[j]:.4f}")
                j += 1
            if i != components - 1:  # if i corresponds to the last index
                print()
        print(40*"#", end="\n\n")

        return popt, perr, gauss
    
#     def hf_fit(self, restfreq, components=3):
#         """
#         Perform hyperfine fitting to the spectrum.
#         Parameters:
#             restfreq (float): the rest frequnecy in GHz.
#         """
#         # physical constants
#         clight = const.c.cgs  # speed of light in vacuum
#         h = const.h.cgs  # planck constant
#         k_B = const.k_B.cgs  # Boltzmann's constant
#         T_bg = 2.725*u.K  # cosmic microwave background temperature
        
#         if components == 3:
            
    def __predict(self, popt, func, xlim=None, linspace_num=10000):
        """
        Private function to predict values based on results of fitting.
        (AKA last forward prop)
        ------
        Parameters:
            popt (list[float]): a list of best-fit parameters
            func (callable): function that gives y given x (that can be used ot predict y-values)
            xlim (list[float], optional): The [min, max] values of x.
                                          Default is to set as (min of x - 1 sd, max of x + 1 sd)
                                            
            linspace_num (int): spacing between two x values in y
        Returns:
            smoothx (np.ndarray[float]): array of x values that correspond to the predicted y values
            predicted_y
        """
        if xlim is None:
            masked_x = self.x[~np.isnan(self.x)]  # remove invalid values (nan)
            x_sd = np.std(masked_x)  # standard deviation of x
            xlim = (masked_x.min() - x_sd, masked_x.max() + x_sd)  # default xlim
        smoothx = np.linspace(xlim[0], xlim[1], linspace_num)
        predicted_y = func(smoothx, *popt)
        return (smoothx, predicted_y)
    
#     def __residual(self, smoothx, observed_y, predicted_y):
#         return predicted_y - observed_y
#        
    def export_csv(self, outname, delimiter=",", overwrite=False):  
        """
        Export the plot data to a CSV file.

        Parameters:
        - outname (str): The name of the output CSV file.
        - delimiter (str, optional): The delimiter to use in the CSV file. Default is a comma (",").
        - overwrite (bool, optional): Whether to overwrite the existing file. Default is False.

        Description:
        - Adds the ".csv" extension to the output file name if not already present.
        - If `overwrite` is False, appends "(1)", "(2)", etc., to the file name to avoid overwriting.
        - If `overwrite` is True and the file exists, deletes the existing file.
        - Writes the data to the CSV file with the specified delimiter, including header information.

        Raises:
        - OSError: If there is an issue writing to the file.

        Notes:
        - The CSV file includes metadata such as the export date, x-axis label and unit, and y-axis label and unit.
        """
        # add file name extension if not in user input
        if not outname.endswith(".csv"):
            outname += ".csv"
        
        # if not overwrite, add (1), (2), (3), etc. to file name before '.fits'
        if not overwrite:
            outname = _prevent_overwriting(outname)
        else:
            if os.path.exists(outname):
                os.system(f"rm -rf {outname}")
        
        file_content = "# Exported from Astroviz. \n"
        file_content += f"# Date: {self.header['date']} \n"
        file_content += f"# xLabel: {self.xlabel} [{self.xunit}] \n"
        file_content += f"# yLabel: {self.ylabel} [{self.yunit}] \n"
        file_content += f"# x,y \n"
        data = np.vstack(self.x, self.y).T.astype(str)
        data_lines = np.apply_along_axis(lambda arr: ",".join(arr), 1, data)
        data_lines = "\n".join(data_lines)
        file_content += data_lines
        
        with open(outname, "w") as f:
            f.write(file_content)
        print(f"File saved as '{outname}'.")


class Region:
    """
    A class for handling regions.

    This class provides functionality to load and read DS9 files and can be inputted into 
    various methods in Datacube, Spatialmap, and PVdiagram.
    """
    def __init__(self, regionfile=None, shape="ellipse", radius=None,
                 semimajor=1., semiminor=1., pa=0., center=(0., 0.), start=None, 
                 end=None, length=None, width=None, height=None, unit="arcsec",
                 relative=True, quiet=False):
        """
        Initializes a new instance of the Region class.

        Parameters:
            regionfile (str, optional): Path to the DS9 or CRTF region file.
            shape (str): Shape of the region. Supported shapes are 'circle', 'ellipse', 'box', 'line'.
            radius (float, optional): Radius of the circle. Required if shape is 'circle'.
            semimajor (float): Semi-major axis of the ellipse or box. Defaults to 1.0.
            semiminor (float): Semi-minor axis of the ellipse or box. Defaults to 1.0.
            pa (float): Position angle of the shape in degrees. Defaults to 0.
            center (tuple): Central position of the shape. Given as (x, y) coordinates.
            start (tuple, optional): Starting point of the line. Required if shape is 'line'.
            end (tuple, optional): End point of the line. Required if shape is 'line'.
            length (float, optional): Length of the line. Calculated if start and end are provided.
            width (float, optional): Width of the box. Required if shape is 'box'.
            height (float, optional): Height of the box. Required if shape is 'box'.
            unit (str): Unit of the coordinates (e.g., 'arcsec', 'degree'). Defaults to 'arcsec'.
            relative (bool): If True, coordinates are interpreted as relative to the center. Defaults to True.
            quiet (bool): If False, outputs more detailed information during processing. Defaults to False.

        Raises:
            ValueError: If required parameters for the specified shape are not provided.

        Note:
            - Some parameters are optional for shapes. For instance, if 'center', 'length', and 'pa'
              of line are specified, then the 'start' and 'end' are not required.
            - If 'regionfile' is provided, the shape and other parameters are read from the file.
            - The position angle 'pa' is relevant for shapes 'ellipse', 'box', and 'line'.
        """

        self.regionfile = regionfile
        shape = shape.lower()
        unit = unit.lower()
        isJ2000 = (isinstance(center, str) and not center.isnumeric()) \
                    or (len(center)==2 and isinstance(center[0], str) and isinstance(center[1], str)) 
        
        relative = False if isJ2000 else relative
        
        if length is not None or start is not None or end is not None:
            shape = "line"
            
        if width is not None or height is not None:
            shape = "box"
        
        if regionfile is None:
            self.filetype = None
            if shape == "line":
                self.shape = "line"
                if length is None and start is not None and end is not None:
                    start_isJ2000 = (isinstance(start, str) and not start.isnumeric()) \
                        or (len(start)==2 and isinstance(start[0], str) and isinstance(start[1], str))
                    end_isJ2000 = (isinstance(end, str) and not end.isnumeric()) \
                        or (len(end)==2 and isinstance(end[0], str) and isinstance(end[1], str))
                    endpoints_are_J2000 = start_isJ2000 and end_isJ2000
                    if endpoints_are_J2000:
                        if not isinstance(start, str):
                            start_coord = SkyCoord(ra=start[0], dec=start[1], 
                                                   unit=(u.hourangle, u.deg), frame="icrs")
                        else:
                            start_coord = SkyCoord(start, unit=(u.hourangle, u.deg), frame="icrs")
                        start_ra, start_dec = start_coord.ra, start_coord.dec
                        if not isinstance(end, str):
                            end_coord = SkyCoord(ra=end[0], dec=end[1], unit=(u.hourangle, u.deg), frame="icrs")
                        else:
                            end_coord = SkyCoord(end, unit=(u.hourangle, u.deg), frame="icrs")
                        end_ra, end_dec = end_coord.ra, end_coord.dec
                        length = np.sqrt((start_ra-end_ra)**2+(start_dec-end_dec)**2).to(unit).value
                        dx, dy = start_ra-end_ra, end_dec-start_dec
                        pa = -(np.arctan(dx/dy)).to(u.deg).value if dy != 0 else (90. if dx > 0 else -90.)
                        center = SkyCoord(ra=(start_ra+end_ra)/2, dec=(start_dec+end_dec)/2, 
                                          frame="icrs").to_string("hmsdms")
                        start, end = start_coord.to_string("hmsdms"), end_coord.to_string("hmsdms")
                        relative = False
                    else:
                        length = np.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
                        dx, dy = start[0]-end[0], end[1]-start[1]
                        pa = -np.rad2deg(np.arctan(dx/dy)) if dy != 0 else (90. if dx > 0 else -90.)
                        center = (start[0]+end[0])/2, (start[1]+end[1])/2
                elif start is None and end is None and pa is not None \
                     and length is not None and center is not None:
                    dx, dy = length*np.sin(-np.deg2rad(pa)), length*np.cos(-np.deg2rad(pa))
                    if not isJ2000:
                        start = (center[0]-dx/2, center[1]+dy/2)
                        end = (center[0]+dx/2, center[1]-dy/2)
                    else:
                        dx, dy = u.Quantity(dx, unit), u.Quantity(dy, unit)
                        if not isinstance(center, str): 
                            center_coord = SkyCoord(ra=center[0], dec=center[1], 
                                                    unit=(u.hourangle, u.deg), frame='icrs')
                            center = center_coord.ra, center_coord.dec
                        else:
                            center_coord = SkyCoord(center, unit=(u.hourangle, u.deg), frame='icrs')
                            center = center_coord.ra, center_coord.dec
                        start = (center[0]-dx/2, center[1]+dy/2)
                        start = SkyCoord(ra=start[0], dec=start[1], frame="icrs").to_string("hmsdms")
                        end = (center[0]+dx/2, center[1]-dy/2)
                        end = SkyCoord(ra=end[0], dec=end[1], frame="icrs").to_string("hmsdms")
                        center = center_coord.to_string("hmsdms")
                        
                self.start = start
                self.end = end
                self.center = center
                self.pa = pa
                self.length = length
                self.relative = relative
                self.header = {"shape": shape,
                               "start": start,
                               "end": end,
                               "center": center,
                               "length": length,
                               "pa": pa,
                               "unit": unit,
                               "relative": relative,}
                
            if shape == "ellipse":
                self.shape = "ellipse"
                self.center = center
                if semimajor == semiminor or radius is not None:
                    self.radius = self.semimajor = self.semiminor = radius if radius is not None else semimajor
                    self.shape = "circle"
                    self.unit = unit
                    self.pa = 0.
                    self.header = {"shape": shape,
                                   "center": center,
                                   "radius": radius if radius is not None else semimajor,
                                   "unit": unit}
                else:
                    self.semimajor = semimajor
                    self.semiminor = semiminor
                self.pa = pa
                self.unit = unit
                self.header = {"shape": shape,
                               "center": center,
                               "semimajor": semimajor,
                               "semiminor": semiminor,
                               "pa": pa,
                               "unit": unit,
                              }
            if shape == "circle":
                self.shape = "circle"
                self.center = center
                if semiminor is not None:
                    self.radius = semiminor
                if semimajor is not None:
                    self.radius = semimajor
                if radius is not None:
                    self.radius = radius
                self.semimajor = self.semiminor = self.radius
                self.unit = unit
                self.pa = 0.
                self.header = {"shape": shape,
                               "center": center,
                               "radius": semimajor,
                               "unit": unit}
            if shape in ["box", "rectangle", "square"]:
                if shape == "square":
                    if width is not None:
                        self.width = self.height = width
                    elif height is not None:
                        self.width = self.height = height
                    elif length is not None:
                        self.width = self.height = length
                    elif semimajor is not None:
                        self.width = self.height = semimajor*2
                    else:
                        raise ValueError("Must specify the 'width', 'height', or 'length' parameter.")
                else:
                    if width is not None:
                        self.width = width
                    elif semimajor is not None:
                        self.width = semimajor*2
                        
                    if height is not None:
                        self.height = height
                    elif semiminor is not None:
                        self.height = semiminor*2
                    elif length is not None:
                        self.height = length
                    else:
                        raise ValueError("Must specify the 'height'.")
                self.shape = "box"
                self.pa = pa
                self.center = center
                self.unit = unit
                self.header = {"shape": self.shape,
                               "center": self.center,
                               "width": self.width,
                               "height": self.height,
                               "pa": self.pa,
                               "unit": self.unit,
                              }
                    
            self.relative = self.header["relative"] = relative
        else:
            self.__readfile(quiet=quiet)
    
    def __readfile(self, quiet=False):
        if not os.path.exists(self.regionfile):
            if not quiet:
                print(f"Given directory '{fitsfile}' does not exist as a relative directory. " + \
                       "Recursively finding file...")
            maybe_filename = _find_file(self.regionfile)
            if maybe_filename is not None:
                self.regionfile = maybe_filename
                if not quiet:
                    print(f"Found a matching filename: '{self.regionfile}'")
            else:
                raise FileNotFoundError(f"Filename '{self.regionfile}' does not exist.")
        
        with open(self.regionfile) as f:
            filelst = list(f)
            self.__filelst = filelst
        
        if "DS9" in filelst[0]:
            self.filetype = "DS9"
            self.shape = filelst[3].split("(")[0]
            self.__readDS9()
        elif "CRTF" in filelst[0]:
            self.filetype = "CRTF"
            self.shape = filelst[1].split()[0]
            self.__readCRTF()
        
    def saveregion(self, filetype="DS9"):
        raise Exception("To be implemented.")
        
    def __readDS9(self):
        shape = self.shape
        if shape == "line":
            self.unit = unit = "deg"
            coord_tup = self.__filelst[3].replace(self.shape, "")
            coord_tup = coord_tup.split(")")[0] + ")"   # string
            coord_tup = eval(coord_tup)
            self.start = start = coord_tup[:2]
            self.end = end = coord_tup[2:]
            self.length = length = np.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
            dx, dy = start[0]-end[0], end[1]-start[1]
            pa = -np.rad2deg(np.arctan(dx/dy)) if dy != 0 else (90. if dx > 0 else -90.)
            self.center = center = (start[0]+end[0])/2, (start[1]+end[1])/2
            self.relative = relative = False
            self.pa = pa
            self.header = {"shape": shape,
                           "start": start,
                           "end": end,
                           "center": center,
                           "length": length,
                           "pa": pa,
                           "unit": unit,
                           "relative": relative,}
        elif shape == "ellipse":
            self.unit = unit = "deg"
            coord_tup = self.__filelst[3].replace(self.shape+"(", "")
            coord_tup = coord_tup.split(")")[0]  # string
            elements = coord_tup.split(", ")
            x, y, semimajor, semiminor, pa = elements
            semimajor = semimajor.replace('"', "arcsec").replace("'", "arcmin").replace("∘", "deg")
            semiminor = semiminor.replace('"', "arcsec").replace("'", "arcmin").replace("∘", "deg")
            self.center = center = float(x), float(y)
            self.semimajor = semimajor = float(u.Quantity(semimajor).to_value(u.deg))
            self.semiminor = semiminor = float(u.Quantity(semiminor).to_value(u.deg))
            self.pa = pa = float(pa)-90.
            self.relative = relative = False
            self.header = {"shape": shape,
                           "center": center,
                           "semimajor": semimajor,
                           "semiminor": semiminor,
                           "pa": pa,
                           "unit": unit,
                           "relative": relative
                           }
        elif shape == "box":
            self.unit = unit = "deg"
            coord_tup = self.__filelst[3].replace(self.shape+"(", "")
            coord_tup = coord_tup.split(")")[0]  # string
            elements = coord_tup.split(", ")
            center_ra, center_dec, width_str, height_str, pa_str = elements
            self.center = center = float(center_ra), float(center_dec)
            width_str = width_str.replace('"', "arcsec").replace("'", "arcmin").replace("∘", "deg")
            height_str = height_str.replace('"', "arcsec").replace("'", "arcmin").replace("∘", "deg")
            self.width = width = float(u.Quantity(width_str).to_value(u.deg))
            self.height = height = float(u.Quantity(height_str).to_value(u.deg))
            self.pa = pa = float(pa_str)
            
            self.relative = relative = False
            self.header = {"shape": shape,
                           "center": center,
                           "width": width,
                           "height": height,
                           "pa": pa,
                           "unit": unit,
                           "relative": relative
                           }
        else:
            raise Exception("Region not supported in this current version.")

    def __readCRTF(self):
        shape = self.shape
        unit = self.unit = "deg"
        if shape == "line":
            coord_lst = self.__filelst[1].split()[1:5]
            
            # start coordinate
            start_ra = coord_lst[0][2:-1]
            start_dec = coord_lst[1][:-2]
            start_dec = start_dec.split(".")
            start_dec = start_dec[0] + ":" + start_dec[1] + ":" + start_dec[2] + "." + start_dec[3]
            start_str = start_ra + " " + start_dec
            start_coord = SkyCoord(start_str, unit=(u.hourangle, u.deg), frame="icrs")
            self.start = start = start_coord.to_string("hmsdms")
            
            # end coordinate
            end_ra = coord_lst[2][1:-1]
            end_dec = coord_lst[3][:-2]
            end_dec = end_dec.split(".")
            end_dec = end_dec[0] + ":" + end_dec[1] + ":" + end_dec[2] + "." + end_dec[3]
            end_str = end_ra + " " + end_dec
            end_coord = SkyCoord(end_str, unit=(u.hourangle, u.deg), frame="icrs")
            self.end = end = end_coord.to_string("hmsdms")
            
            # central coordinate
            center_ra = (end_coord.ra + start_coord.ra)/2
            center_dec = (end_coord.dec + start_coord.dec)/2
            self.center = center = SkyCoord(ra=center_ra, dec=center_dec, frame="icrs").to_string("hmsdms")
            
            # position angle 
            dx = start_coord.ra - end_coord.ra
            dy = end_coord.dec - start_coord.dec
            self.length = length = np.sqrt(dx**2+dy**2).to_value(self.unit)
            pa = -np.arctan(dx/dy).to_value(u.deg) if dy != 0 else (90. if dx>0 else -90.)
            self.pa = pa
            
            self.relative = relative = False
            self.header = {"shape": shape,
                           "start": start,
                           "end": end,
                           "center": center,
                           "length": length,
                           "pa": pa,
                           "unit": unit,
                           "relative": relative}
        elif shape == "ellipse":
            raise Exception("Not implemented yet.")
        elif shape == "box":
            raise Exception("Not implemented yet.")
        else:
            raise Exception("Not implemented yet.")


class ImageMatrix:
    """
    A class for handling and plotting multiple images with customizable annotations.
    """
    def __init__(self, figsize=(11.69, 16.57), axes_padding='auto', 
                 dpi=600, labelsize=10., fontsize=12., axeslw=1., 
                 tickwidth=None, ticksize=3., tick_direction="in", 
                 cbarloc="right", **kwargs):
        """
        Initialize the ImageMatrix instance with default parameters.

        Parameters:
        - figsize (tuple of float or int, optional): The size of the figure in inches. Default is (11.69, 8.27).
        - axes_padding (tuple of float, optional): The padding between axes. Default is (0.2, 0.2).
        - dpi (float or int, optional): The dots per inch (DPI) setting for the figure. Default is 600.
        - **kwargs: Additional keyword arguments to set or update the plotting parameters.

        Description:
        - This method initializes an ImageMatrix instance with default plotting parameters.
        - It sets up attributes such as `images`, `figsize`, `shape`, `size`, `axes_padding`, and `dpi`.
        - Default plotting parameters are defined in `self.default_params`.
        - Additional keyword arguments can be passed to customize plotting parameters, which are set using the `set_params` method.
        - Initializes internal dictionaries to store special parameters, shapes, lines, and text annotations.
        - Initializes `fig` and `axes` attributes for the Matplotlib figure and axes.

        Attributes:
        - images (list): A list to store images.
        - figsize (tuple): The size of the figure in inches.
        - shape (tuple): The shape of the image matrix (initially (0, 0)).
        - size (int): The size of the image matrix (initially 0).
        - axes_padding (tuple): The padding between axes.
        - dpi (float or int): The dots per inch (DPI) setting for the figure.
        - default_params (dict): A dictionary of default plotting parameters.
        - other_kwargs (dict): A dictionary for additional keyword arguments for plotting parameters.
        - specific_kwargs (dict): A dictionary for specific keyword arguments for individual images.
        - fig (matplotlib.figure.Figure or None): The Matplotlib figure object.
        - axes (list or None): The list of Matplotlib axes objects.
        """
        # set default parameters
        self.images: list = []  # empty list
        self.figsize: Tuple[Union[float, int]] = figsize
        self.shape: Tuple[int] = (0, 0)  # empty tuple
        self.size: int = 0   # empty shape
        
        # parse 'axes padding' parameter 
        if isinstance(axes_padding, (int, np.integer, float, np.floating)):
            self.axes_padding = (axes_padding, axes_padding)
        elif hasattr(axes_padding, "__iter__") and len(axes_padding) == 2:
            self.axes_padding = axes_padding
        elif axes_padding == "auto":
            self.axes_padding = axes_padding
        else:
            raise ValueError("'axes_padding' must be a number, a tuple of two numbers, or 'auto'")
            
        # parse dpi
        self.dpi: Union[float, int] = dpi
        
        # default parameters
        if tickwidth is None:
            tickwidth = axeslw  # set it to be the same as axeslw if not specified
            
        # parse 'cbarloc'
        if not cbarloc.islower():
            cbarloc = cbarloc.lower()
        if cbarloc not in ("right", "top"):
            raise ValueError("'cbarloc' must be either 'right' or 'top'.")
            
        self.default_params: dict = {'axes.labelsize': labelsize,
                                     'axes.titlesize': fontsize,
                                     'font.size': fontsize,
                                     'xtick.labelsize': labelsize,
                                     'ytick.labelsize': labelsize,
                                     'xtick.top': True,  # draw ticks on the top side
                                     'xtick.major.top': True,
                                     'figure.figsize': figsize,
                                     'font.family': _fontfamily,
                                     'mathtext.fontset': _mathtext_fontset,
                                     'mathtext.tt': _mathtext_tt,
                                     'axes.linewidth': axeslw, 
                                     'xtick.major.width': tickwidth,
                                     'xtick.major.size': ticksize,
                                     'xtick.direction': tick_direction,
                                     'ytick.major.width': tickwidth,
                                     'ytick.major.size': ticksize,
                                     'ytick.direction': tick_direction,
                                     }
            
        self.other_kwargs: dict = {"labelsize": labelsize, 
                                   "fontsize": fontsize, 
                                   "cbarloc": cbarloc}
        self.specific_kwargs: dict = {}
        self.set_params(new_params=kwargs)
        
        # initialize private attributes:
        self.__label_abc: bool = False
        self.__lines: dict = {}
        self.__shapes: dict = {}
        self.__texts: dict = {}
        self.__show_mag: list = []
        self.__set_positions: dict = {}
        self.__cbarloc = cbarloc
                    
        self.fig = None
        self.axes = None
        
    def __ravel_index(self, multi_dim_index):
        """
        Converts a multi-dimensional index to a single-dimensional index or 
        validates a single-dimensional index.

        Parameters:
            - multi_dim_index (int or iterable): The index to be converted or validated. 
                                               This can either be a single integer 
                                               representing a flat index or an iterable 
                                               (like a tuple or list) of length 2 
                                               representing a 2D index.

        Returns:
            - (int) The single-dimensional index corresponding to the provided multi-dimensional index.

        Raises:
            - ValueError
                - If `multi_dim_index` is an int and is out of bounds for the array size.
                - If `multi_dim_index` is an iterable and its length is not 2.
                - If `multi_dim_index` is an iterable and contains indices out of bounds 
                  for the array shape.
                - If `multi_dim_index` is neither an int nor an iterable of length 2.
        """
        if isinstance(multi_dim_index, (int, np.integer)):
            # check if it can be parsed as a one-dim index
            if multi_dim_index > self.size - 1:
                raise ValueError(f"Index {multi_dim_index} is out of bounds for size {self.size}")
            # return itself if it is an integer
            return multi_dim_index
        elif hasattr(multi_dim_index, "__iter__") and hasattr(multi_dim_index, "__getitem__"):
            # check if it can be parsed as a two-dim index
            if len(multi_dim_index) != 2:
                raise ValueError("'multi_dim_index' must be an iterable of length 2")
            row, col = multi_dim_index
            if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
                raise ValueError(f"Index {multi_dim_index} is out of bounds for shape {self.shape}")
            return np.ravel_multi_index(multi_dim_index, self.shape)
        else:
            raise ValueError("multi_dim_index must be an int or an iterable of length 2")
            
    def set_params(self, new_params=None, **kwargs):
        """
        Set or update the plotting parameters for the image matrix.

        Parameters:
        - new_params (dict, optional): A dictionary of new parameters to set or update. Default is None.
        - **kwargs: Additional keyword arguments to set or update the plotting parameters.

        Returns:
        - self (ImageMatrix): The updated ImageMatrix instance with the new parameters.

        Description:
        - This method sets or updates the default plotting parameters for the image matrix.
        - The parameters can be provided either as a dictionary (`new_params`) or as keyword arguments.
        - If a parameter key is found in `rcParams`, it updates the corresponding entry in `self.default_params`.
        - Otherwise, it updates the entry in `self.other_kwargs`.
        - This allows for flexible customization of plotting parameters for the image matrix.
        """
        # set new dictionary
        if new_params is None:
            new_params = kwargs
        
        # iterate over 'new_params' dict to replace default values
        for key, value in new_params.items():
            if key in rcParams:
                self.default_params[key] = value
            else:
                self.other_kwargs[key] = value

        return self
    
    def copy(self):
        """
        Create a deep copy of the ImageMatrix instance.

        Returns:
        - ImageMatrix: A deep copy of the current ImageMatrix instance.

        Description:
        - This method creates and returns a deep copy of the current ImageMatrix instance.
        - A deep copy ensures that all nested objects and attributes within the ImageMatrix are also copied, 
          preventing any changes to the original instance from affecting the copy.

        Example:
        - To create a copy of an existing ImageMatrix instance:
          >>> matrix_copy = original_matrix.copy()
        """
        return copy.deepcopy(self)
        
    def reshape(self, *shape, inplace=True):
        """
        Reshape the image matrix into a different 2D shape.

        Parameters:
        - shape (tuple of int): The new shape for the image matrix. It should be a tuple containing two integers.
        - inplace (bool, optional): Whether to modify the image matrix in place. Default is True.

        Returns:
        - matrix (ImageMatrix): The reshaped image matrix.

        Raises:
        - ValueError: If the length of `shape` is greater than 2 or if any element in `shape` is not an integer.

        Description:
        - This method changes the shape of the image matrix to the specified `shape`.
        - If `inplace` is False, a copy of the image matrix is created and modified.
        - If the length of `shape` is 1, it is converted to a 2D shape with one row.
        - The shape and size of the image matrix are updated to match the new shape.
        """
        # old shape
        old_shape = self.shape
        
        # create copy if inplace is set to True
        matrix = self if inplace else self.copy()
        
        # parse args
        if len(shape) == 1:
            shape = shape[0]
        
        # check if shape satisfies length requirement
        if len(shape) == 1:
            shape = (1, shape[0])
        elif len(shape) > 2:
            raise ValueError("Length of 'shape' is greater than 2.")
        
        # change items in shape to int:
        if any(not isinstance(item, (int, np.integer)) for item in shape):
            shape = tuple(int(item) for item in shape)

        # raise value error
        new_size = np.multiply(*shape)
        length_of_images = len(self.images)
        if new_size < length_of_images:
            warnings.warn(f"Not all {length_of_images} images will be plotted with shape {shape}.")

        # modify shape and size
        matrix.shape = shape
        matrix.size = new_size
                
        return matrix
        
    def add_image(self, image=None, **kwargs) -> None:
        """
        Add an image to the ImageMatrix.

        Parameters:
        - image (Spatialmap, PVdiagram, Plot2D, or None): The image to be added to the matrix.
        - plot (bool, optional): Whether to plot the image after adding it. Default is False.
        - **kwargs: Additional keyword arguments to be passed to the plot method.

        Returns:
        - self (ImageMatrix): The updated ImageMatrix instance.

        Raises:
        - ValueError: If the image added is not of an acceptable type (Spatialmap, PVdiagram, Plot2D, or None).

        Description:
        - Checks if the provided image is of an acceptable type.
        - Adds the image to the image matrix.
        - Reshapes the matrix if the number of images exceeds the current size.
        - Stores specific keyword arguments for the added image.
        - Optionally plots the image if the `plot` parameter is set to True.
        """
        # check type of input
        acceptable_types = (Spatialmap, PVdiagram, Plot2D, type(None))
        if not isinstance(image, acceptable_types):
            raise ValueError("The image added is not of an acceptable type.")
        
        # add image
        self.images.append(image)
        
        
        if self.size == 0:
            self.reshape((1, 1), inplace=True)
        elif self.size == 1:
            self.reshape((1, 2), inplace=True)
        elif self.size == 2:
            self.reshape((1, 3), inplace=True)
        elif self.size == 3:
            self.reshape((2, 2), inplace=True)
        elif len(self.images) > self.size:
            new_shape = (self.shape[0]+1, self.shape[1])  # add a new row if it exceeds current size
            self.reshape(new_shape, inplace=True)
            
        # set specific keyword arguments
        self.specific_kwargs[len(self.images)-1] = kwargs

    def __create_panel_labels(self) -> List[str]:
        """
        Generates a list of panel labels for the instance.

        This method creates panel labels in a sequential pattern, starting from 'a', 'b', 'c', ..., 'z',
        and then 'aa', 'ab', 'ac', ..., 'az', 'ba', 'bb', ..., 'zz', and so on. The length of the list
        of labels is determined by the number of items in `self.images`.

        Returns:
            List[str]: A list of panel labels.
        """
        # create generator object
        def labels() -> iter:
            i = 1
            while True:
                for label in itertools.product(string.ascii_lowercase, repeat=i):
                    yield ''.join(label)
                i += 1

        # allocate memory for labels by creating a list
        generator = labels()
        return list(next(generator) for _ in range(len(self.images)))
                
    def plot(self, plot=True):
        """
        Plot the images in the image matrix along with any added annotations.

        Parameters:
        - plot (bool, optional): Whether to display the plot. Default is True.

        Returns:
        - fig (matplotlib.figure.Figure): The figure object containing the plots.
        - axes (list of matplotlib.axes.Axes): The axes objects of the plots.

        Description:
        - This method sets the default plotting parameters and creates a figure with subplots
          according to the shape of the image matrix.
        - Each image in the image matrix is plotted in its respective subplot.
        - If an image is 'None' or the subplot index exceeds the number of images, the subplot is turned off.
        - Additional keyword arguments for plotting are passed to the `imview` method of each image.
        - If `self.__label_abc` is True, each subplot is labeled with a sequential letter (a, b, c, ...).
        - The figure's padding and dpi are adjusted according to the instance's attributes.
        - After plotting the images, any lines, shapes, or texts stored in the instance are drawn on the figure.
        - The figure is displayed if `plot` is True, and the figure and axes objects are returned.

        Example:
        - To plot the image matrix and display it:
          >>> fig, axes = image_matrix.plot()

        - To plot the image matrix without displaying it (e.g., for saving to a file):
          >>> fig, axes = image_matrix.plot(plot=False)
        """
        rcParams.update(self.default_params)  # set default parameters
        if self.axes_padding == "auto":
            fig, axes = plt.subplots(*self.shape, figsize=self.figsize, 
                                     constrained_layout=True)  # set fig, axes, etc.
        else:
            fig, axes = plt.subplots(*self.shape, figsize=self.figsize)  # set fig, axes, etc.
            
        if isinstance(axes, mpl.axes._axes.Axes):
            axes = [axes]
        else: 
            axes = axes.flatten()  # flatten axes
        other_kwargs = self.other_kwargs  # other keyword arguments
        
        label_idx: int = 0
        all_labels: List[str] = self.__create_panel_labels()
        for i, ax in enumerate(axes):
            # plot blank images if image is 'None' or the ax is out of range.
            if i >= len(self.images) or (image := self.images[i]) is None:  
                ax.spines["bottom"].set_color('none')
                ax.spines["top"].set_color('none')
                ax.spines["left"].set_color('none')
                ax.spines["right"].set_color('none')
                ax.axis('off')
            else:
                specific_kwargs = self.specific_kwargs[i]
                this_kwarg = copy.deepcopy(other_kwargs)
                
                # check if all parameters are valid
                valid_args = tuple(inspect.signature(image.imview).parameters.keys())
                this_kwarg = dict((key, value) for key, value in this_kwarg.items() if key in valid_args)
                this_kwarg.update(specific_kwargs)
                
                # add figure labels if necessary
                if "title" in this_kwarg:
                    title = this_kwarg["title"]
                    this_kwarg.pop("title")
                else:
                    title = ""
                    
                # add panel labels if necessary
                if self.__label_abc:
                    title = f"({all_labels[label_idx]}) " + title
                    label_idx += 1
                    
                # plot using 'imview' method
                if hasattr(image, "imview"):
                    ax = image.imview(ax=ax, plot=False, title=title, **this_kwarg)
                else:
                    ax = image.plot(ax=ax, plot=False, title=title, **this_kwarg)
                  
        # adjust padding
        if self.axes_padding != "auto":
            wspace, hspace = self.axes_padding
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
        
        # set dpi
        fig.set_dpi(self.dpi)
        
        # set instances
        self.fig = fig
        self.axes = axes
        
        # draw other annotations
        fig, axes = self.__draw_lines(fig, axes)
        fig, axes = self.__draw_shapes(fig, axes)
        fig, axes = self.__add_texts(fig, axes)
        fig, axes = self.__plot_mag(fig, axes)
        
        # set horizontal positions
        fig, axes = self.__set_horizontal_positions(fig, axes)
        
        if plot:
            plt.show()
        
        return fig, axes
        
    def label_panels(self) -> None:
        """
        Enable labeling of panels with sequential letters (a, b, c, ...).

        Description:
        - This method sets an internal flag to enable the labeling of panels with sequential letters.
        - When this flag is set to True, each panel in the image matrix will be 
          labeled with a sequential letter (a, b, c, ...).
        """
        self.__label_abc = True
        
    def show_magnification(self, zoomed_out_idx, zoomed_in_idx, 
                           linestyle="dashed", linecolor="skyblue", 
                           linewidth=1.0, edgecolor=None, 
                           facecolor="none", **kwargs) -> None:
        """
        Highlights a zoomed-in region within a zoomed-out image and draws connecting lines.

        This method adds a rectangle to the zoomed-out image, highlighting the region that is 
        zoomed in on another subplot. Additionally, it draws lines connecting the corners of 
        the zoomed-in region to the corresponding region in the zoomed-out image.

        Parameters:
            zoomed_out_idx (int or tuple): The index or coordinates of the zoomed-out image 
                                           in the image matrix.
            zoomed_in_idx (int or tuple): The index or coordinates of the zoomed-in image in 
                                          the image matrix.
            linestyle (str, optional): The style of the connecting lines. Default is "dashed".
            linecolor (str, optional): The color of the connecting lines. Default is "skyblue".
            linewidth (float, optional): The width of the connecting lines. Default is 1.0.
            edgecolor (str, optional): The edge color of the rectangle. If None, defaults to 
                                       the value of `linecolor`. Default is None.
            facecolor (str, optional): The face color of the rectangle. Default is "none".
            **kwargs: Additional keyword arguments to be passed to the `Rectangle` patch.

        Returns:
            None

        Description:
        - This method transforms the provided indices into raveled indices and stores the 
          parameters required for highlighting and connecting the zoomed regions.
        - The actual drawing of the rectangle and lines is handled in the `__plot_mag` method, 
          which is called during the plotting process.
        """
        # ravel indicies:
        zoomed_out_idx = self.__ravel_index(zoomed_out_idx)
        zoomed_in_idx = self.__ravel_index(zoomed_in_idx)
        
        if zoomed_out_idx == zoomed_in_idx:
            raise ValueError("Same indicies for zoomed in/out images were provided.")
        
        # default value of edge color:
        if edgecolor is None:
            edgecolor = linecolor
            
        # set keyword arguments as dictionary:
        set_kwargs = {"linestyle": linestyle, "linecolor": linecolor,
                      "linewidth": linewidth, "facecolor": facecolor,
                      "edgecolor": edgecolor}
        
        # append to private variable
        self.__show_mag.append((zoomed_out_idx, zoomed_in_idx, 
                                set_kwargs, kwargs))
        
    def __plot_mag(self, fig, axes):
        if not self.__show_mag:
            return fig, axes
        
        for (out_idx, in_idx, set_kwargs, kwargs) in self.__show_mag:
            # skip loop if one of the images is a blank one
            if self.images[out_idx] is None or self.images[in_idx] is None:
                continue
            
            # get values
            out_ax = axes[out_idx]  # zoomed-out axis
            out_xlim = out_ax.get_xlim()
            out_xrange = abs(out_xlim[1] - out_xlim[0])
            out_ylim = out_ax.get_ylim()
            out_yrange = abs(out_ylim[1] - out_ylim[0])
            
            in_ax = axes[in_idx]  # zoomed-in axis
            in_xlim = in_ax.get_xlim()
            in_xrange = abs(in_xlim[1] - in_xlim[0])
            in_ylim = in_ax.get_ylim()
            in_yrange = abs(in_ylim[1] - in_ylim[0])
            
            # swap out and in indicies if they are opposite
            if in_xrange > out_xrange and in_yrange > out_yrange:
                out_idx, in_idx = in_idx, out_idx  # python swap!
                out_ax = axes[out_idx]  # zoomed-out axis
                out_xlim = out_ax.get_xlim()
                out_xrange = abs(out_xlim[1] - out_xlim[0])
                out_ylim = out_ax.get_ylim()
                out_yrange = abs(out_ylim[1] - out_ylim[0])

                in_ax = axes[in_idx]  # zoomed-in axis
                in_xlim = in_ax.get_xlim()
                in_xrange = abs(in_xlim[1] - in_xlim[0])
                in_ylim = in_ax.get_ylim()
                in_yrange = abs(in_ylim[1] - in_ylim[0])
            
            # add rectangle in zoomed-out image:
            br_xy = (min(in_xlim), min(in_ylim))  # bottom right xy coord
            rect = patches.Rectangle(xy=br_xy, 
                                     width=in_xrange, 
                                     height=in_yrange, 
                                     angle=0,
                                     linewidth=set_kwargs["linewidth"], 
                                     linestyle=set_kwargs["linestyle"], 
                                     edgecolor=set_kwargs["edgecolor"], 
                                     facecolor=set_kwargs["facecolor"], 
                                     **kwargs)
            out_ax.add_patch(rect)
            
            fig.canvas.draw()
            transFigure = fig.transFigure.inverted()
            
            # Get the locations of the axes in the grid
            out_loc = np.unravel_index(out_idx, self.shape)
            in_loc = np.unravel_index(in_idx, self.shape)
            
            # Calculate the coordinates for the lines based on the location of the axes
            if out_loc[0] == in_loc[0]:  # same row -> plot side to side
                if out_loc[1] > in_loc[1]:  # right is zoomed out, left is zoomed in
                    in_coords = [(in_xlim[1], in_ylim[0]), (in_xlim[1], in_ylim[1])]
                    out_coords = [(in_xlim[0], in_ylim[0]), (in_xlim[0], in_ylim[1])]
                else:  # left is zoomed out, right is zoomed in
                    in_coords = [(in_xlim[0], in_ylim[0]), (in_xlim[0], in_ylim[1])]
                    out_coords = [(in_xlim[1], in_ylim[0]), (in_xlim[1], in_ylim[1])]
            else:  # different rows -> plot one above the other
                if out_loc[0] > in_loc[0]:  # top is zoomed out, bottom is zoomed in
                    in_coords = [(in_xlim[0], in_ylim[0]), (in_xlim[1], in_ylim[0])]
                    out_coords = [(in_xlim[0], in_ylim[1]), (in_xlim[1], in_ylim[1])]
                else:  # bottom is zoomed out, top is zoomed in
                    in_coords = [(in_xlim[0], in_ylim[1]), (in_xlim[1], in_ylim[1])]
                    out_coords = [(in_xlim[0], in_ylim[0]), (in_xlim[1], in_ylim[0])]
            
            # plot
            for in_coord, out_coord in zip(in_coords, out_coords):
                in_fig_coord = transFigure.transform(in_ax.transData.transform(in_coord))
                out_fig_coord = transFigure.transform(out_ax.transData.transform(out_coord))
                
                line = plt.Line2D((in_fig_coord[0], out_fig_coord[0]), 
                                  (in_fig_coord[1], out_fig_coord[1]),
                                  transform=fig.transFigure, 
                                  color=set_kwargs["linecolor"],
                                  linestyle=set_kwargs["linestyle"], 
                                  linewidth=set_kwargs["linewidth"])
                fig.lines.append(line)

        return fig, axes
            
    def add_text(self, img_idx, x, y, s, **kwargs) -> None:
        """
        Add text to the specified image in the image matrix.

        Parameters:
        - img_idx (int): The index of the image to which the text will be added.
        - x (float): The x-coordinate for the text position.
        - y (float): The y-coordinate for the text position.
        - s (str): The text string to be displayed.
        - **kwargs: Additional keyword arguments to be passed to the `text` method of the axes.

        Returns:
        - None

        Raises:
        - ValueError: If the specified image index is invalid.

        Description:
        - This method creates a text object with specified properties and adds it to the specified image in the image matrix.
        - The text properties such as position, string, and other text attributes can be customized through `kwargs`.
        - The text specifications are stored in `self.__texts` and will be drawn when the image matrix is plotted.
        """
        img_idx = self.__ravel_index(img_idx)
        
        kwargs.update({"s": s, "x": x, "y": y})
        
        if img_idx in self.__texts:
            self.__texts[img_idx].append(kwargs)
        else:
            self.__texts[img_idx] = [kwargs]
            
    def __add_texts(self, fig, axes) -> None:
        """
        Draw stored text objects on the specified axes in the image matrix.

        Parameters:
        - fig (matplotlib.figure.Figure): The figure object containing the plots.
        - axes (list of matplotlib.axes.Axes): The axes objects of the plots.

        Returns:
        - fig (matplotlib.figure.Figure): The updated figure object with the text objects drawn.
        - axes (list of matplotlib.axes.Axes): The updated axes objects.

        Description:
        - This method draws text objects stored in `self.__texts` on the specified figure and axes.
        - Each text object is associated with an image index and is added to the corresponding subplot.
        - If there are no text objects to be drawn (`self.__texts` is empty), the method returns the figure and axes unmodified.

        Text Drawing Process:
        - The method first checks if there are any text objects to be drawn. If not, it returns the figure and axes unmodified.
        - For each text object defined in `self.__texts`, the method retrieves the image index and the text properties.
        - The text is added to the corresponding axis using the `text` method with the specified properties.
        - The method ensures that the text objects are drawn when the figure is displayed by updating the axes with the text objects.

        Raises:
        - None
        """
        if not self.__texts:
            return fig, axes
        
        for img_idx, kwargs_lst in self.__texts.items():
            for kwargs in kwargs_lst:
                axes[img_idx].text(**kwargs)
                
        return fig, axes
    
    def add_arrow(self, img_idx, xy1, xy2=None, dx=None, dy=None, 
                  width=1., color="tomato", double_headed=False, **kwargs):
        """
        Add an arrow to a specific image in the image matrix.

        Parameters:
        - img_idx (int or tuple): The index or coordinates of the image to which the arrow will be added. 
          If a tuple is provided, it should contain two elements representing coordinates.
        - xy1 (tuple): The starting point of the arrow (x, y).
        - xy2 (tuple, optional): The ending point of the arrow (x, y). If not provided, dx and dy must be specified.
        - dx (float, optional): The length of the arrow along the x-axis. Required if xy2 is not provided.
        - dy (float, optional): The length of the arrow along the y-axis. Required if xy2 is not provided.
        - width (float, optional): The width of the arrow. Default is 1.0.
        - color (str, optional): The color of the arrow. Default is "tomato".
        - **kwargs: Additional keyword arguments to pass to `patches.Arrow`.

        Raises:
        - ValueError: If neither xy2 nor both dx and dy are provided.

        Description:
        - Converts the img_idx to a single-dimensional index using the `__ravel_index` method.
        - Validates that either xy2 or both dx and dy are provided; raises a ValueError if not.
        - Computes dx and dy from xy1 and xy2 if xy2 is provided.
        - Creates an Arrow patch with the specified properties and additional keyword arguments.
        - Adds the Arrow patch to the list of shapes for the specified image index.
        """
        # ravel index and check if the index is valid
        img_idx = self.__ravel_index(img_idx)
        
        # check if all info is provided:
        if xy2 is None and (dx is None or dy is None):
            raise ValueError("Either 'xy2' or both 'dx' and 'dy' must be provided.")
        
        # get x1 and y1
        x1, y1 = xy1
        
        # get dx and dy if xy2 is provided (this overwrites the provided 'dx' and 'dy')
        if xy2 is not None:
            x2, y2 = xy2
            dx = x2 - x1
            dy = y2 - y1

        if double_headed:
            # forward arrow
            self.add_arrow(img_idx=img_idx, xy1=(x1+dx/2, y1+dy/2), dx=dx/2, dy=dy/2,
                           width=width, color=color, double_headed=False, 
                           **kwargs)

            # backward arrow
            self.add_arrow(img_idx=img_idx, xy1=(x1+dx/2, y1+dy/2), dx=-dx/2, dy=-dy/2,
                           width=width, color=color, double_headed=False, 
                           **kwargs)
        else:
            # create new patch object:
            patch = patches.Arrow(x=x1, y=y1, dx=dx, dy=dy, color=color, 
                                  width=width, **kwargs)
            
            # save the patch to the private instance:
            if img_idx in self.__shapes:
                self.__shapes[img_idx].append(patch)
            else:
                self.__shapes[img_idx] = [patch]

    def add_rectangle(self, img_idx, xy=(0, 0), width=1, height=None, angle=0, 
                      linewidth=1., edgecolor="skyblue", facecolor='none', 
                      linestyle="dashed", center=True, **kwargs) -> None:
        """
        Add a rectangle to the specified image in the image matrix.

        Parameters:
        - img_idx (int): The index of the image to which the rectangle will be added.
        - xy (tuple, optional): The (x, y) bottom left corner coordinates of the rectangle. Default is (0, 0).
        - width (float, optional): The width of the rectangle. Default is 1.
        - height (float, optional): The height of the rectangle. If None, it is set equal to the width. Default is None.
        - angle (float, optional): The rotation angle of the rectangle in degrees. Default is 0.
        - linewidth (float, optional): The width of the rectangle edge. Default is 1.
        - edgecolor (str, optional): The edge color of the rectangle. Default is "skyblue".
        - facecolor (str, optional): The fill color of the rectangle. Default is 'none'.
        - linestyle (str, optional): The line style of the rectangle edge. Default is "dashed".
        - **kwargs: Additional keyword arguments to be passed to the `Rectangle` constructor.

        Returns:
        - None

        Raises:
        - ValueError: If the specified image index is invalid.

        Description:
        - This method creates a rectangle patch and adds it to the specified image in the image matrix.
        - The rectangle's properties such as width, height, angle, colors, and line styles can be customized.
        - The rectangle is stored in `self.__shapes` and will be drawn when the image matrix is plotted.
        """
        # ravel index and check if the index is valid
        img_idx = self.__ravel_index(img_idx)
        
        # set height to be same as width if height is not specified:
        if height is None:
            height = width
        
        # calculate bottom right if xy specified should be the center:
        if center:
            xy = (xy[0]-width/2, xy[1]-height/2)
         
        # create new patch object:
        patch = patches.Rectangle(xy, width=width, height=height, angle=angle,
                                  linewidth=linewidth, linestyle=linestyle, 
                                  edgecolor=edgecolor, facecolor=facecolor, 
                                  **kwargs)
        
        # save the patch to the private instance:
        if img_idx in self.__shapes:
            self.__shapes[img_idx].append(patch)
        else:
            self.__shapes[img_idx] = [patch]
        
    def add_ellipse(self, img_idx, xy=(0, 0), width=1, height=None, angle=0, 
                    linewidth=1, facecolor='none', edgecolor="skyblue", 
                    linestyle="dashed", center=True, **kwargs) -> None:
        """
        Add an ellipse to the specified image in the image matrix.

        Parameters:
        - img_idx (int): The index of the image to which the ellipse will be added.
        - xy (tuple, optional): The (x, y) center coordinates of the ellipse. Default is (0, 0).
        - width (float, optional): The width of the ellipse. Default is 1.
        - height (float, optional): The height of the ellipse. If None, it is set equal to the width. Default is None.
        - angle (float, optional): The rotation angle of the ellipse in degrees. Default is 0.
        - linewidth (float, optional): The width of the ellipse edge. Default is 1.
        - facecolor (str, optional): The fill color of the ellipse. Default is 'none'.
        - edgecolor (str, optional): The edge color of the ellipse. Default is "skyblue".
        - linestyle (str, optional): The line style of the ellipse edge. Default is "dashed".
        - **kwargs: Additional keyword arguments to be passed to the `Ellipse` constructor.

        Returns:
        - None

        Raises:
        - ValueError: If the specified coordinates or image index are invalid.

        Description:
        - This method creates an ellipse patch and adds it to the specified image in the image matrix.
        - The ellipse's properties such as width, height, angle, colors, and line styles can be customized.
        - The ellipse is stored in `self.__shapes` and will be drawn when the image matrix is plotted.
        """
        # check whether coordinate specified is correct
        img_idx = self.__ravel_index(img_idx)
            
        if height is None:
            height = width
        
        if center:
            xy = (xy[0]-width/2, xy[1]-height/2)
        
        patch = patches.Ellipse(xy=xy, width=width, height=height, facecolor=facecolor,
                                angle=angle, linestyle=linestyle, **kwargs)
        if img_idx in self.__shapes:
            self.__shapes[img_idx].append(patch)
        else:
            self.__shapes[img_idx] = [patch]
        
    def add_patch(self, patch, img_idx=None) -> None:
        """
        Add a patch (shape) to the specified image(s) in the image matrix.

        Parameters:
        - patch (matplotlib.patches.Patch): The patch (shape) to be added to the image(s).
        - img_idx (int or iterable of int, optional): The index or indices of the image(s) 
          to which the patch will be added. If None, the patch will be added to all non-None 
          images in the image matrix. Default is None.

        Returns:
        - None

        Description:
        - This method adds a patch (shape) to the specified image(s) in the image matrix.
        - If `img_idx` is None, the patch will be added to all non-None images.
        - If `img_idx` is an integer, the patch will be added to the corresponding image.
        - If `img_idx` is an iterable, the patch will be added to each specified image.
        - The patches are stored in `self.__shapes`, which is used by the `__draw_shapes` 
          method to draw the patches on the figure.
        """
        if img_idx is None:
            img_idx = [idx for idx in range(self.size) if self.images[idx] is not None]
        
        if hasattr(img_idx, "__iter__"):
            for idx in img_idx:
                if img_idx in self.__shapes:
                    self.__shapes[img_idx].append(patch)
                else:
                    self.__shapes[img_idx] = [patch]
        elif isinstance(img_idx, int):
            if img_idx in self.__shapes:
                self.__shapes[img_idx].append(patch)
            else:
                self.__shapes[img_idx] = [patch]
        
    def __draw_shapes(self, fig, axes):
        """
        Draw stored shapes on the specified axes in the image matrix.

        Parameters:
        - fig (matplotlib.figure.Figure): The figure object containing the plots.
        - axes (list of matplotlib.axes.Axes): The axes objects of the plots.

        Returns:
        - fig (matplotlib.figure.Figure): The updated figure object with the shapes drawn.
        - axes (list of matplotlib.axes.Axes): The updated axes objects.

        Description:
        - This method draws shapes stored in `self.__shapes` on the specified figure and axes.
        - Each shape is associated with an image index and is added as a patch to the corresponding subplot.
        - If there are no shapes to be drawn (`self.__shapes` is empty), the method returns the figure and axes unmodified.

        Shape Drawing Process:
        - The method first checks if there are any shapes to be drawn. If not, it returns the figure and axes unmodified.
        - For each shape defined in `self.__shapes`, the method retrieves the image index and the patches (shapes) to be drawn.
        - Each patch is added to the corresponding axis using the `add_patch` method.
        - The method ensures that the shapes are drawn when the figure is displayed by updating the axes with the patches.
        """
        if not self.__shapes:
            return fig, axes
        
        for img_idx, patches in self.__shapes.items():
            for patch in patches:
                # make a copy to prevent run-time errors when plotting the same figure the second time:
                patch = copy.deepcopy(patch)
                axes[img_idx].add_patch(patch)
                
        return fig, axes
    
    def add_line(self, coord1, coord2, color="skyblue", linewidth=1.0, 
                 linestyle='dashed', **kwargs) -> None:
        """
        Add a line between two coordinates on the image matrix.

        Parameters:
        - coord1 (tuple): The starting coordinate of the line in the format 
          (image index, x coordinate, y coordinate).
        - coord2 (tuple): The ending coordinate of the line in the format 
          (image index, x coordinate, y coordinate).
        - color (str, optional): Color of the line. Default is 'k' (black).
        - linewidth (float, optional): Width of the line. Default is 1.0.
        - linestyle (str, optional): Style of the line. Default is '-' (solid line).
        - **kwargs: Additional keyword arguments to be passed to the 'plot' method.

        Returns:
        - None

        Raises:
        - ValueError: If `coord1` or `coord2` are not in the correct format or the indices are invalid.

        Description:
        - This method draws a line between two specified coordinates on the image matrix.
        - The coordinates must include the image index and the x and y positions within that image.
        """
        if len(coord1) != 3 or len(coord2) != 3:
            raise ValueError("Length of coordinates specified must be 3 " +\
                             "with the format (image index, x, y)")
            
        img_idx1, *_ = coord1
        img_idx2, *_ = coord2
        if img_idx1 > self.size - 1 or img_idx2 > self.size - 1:
            raise ValueError("Image index specified exceeds size of ImageMatrix.")

        kwargs.update({"color": color, 
                       "linewidth": linewidth,
                       "linestyle": linestyle})
        self.__lines[(coord1, coord2)] = kwargs
        
    def __draw_lines(self, fig, axes):
        """
        Draw stored lines across different axes in the image matrix.

        Parameters:
        - fig (matplotlib.figure.Figure): The figure object containing the plots.
        - axes (list of matplotlib.axes.Axes): The axes objects of the plots.

        Returns:
        - fig (matplotlib.figure.Figure): The updated figure object with the lines drawn.
        - axes (list of matplotlib.axes.Axes): The updated axes objects.

        Description:
        - This method draws lines stored in `self.__lines` on the specified figure and axes.
        - Each line is defined by two coordinates, where each coordinate includes the image index and
          the (x, y) position within that image.
        - The method transforms data coordinates to figure coordinates to draw lines that span across
          different subplots.
        - If there are no lines to be drawn (`self.__lines` is empty), the method returns the figure and axes unmodified.

        Line Drawing Process:
        - The method first checks if there are any lines to be drawn. If not, it returns the figure and axes unmodified.
        - It then draws the canvas and uses the `transFigure` transformation to convert data coordinates to figure coordinates.
        - For each line defined in `self.__lines`, the method retrieves the starting and ending coordinates, transforms them, 
          and creates a `Line2D` object to draw the line across the figure.
        - The `Line2D` object is appended to the figure's lines, ensuring the line is drawn when the figure is displayed.
        """
        # don't do anything if nothing needs to be plotted
        if not self.__lines:
            return fig, axes
        
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        
        for (coord1, coord2), line_kwargs in self.__lines.items():
            idx1, x1, y1 = coord1
            idx2, x2, y2 = coord2
            
            ax1 = axes[idx1]
            ax2 = axes[idx2]
            
            coord1_fig = transFigure.transform(ax1.transData.transform([x1, y1]))
            coord2_fig = transFigure.transform(ax2.transData.transform([x2, y2]))
            
            line = plt.Line2D((coord1_fig[0], coord2_fig[0]), (coord1_fig[1], coord2_fig[1]),
                              transform=fig.transFigure, **line_kwargs)
            fig.lines.append(line)
        
        return fig, axes
            
    def clear_annotations(self, inplace=True) -> None:
        """
        Clear all stored annotations (lines, shapes, and texts) from the image matrix.
        
        Parameters:
        - inplace (bool): Whether to modify the image matrix in place. Default is True.
        
        Description:
        - This method clears all lines, shapes, and text annotations that have been added to the image matrix.
        - After calling this method, the image matrix will have no annotations.
        - This can be useful if you want to reset the annotations and start fresh.
        """
        matrix = self if inplace else self.copy()
        matrix.__lines = {}
        matrix.__shapes = {}
        matrix.__texts = {}
        matrix.__label_abc = False
        
    def savefig(self, fname, format="pdf", bbox_inches="tight", transparent=True, 
                overwrite=False, **kwargs) -> None:
        """
        Save the current figure to a file.

        Parameters:
        - fname (str): The name of the file to save the figure to. If no extension is provided, '.pdf' will be added.
        - format (str, optional): The format to save the figure in. Default is 'pdf'.
        - bbox_inches (str, optional): Bounding box in inches. Default is 'tight'.
        - transparent (bool, optional): If True, the background of the figure will be transparent. Default is True.
        - overwrite (bool, optional): Whether to overwrite an existing file with the same name. Default is False.
        - **kwargs: Additional keyword arguments to pass to `fig.savefig`.

        Returns:
        - None

        Raises:
        - Exception: If no figure is available to save.

        Description:
        - Checks if a figure is available to save; raises an exception if not.
        - Adds a file extension to the filename if it is not provided.
        - If `overwrite` is False and a file with the specified name exists, 
          modifies the filename to avoid overwriting by appending a number.
        - Saves the figure with the specified parameters.
        """
        fig = self.fig
        
        # check if figure needs to be plotted:
        if fig is None:
            raise Exception("No figure available to save. Plot the figure using '.plot()' before saving.")
            
        # check if format is correct:
        if not format.islower():  # make it case insensitive
            format = format.lower()
        supported_formats = mpl.figure.FigureCanvasBase.get_supported_filetypes()
        if format not in supported_formats:
            supported_formats = str(tuple(supported_formats))[1:-1]
            raise ValueError(f"{format} is not a supported filetype: {supported_formats}.")
        
        # add file extension if not provided:
        extension = "." + format
        len_ext = len(extension)
        if not fname.endswith(extension):
            fname += extension
        
        # add numbers if file exists:
        if not overwrite:
            fname = _prevent_overwriting(fname)
        
        # save file:
        fig.savefig(fname, format=format, bbox_inches=bbox_inches, 
                    transparent=transparent, **kwargs)
        
        # print directory:
        print(f"Image successfully saved as '{fname}'")

    def align_center(self, row) -> None:
        ncol: int = self.shape[1]
        begin_idx: int = self.__ravel_idx((row, 0))
        end_idx: int = self.__ravel_idx((row, ncol))  # not included
        number_of_nonblank_images: int = sum(image is not None for image in self.images[begin_idx, end_idx])
        positions: np.ndarray = np.linspace(0, 1, number_of_nonblank_images+2)

        i: int = 0  # increases by one when image is not None
        for idx in enumerate(range(begin_idx, end_idx)):
            if self.images[idx] is None:
                continue
            self.set_horizontal_position(self, idx, center=positions[i])
            i += 1  # increment

    def set_horizontal_position(self, img_idx, center=0.5) -> None:
        # warning message
        if self.axes_padding == "auto":
            warnings.warn("Setting 'axes_padding' to 'auto' may result in different subplot sizes.")

        # parse and ravel image index:
        img_idx = self.__ravel_index(img_idx)
        
        # parse central coordinate:
        if not isinstance(center, (float, np.floating, int, np.integer)):
            raise ValueError("'center' must be an iterable or a float, int, or their numpy equivalents")
            
        if not (0 <= center <= 1):
            raise ValueError("'center' must be within the range [0, 1]")
        
        self.__set_positions[img_idx] = center
        
    def __set_horizontal_positions(self, fig, axes):
        if not self.__set_positions:
            return fig, axes
        
        for idx, center in self.__set_positions.items():
            # get ax
            ax = axes[idx]
            
            # get current position to calculate width and height
            points = ax.get_position().get_points()
            x1, y1 = points[0]
            x2, y2 = points[1]
            width = abs(x2-x1)
            height = abs(y2-y1)
            
            # calculate new position
            left = center - width / 2
            bottom = y1 - height / 2  
            new_position = [left, bottom, width, height]
            
            # set new position
            ax.set_position(new_position)
        return fig, axes
        
    def clean_labels(self, only_one=False) -> None:
        """
        Adjusts the labels and color bar labels in a grid of subplots.

        This method iterates over a grid of subplots and ensures that:
            - X-axis and Y-axis labels are only displayed on the bottom-left subplot.
            - Color bar labels are only displayed on the rightmost subplots.

        The method updates the `specific_kwargs` dictionary for each subplot, which is used to configure
        the display properties of each subplot in the grid.
        """
        if only_one:
            for i in range(self.shape[0]):  # row 
                for j in range(self.shape[1]):  # column
                    # turn on labels only at bottom left corner
                    if i == self.shape[0] - 1 and j == 0:
                        xlabelon = True
                        ylabelon = True
                    else:
                        xlabelon = False
                        ylabelon = False

                    # turn on color bar labels only on right/top columns
                    if self.__cbarloc == "right":
                        cbarlabelon = (j == self.shape[1] - 1)
                    else:
                        cbarlabelon = (i == 0)

                    # set parameters
                    ravelled_idx = self.__ravel_index((i, j))
                    if ravelled_idx > len(self.images) - 1:
                        break
                    self.specific_kwargs[ravelled_idx]["xlabelon"] = xlabelon
                    self.specific_kwargs[ravelled_idx]["ylabelon"] = ylabelon
                    self.specific_kwargs[ravelled_idx]["cbarlabelon"] = cbarlabelon
        else:
            for i in range(self.shape[0]):  # row 
                for j in range(self.shape[1]):  # column
                    # turn on labels only at bottom left corner
                    xlabelon = False
                    ylabelon = False
                    if i == self.shape[0] - 1:
                        xlabelon = True
                    if j == 0:
                        ylabelon = True

                    # turn on color bar labels only on right/top columns
                    if self.__cbarloc == "right":
                        cbarlabelon = (j == self.shape[1] - 1)
                    else:
                        cbarlabelon = (i == 0)

                    # set parameters
                    ravelled_idx = self.__ravel_index((i, j))
                    if ravelled_idx > len(self.images) - 1:
                        break
                    self.specific_kwargs[ravelled_idx]["xlabelon"] = xlabelon
                    self.specific_kwargs[ravelled_idx]["ylabelon"] = ylabelon
                    self.specific_kwargs[ravelled_idx]["cbarlabelon"] = cbarlabelon
        
    def pop(self, image_location=-1, inplace=True):
        """
        Remove and return the image at the specified location from the image matrix.

        Parameters:
        - image_location (int or tuple, optional): The index or coordinates of the image to remove.
          Default is -1, which removes the last image.
        - inplace (bool, optional): Whether to modify the image matrix in place.
          Default is True.

        Returns:
        - image (object): The removed image.

        Raises:
        - ValueError: If the specified location is invalid or exceeds the size of the image matrix.

        Description:
        - If `inplace` is False, creates a copy of the image matrix to modify.
        - If `image_location` is a tuple, it should contain two elements representing coordinates.
          The location is then flattened to a single index.
        - If the specified `image_location` exceeds the size of the image matrix, a ValueError is raised.
        - The image at the specified location is removed and replaced with `None`.
        """
        # parse index
        image_location = self.__ravel_index(image_location)
        
        # make copy if user doesn't want to modify in-place
        matrix = self if inplace else self.copy()
        
        # remove image by replacing it with a blank one
        image = matrix.images[image_location]
        matrix.images[image_location] = None
        
        return image


def plt_1ddata(xdata=None, ydata=None, xlim=None, ylim=None, mean_center=False, title=None,
               legendon=True, xlabel="", ylabel="", threshold=None, linewidth=0.8,
               xtick_spacing=None, ytick_spacing=None, borderwidth=0.7,
               labelsize=7, fontsize=8, ticksize=3, legendsize=6, title_position=0.92,
               bbox_to_anchor=(0.6, 0.95), legendloc="best", threshold_color="gray",
               linecolor="k", figsize=(2.76, 2.76), bins="auto", hist=False,
               dpi=600, plot=True, **kwargs):
    
    fontsize = labelsize if fontsize is None else fontsize
    
    if xlim is None:
        xlim = (np.nanmin(xdata), np.nanmax(xdata))
    if ylim is None:
        ylim = []
    
    params = {'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'font.size': fontsize,
              'legend.fontsize': legendsize,
              'xtick.labelsize': labelsize,
              'ytick.labelsize': labelsize,
              'figure.figsize': figsize,
              'figure.dpi': dpi,
              'font.family': _fontfamily,
              "mathtext.fontset": _mathtext_fontset, #"Times New Roman"
              'mathtext.tt': _mathtext_tt,
              'axes.linewidth': borderwidth,
              'xtick.major.width': borderwidth,
              'ytick.major.width': borderwidth,
              'xtick.major.size': ticksize,
              'ytick.major.size': ticksize,
              }
    rcParams.update(params)
    
    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False)
    plt.subplots_adjust(wspace=0.4)
    
    if hist:
        ax.hist(xdata, bins=bins, lw=linewidth, **kwargs)
    else:
        ax.plot(xdata, ydata, color=linecolor, lw=linewidth, **kwargs)
        
    if legendon:
        ax.legend(frameon=False, loc=legendloc, bbox_to_anchor=bbox_to_anchor)
        
    ax.tick_params(which='both',direction='in',bottom=True, top=True, left=True, right=True,
                   colors="k", labelrotation=0, labelcolor="k")
    
    if len(xlim) == 2:
        ax.set_xlim(xlim)
    if len(ylim) == 2:
        ax.set_ylim(ylim)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    if xtick_spacing is not None:
        ax.set_xticks(np.arange(-100, 100, xtick_spacing))
        ax.set_xlim(xlim)
    if ytick_spacing is not None:
        ax.set_yticks(np.arange(-100, 100, ytick_spacing))
        ax.set_ylim(ylim)
        
    if mean_center:
        xlim = ax.get_xlim()
        half_range = (np.max(xlim) - np.min(xlim))/2
        xmean = np.trapz(x=xdata, y=xdata*ydata)/np.trapz(x=xdata, y=ydata)
        xlim = (xmean-half_range, xmean+half_range)
        ax.set_xlim(xlim)
    
    if title is not None:
        xlim = ax.get_xlim() 
        ylim = ax.get_ylim() 
        xrange = np.max(xlim) - np.min(xlim)
        yrange = np.max(ylim) - np.min(ylim)
        xposition = xlim[0]+(1-title_position)*xrange
        yposition = ylim[0]+title_position*yrange
        ax.text(x=xposition, y=yposition, 
                s=title, color="k", ha="left", va="top", fontsize=fontsize)
    
    if threshold is not None:
        ax.axhline(y=threshold, linestyle="dashed", linewidth=linewidth, color=threshold_color)
        
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    
    if plot:
        plt.plot()
    
    return ax


def exportfits(image, *args, **kwargs):
    """
    Duplicate function of 'exportfits' method to export the given image to a FITS file.

    Parameters:
    - image (object): An image instance that has an `exportfits` method.
    - *args: Additional positional arguments to be passed to the `exportfits` method of the image instance.
    - **kwargs: Additional keyword arguments to be passed to the `exportfits` method of the image instance.

    Returns:
    - The result of the `exportfits` method of the image instance.

    Raises:
    - AttributeError: If the 'image' parameter does not have an `exportfits` method.
    
    Description:
    - This function acts as a wrapper around the `exportfits` method of the provided image instance.
    - It simply forwards any additional arguments to the `exportfits` method of the image instance.
    """
    return image.exportfits(*args, **kwargs)


def imview(image, *args, **kwargs):
    """
    Duplicated function of 'imview' method to quickly plot a FITS file if directory is known

    Parameters:
    - image (str or object): The image to be displayed. This can be:
        - A string representing the directory path to a FITS file.
        - An instance of Datacube, Spatialmap, or PVdiagram.
    - *args: Additional positional arguments to be passed to the `imview` method of the image instance.
    - **kwargs: Additional keyword arguments to be passed to the `imview` method of the image instance.

    Returns:
    - The result of the `imview` method of the image instance.

    Raises:
    - ValueError: If the 'image' parameter is neither a directory to a FITS file nor an image instance.

    Description:
    - If `image` is an instance of Datacube, Spatialmap, or PVdiagram, calls its `imview` method.
    - If `image` is a string, treats it as a directory to a FITS file, imports the FITS file, and calls its `imview` method.
    - Otherwise, raises a ValueError indicating an invalid 'image' parameter.
    """
    if isinstance(image, (Datacube, Spatialmap, PVdiagram)):
        return image.imview(*args, **kwargs)
    elif isinstance(image, str):
        image = importfits(image)
        return image.imview(*args, **kwargs)
    else:
        raise ValueError("'image' parameter must be a directory to a FITS file or an image instance.")


def search_molecular_line(restfreq, unit="GHz", species_id=None, 
                          printinfo=True, return_table=False):
    """
    Search for molecular line information given a rest frequency.
    
    Parameters:
    - restfreq (float): The rest frequency of the molecular line.
    - unit (str, optional): The unit of the rest frequency. Default is "GHz".
    - printinfo (bool, optional): Whether to print the retrieved information. Default is True.
    
    Returns:
    - tuple: A tuple containing the following information about the molecular line:
        - species (str): The species of the molecular line.
        - chemical_name (str): The chemical name of the species.
        - freq (float): The frequency of the molecular line in GHz.
        - freq_err (float): The measurement error of the frequency in GHz.
        - qns (str): The resolved quantum numbers.
        - CDMS_JPL_intensity (float): The CDMS/JPL intensity.
        - Sijmu_sq (float): The S_ij * mu^2 value in Debye^2.
        - Sij (float): The S_ij value.
        - Aij (float): The Einstein A coefficient in 1/s.
        - Lovas_AST_intensity (float): The Lovas/AST intensity.
        - lerg (float): The lower energy level in Kelvin.
        - uerg (float): The upper energy level in Kelvin.
        - gu (float): The upper state degeneracy.
        - constants (tuple): The rotational constants (A, B, C) in MHz.
        - source (str): The source of the data.
    
    Raises:
    - ValueError: If the rest frequency is None.
    
    Notes:
    - This function requires an internet connection to query the Splatalogue database.
    - If the frequency does not closely match any known molecular lines, a warning will be printed.
    """
    # error checking for rest frequency
    if restfreq is None:
        raise ValueError("The rest frequency cannot be 'None'.")
        
    if unit != "GHz":
        restfreq = u.Quantity(restfreq, unit).to_value(u.GHz)   # convert unit to GHz
    
    results = _best_match_line(restfreq, species_id=species_id, return_table=return_table)
    if return_table:
        return results

    # find information of the line 
    species_id = results["SPECIES_ID"]
    species = results["SPECIES"]
    chemical_name = results["CHEMICAL_NAME"]
    freq = results["FREQUENCY"]
    qns = results["QUANTUM_NUMBERS"]
    intensity = results["INTENSITY"]
    Sijmu_sq = results["SMU2"]
    log10_Aij = results["LOGA"]
    Aij = 10**log10_Aij
    lerg = results["EL"]
    uerg = results["EU"]
    try:
        upper_state, lower_state = map(int, qns.strip("J=").split('-'))
        gu = 2 * upper_state + 1
    except ValueError:
        warnings.warn("Failed to calculated upper state degeneracy.")
        gu = None
    source = results["LINELIST"]

    # find species id
    if species_id is None:
        warnings.warn("Failed to find species ID / rotational constants. \n")
        constants = None
        url = None
        display_url = None
    else:
        # find rotational constant
        url = f"https://splatalogue.online/splata-slap/species/{species_id}"
        display_url = f"https://splatalogue.online/#/species?id={species_id}"
        try:
            # search the web for rotational constants 
            from urllib.request import urlopen
            page = urlopen(url)
            html = page.read().decode("utf-8")
            metadata = eval(html.replace("null", "None"))[0]['metaData']  # list of dictionaries
        except:
            print(f"Failed to read webpage: {display_url} \n")
            print(f"Double check internet connection / installation of 'urllib' module.")
            url = None
            constants = None
        else:
            a_const = metadata.get("A")
            b_const = metadata.get("B")
            c_const = metadata.get("C")
            constants = tuple((float(rot_const) if rot_const is not None else None) \
                              for rot_const in (a_const, b_const, c_const))
                
    # store data in a list to be returned and convert masked data to NaN
    data = [species, chemical_name, freq, None, qns, intensity, Sijmu_sq,
            None, Aij, None, lerg, uerg, gu, constants, source]
    
    for i, item in enumerate(data):
        if np.ma.is_masked(item):
            data[i] = np.nan
    
    # print information if needed
    if printinfo:
        print(15*"#" + "Line Information" + 15*"#")
        print(f"Species ID: {species_id}")
        print(f"Species: {data[0]}")
        print(f"Chemical name: {data[1]}")
        print(f"Frequency: {data[2]} +/- {data[3]} [GHz]")
        print(f"Resolved QNs: {data[4]}")
        if not np.isnan(data[5]) and data[5] != 0:
            print(f"Intensity: {data[5]}")        
        print(f"Sij mu2: {data[6]} [Debye2]")
        print(f"Sij: {data[7]}")
        print(f"Einstein A coefficient (Aij): {data[8]:.3e} [1/s]")
        print(f"Lower energy level: {data[10]} [K]")
        print(f"Upper energy level: {data[11]} [K]")
        print(f"Upper state degeneracy (gu): {data[12]}")
        if data[13] is not None:
            print("Rotational constants:")
            if data[13][0] is not None:
                print(f"    A0 = {data[13][0]} [MHz]")
            if data[13][1] is not None:
                print(f"    B0 = {data[13][1]} [MHz]")
            if data[13][2] is not None:
                print(f"    C0 = {data[13][2]} [MHz]")
        print(f"Source: {data[14]}")
        print(46*"#")
        if url is not None:
            print(f"Link to species data: {display_url} \n")
    
    # return data
    return tuple(data)


def planck_function(v, T):
    """
    Public function to calculate the planck function value.
    Parameters:
        v (float): frequency of source [GHz]
        T (float): temperature [K]
    Returns:
        Planck function value [Jy]
    """
    # constants
    h = const.h.cgs
    clight = const.c.cgs
    k = const.k_B.cgs
    
    # assign units
    if not isinstance(v, u.Quantity):
        v *= u.GHz
    if not isinstance(T, u.Quantity):
        T *= u.K
    
    # calculate
    Bv = 2*h*v**3/clight**2 / (np.exp(h*v/k/T)-1)
    
    # return value
    return Bv.to_value(u.Jy)


def clip_percentile(data, area=0.95):
    """
    Public function to calculate the interval containing the percentile (from middle).
    Parameters:
        data: the 2D data of an image
        area: the total area (from center)
        bins: the number of bins of the histogram
        xlim: the x field of view of the histogram
    Returns:
        The interval (tuple(float)) containing the area (from middle).
    """
    if 0 <= area <= 1:
        area *= 100
    else:
        raise ValueError("Percentile must be between 0 and 1.")
        
    upper_tail = (100+area)/2
    lower_tail = (100-area)/2
    flattened_data = data.flatten()
    flattened_data = flattened_data[~np.isnan(flattened_data)]
    lower_bound = np.percentile(flattened_data, lower_tail, method="linear")
    upper_bound = np.percentile(flattened_data, upper_tail, method="linear")
    interval = (lower_bound, upper_bound)
    
    # print info
    print(f"{area}th percentile".center(40, "#"))
    print(f"Interval: {interval}")
    print(40*"#", end="\n\n")

    return interval


def H2_column_density(continuum, T_dust, k_v):
    """
    Public function to calculate the H2 column density from continuum data.
    Parameters:
        continuum (Spatialmap): the continuum map
        T_dust (float): dust temperature [K]
        k_v (float): dust-mass opacity index [cm^2/g]
    Returns:
        H2_cd (Spatialmap): the H2 column density map [cm^-2]
    """
    # constants
    m_p = const.m_p.cgs  # proton mass
    mmw = 2.8            # mean molecular weight
    
    # convert continuum to brightness temperature
    I_v = continuum.conv_bunit("Jy/sr", inplace=False)
    if not isinstance(I_v.data, u.Quantity):
        I_v *= u.Jy
    v = continuum.restfreq * u.Hz
    
    # assign units:
    if not isinstance(T_dust, u.Quantity):
        T_dust *= u.K
    if not isinstance(k_v, u.Quantity):
        k_v *= u.cm**2 / u.g
    
    # start calculating
    B_v = planck_function(v=v.to_value(u.GHz), T=T_dust)*u.Jy
    H2_cd = (I_v / (k_v*B_v*mmw*m_p)).to_value(u.cm**-2)
    return H2_cd


def J_v(v, T):
    """
    Public function to calculate the Rayleigh-Jeans equivalent temperature.
    Parameters:
        v (float): frequency (GHz)
        T (float): temperature (K)
    Returns:
        Jv (float): Rayleigh-Jeans equivalent temperature (K)
    """
    # constants
    k = const.k_B.cgs
    h = const.h.cgs
    
    # assign unit
    if not isinstance(v, u.Quantity):
        v *= u.GHz
    if not isinstance(T, u.Quantity):
        T *= u.K
    
    # calculate R-J equivalent temperature
    Jv = (h*v/k) / (np.exp(h*v/k/T)-1)
    return Jv.to(u.K)


def column_density_linear_optically_thin(image, T_ex, T_bg=2.726, B0=None, R_i=1, f=1.):
    """
    Public function to calculate the column density of a linear molecule using optically thin assumption.
    Source: https://doi.org/10.48550/arXiv.1501.01703
    Parameters:
        image (Spatialmap/Datacube): the moment 0 / datacube map
        T_ex (float): the excitation temperature [K]
        T_bg (float): background temperature [K]. 
                      Default is to use cosmic microwave background (2.726 K).
        R_i (float): Relative intensity of transition. 
                     Default is to consider simplest case (= 1)
        f (float): a correction factor accounting for source area being smaller than beam area.
                   Default = 1 assumes source area is larger than beam area.
    Returns:
        cd_img (Spatialmap/Datacube): the column density map
    """
    # constants
    k = const.k_B.cgs
    h = const.h.cgs
    
    # assign units
    if not isinstance(T_ex, u.Quantity):
        T_ex *= u.K
    if not isinstance(T_bg, u.Quantity):
        T_bg *= u.K
    
    # convert units 
    if isinstance(image, Spatialmap):
        image = image.conv_bunit("K.km/s", inplace=False)*u.K*u.km/u.s
    elif isinstance(image, Datacube):
        image = image.conv_bunit("K", inplace=False)
        image = image.conv_specunit("km/s", inplace=False)
        dv = image.header["dv"]
        image = image*dv*(u.K*u.km/u.s)
    else:
        raise ValueError(f"Invalid data type for image: {type(image)}")
    
    # get info
    line_data = image.line_info(printinfo=True)
    v = line_data[2]*u.GHz  # rest frequency
    S_mu2 = line_data[6]*(1e-18**2)*(u.cm**5*u.g/u.s**2)  # S mu^2 * g_i*g_j*g_k [debye2]
    E_u = line_data[11]*u.K  # upper energy level 
    if B0 is None:
        B0 = line_data[13][1]*u.MHz  # rotational constant
    elif not isinstance(B0, u.Quantity):
        B0 *= u.MHz
    Q_rot = _Qrot_linear(T=T_ex, B0=B0)  # partition function
        
    # error checking to make sure molecule is linear
    if line_data[13] is not None:
        if line_data[13][0] is not None or line_data[13][2] is not None:
            raise Exception("The molecule is not linear.")

    # calculate column density
    aa = 3*h/(8*np.pi**3*S_mu2*R_i)
    bb = Q_rot
    cc = np.exp(E_u/T_ex) / (np.exp(h*v/k/T_ex)-1)
    dd = 1 / (J_v(v=v, T=T_ex)-J_v(v=v, T=T_bg))
    constant = aa*bb*cc*dd/f
    cd_img = constant*image
    cd_img = cd_img.to_value(u.cm**-2)
    return cd_img


def column_density_linear_optically_thick(moment0_map, T_ex, tau, T_bg=2.726, R_i=1, f=1):
    """
    Function to calculate the column density of a linear molecule using optically thick assumption.
    Source: https://doi.org/10.48550/arXiv.1501.01703
    Parameters:
        moment0_map (Spatialmap): the moment 0 map
        T_ex (float): the excitation temperature [K]
        tau (float): gas optical depth
        T_bg (float): background temperature [K]. 
                      Default is to use cosmic microwave background (2.726 K).
        R_i (float): Relative intensity of transition. 
                     Default is to consider simplest case (= 1)
        f (float): a correction factor accounting for source area being smaller than beam area.
                   Default = 1 assumes source area is larger than beam area.
    Returns:
        cd_img (Spatialmap): the column density map
    """
    # calculate using optically thin assumption
    cd_opt_thin = column_density_linear_optically_thin(moment0_map=moment0_map,
                                                       T_ex=T_ex,
                                                       T_bg=T_bg,
                                                       R_i=R_i,
                                                       f=f)
    # correction factor, relates optically thin to optically thick case
    corr_factor = tau/(1-np.exp(-tau))
    cd_img = corr_factor*cd_opt_thin
    return cd_img


def to_casa(image, outname, whichrep=0, whichhdu=-1, 
            zeroblanks=True, overwrite=False, defaultaxes=False,
            defaultaxesvalues=[], beam=[]) -> None:
    """
    Function to convert image object into CASA image format. 
    Wraps the 'importfits' function of casatasks.

    Parameters:
        image (Datacube, Spatialmap, PVdiagram): The image object to be converted.
        outname (str): The output name for the CASA image file. Must end with ".image".
        whichrep (int, optional): The FITS representation to convert. Defaults to 0.
        whichhdu (int, optional): The HDU (Header/Data Unit) number to convert. Defaults to -1.
        zeroblanks (bool, optional): Replace undefined values with zeros. Defaults to True.
        overwrite (bool, optional): Overwrite the output file if it already exists. Defaults to False.
        defaultaxes (bool, optional): Use default axes for the output CASA image. Defaults to False.
        defaultaxesvalues (str, optional): Default axes values, provided as a string. Defaults to '[]'.
        beam (str, optional): Beam parameters, provided as a string. Defaults to '[]'.

    Raises:
        ValueError: If 'image' is not an instance of 'Datacube', 'Spatialmap', or 'PVdiagram'.
        ValueError: If 'outname' is not a string.

    Returns:
        None

    Example:
        to_casa(my_image, "output_image")
    """
    # within-function import statement
    import casatasks

    # error checking & parse parameters
    if not isinstance(image, (Datacube, Spatialmap, PVdiagram)):
        raise ValueError("'image' must be an instance of 'Datacube', 'Spatialmap', or 'PVdiagram'")

    if not isinstance(outname, str):
        raise ValueError("'outname' must be a string (a directory that the image will be saved as)")

    if not outname.endswith(".image"):
        outname += ".image"  # add file extension

    if not overwrite:
        outname = _prevent_overwriting(outname)

    # generate temporary FITS file
    temp_fits_name = str(dt.datetime.now()).replace(" ", "") + ".fits"
    temp_fits_name = _prevent_overwriting(temp_fits_name)
    hdu = fits.PrimaryHDU(data=image.data, header=image.get_hduheader())
    hdu.writeto(temp_fits_name, overwrite=False)

    # convert FITS file to CASA image format
    casatasks.importfits(fitsimage=temp_fits_name,
                         imagename=outname,
                         whichrep=whichrep,
                         whichhdu=whichhdu,
                         zeroblanks=zeroblanks,
                         overwrite=overwrite,
                         defaultaxes=defaultaxes,
                         defaultaxesvalues=defaultaxesvalues,
                         beam=beam)

    # remove temporary FITS file
    os.remove(temp_fits_name)

    # Tell user that the image has been successfully saved.
    print(f"Image successfully saved as '{outname}' in CASA format")
    

# ------------------- Below are functions intended for internal use -----------------------


def _get_moment_map(moment: int, data: np.ndarray, 
                    vaxis: np.ndarray, ny: int, 
                    nx: int, keep_nan: bool,
                    bunit: str, specunit: str,
                    header: dict) -> np.ndarray:
    """
    Private method to get the data array of the specified moment map.
    Used for the public method 'immoments'.
    """
    # intialize parameters
    nv = vaxis.size
    dv = vaxis[1] - vaxis[0]

    # mean value of spectrum (unit: intensity)
    if moment == -1:
        momdata = np.nanmean(data, axis=1)
    
    # integrated value of the spectrum (unit: intensity*km/s)
    elif moment == 0:
        momdata = np.nansum(data, axis=1)*dv
    
    # intensity weighted coordinate (unit: km/s) 
    elif moment == 1:
        reshaped_vaxis = vaxis[np.newaxis, :, np.newaxis, np.newaxis]
        momdata = np.nansum(data*reshaped_vaxis, axis=1) / np.nansum(data, axis=1)
    
    # intensity weighted dispersion of the coordinate (unit: km/s)
    elif moment == 2:
        reshaped_vaxis = vaxis.reshape((1, nv, 1, 1))
        vv = np.broadcast_to(reshaped_vaxis, (1, nv, ny, nx))
        meanvel = np.nansum(data*vv, axis=1) / np.nansum(data, axis=1)
        momdata = np.sqrt(np.nansum(data*(vv-meanvel)**2, axis=1)/np.nansum(data, axis=1))
        
    # median value of the spectrum (unit: intensity)
    elif moment == 3:
        momdata = np.nanmedian(data, axis=1)
    
    # median coordinate (unit: km/s)
    elif moment == 4:
        momdata = np.empty((1, 1, data.shape[2], data.shape[3]))
        for i in range(momdata.shape[2]):
            for j in range(momdata.shape[3]):
                pixdata = data[0, :, i, j]  # shape: (nv,)
                if np.all(np.isnan(pixdata)):
                    momdata[0, 0, i, j] = np.nan
                else:
                    sorted_data = np.sort(pixdata)
                    sorted_data = sorted_data[~np.isnan(sorted_data)] # remove nan so that median is accurate
                    median_at_pix = sorted_data[sorted_data.size//2]  # this is approximation of median, not always exact
                    vidx = np.where(pixdata == median_at_pix)[0][0]   # get index of coordinate of median
                    momdata[0, 0, i, j] = vaxis[vidx]
    
    # standard deviation about the mean of the spectrum (unit: intensity)
    elif moment == 5:
        momdata = np.nanstd(data, axis=1)
    
    # root mean square of the spectrum (unit: intensity)
    elif moment == 6:
        momdata = np.sqrt(np.nanmean(data**2, axis=1))
    
    # absolute mean deviation of the spectrum (unit: intensity)
    elif moment == 7:
        momdata = np.nanmean(np.abs(data-np.nanmean(data, axis=1)), axis=1)
        
    # maximum value of the spectrum (unit: intensity)
    elif moment == 8:
        momdata = np.nanmax(data, axis=1)
        
    # coordinate of the maximum value of the spectrum (unit: km/s)
    elif moment == 9:
        momdata = np.empty((1, 1, data.shape[2], data.shape[3]))
        for i in range(momdata.shape[2]):
            for j in range(momdata.shape[3]):
                pixdata = data[0, :, i, j]  # intensity values of data cube at this pixel
                if np.all(np.isnan(pixdata)):
                    momdata[0, 0, i, j] = np.nan  # assign nan if all nan at that pixel
                else:
                    momdata[0, 0, i, j] = vaxis[np.nanargmax(pixdata)]
        
    # minimum value of the spectrum (unit: intensity)
    elif moment == 10:
        momdata = np.nanmin(data, axis=1)
    
    # coordinate of the minimum value of the spectrum (unit: km/s) 
    elif moment == 11:
        momdata = np.empty((1, 1, data.shape[2], data.shape[3]))
        for i in range(momdata.shape[2]):
            for j in range(momdata.shape[3]):
                pixdata = data[0, :, i, j]  # intensity values of data cube at this pixel
                if np.all(np.isnan(pixdata)):
                    momdata[0, 0, i, j] = np.nan  # assign nan if all nan at that pixel
                else:
                    momdata[0, 0, i, j] = vaxis[np.nanargmin(pixdata)]  

    # reshape moment map data
    momdata = momdata.reshape((1, 1, ny, nx))
                    
    # replace map pixels with nan if entire spectrum at that pixel is nan
    if keep_nan:
        mask = np.all(np.isnan(data), axis=1, keepdims=True)
        momdata[mask] = np.nan

    if moment in (-1, 3, 5, 6, 7, 8, 10):
        bunit = bunit
    elif moment == 0:
        bunit = f"{bunit}.{specunit}"
    else:
        bunit = specunit

    # update header information
    newheader = copy.deepcopy(header)
    newheader["imagetype"] = "spatialmap"
    newheader["shape"] = momdata.shape
    newheader["vrange"] = None
    newheader["nchan"] = 1
    newheader["dv"] = None
    newheader["bunit"] = bunit
    
    return Spatialmap(header=newheader, data=momdata)
