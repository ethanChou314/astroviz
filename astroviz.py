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

# numerical packages
import numpy as np

# standard packages
import copy
import os
import sys
import datetime as dt

# scientific packages
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from astropy import units as u, constants as const
from astropy.units import Unit, def_unit, UnitConversionError
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.coordinates import Angle, SkyCoord
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian2DKernel, Box2DKernel, convolve, convolve_fft

# data visualization packages
import matplotlib as mpl
from matplotlib import cm, rcParams, ticker, patches, colors, pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# clarify module's public interface
__all__ = ["importfits", "Datacube", "Spatialmap", "PVdiagram", "Region", "plt_1ddata",
           "search_molecular_line", "set_font", "exportfits"]

# variables to control all map fonts globally
_fontfamily = "Times New Roman"
_mathtext_fontset = "stix"
_mathtext_tt = "Times New Roman"


def importfits(fitsfile, hduindex=0, spatialunit="arcsec", specunit="km/s", quiet=False):
    """
    This function reads the given FITS file directory and returns the corresponding image object.
    Parameters:
        fitsfile (str): the directory of the fitsfile. 
        hduindex (int): the index of the HDU in the list of readable HDUs
        spatialunit (str): the unit of the spatial axes
        quiet (bool): Nothing will be printed to communicate with user. 
                      Useful for iterative processes, as it allows for faster file reading time.
    Returns:
        The image object (Datacube/Spatialmap/PVdiagram)
    ----------
    Additional notes:
        If given fitsfile directory does not exist, the function will attempt to recursively 
        find the file within all subdirectories.
    """
    # function that recursively finds file
    def find_file(file):  
        for root, dirs, files in os.walk(os.getcwd()):
            if file in files:
                return os.path.join(root, file)
        return None 
    
    # try to recursively find file if relative directory does not exist
    if not os.path.exists(fitsfile):
        if not quiet:
            print(f"Given directory '{fitsfile}' does not exist as a relative directory. " + \
                   "Recursively finding file...")
        maybe_filename = find_file(fitsfile)
        if maybe_filename is not None:
            fitsfile = maybe_filename
            if not quiet:
                print(f"Found a matching filename: '{fitsfile}'")
        else:
            raise FileNotFoundError(f"Filename '{fitsfile}' does not exist.")
    
    # read file
    hdu_lst = fits.open(fitsfile)
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
                    print(f"Found {len(good_hdu)} readable HDUs. Change 'hduindex' parameter to read a different HDU.")
                    print("Below are the readable HDUs and their corresponding 'hduindex' parameters:")
                    for i in range(len(good_hdu)):
                        print(f"[{i}] {good_hdu[i]}")
            hduindex = int(hduindex) if not isinstance(hduindex, int) else hduindex
            data = good_hdu[hduindex].data
            hdu_header = dict(good_hdu[hduindex].header)
    
    # start reading and exporting header
    ctype = [hdu_header.get(f"CTYPE{i}", "") for i in range(1, data.ndim+1)] # store ctypes as list, 1-based indexing
    
    # get rest frequency
    if "RESTFRQ" in hdu_header:
        restfreq = hdu_header["RESTFRQ"]
    elif "RESTFREQ" in hdu_header:
        restfreq = hdu_header["RESTFREQ"]
    elif "FREQ" in hdu_header:
        restfreq = hdu_header["FREQ"]
    else:
        raise Exception("Failed to read rest frequnecy.")
    
    # stokes axis
    nstokes = hdu_header.get(f"NAXIS{ctype.index('STOKES')+1}") if "STOKES" in ctype else 1

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
        vunit = hdu_header.get(f"CUNIT{freq_axis_num}", "km/s" if is_vel else "Hz")
        dv = hdu_header.get(f"CDELT{freq_axis_num}")
        startv = hdu_header.get(f"CRVAL{freq_axis_num}")
        endv = startv+dv*(nchan-1)
        if u.Unit(vunit) != u.Unit(specunit):
            equiv = u.doppler_radio(restfreq*u.Hz)
            startv = u.Quantity(startv, vunit).to_value(specunit, equivalencies=equiv)
            endv = u.Quantity(endv, vunit).to_value(specunit, equivalencies=equiv)
            dv = (endv-startv)/(nchan-1)
            if specunit == "km/s":
                rounded_startv = round(startv, 5)
                if np.isclose(startv, rounded_startv):
                    start = rounded_startv
                rounded_endv = round(endv, 5)
                if np.isclose(endv, rounded_endv):
                    endv = rounded_endv
                rounded_dv = round(dv, 5)
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
        refnx = hdu_header.get(f"CRPIX{xaxis_num}")
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
        refnx = hdu_header.get(f"CRPIX{xaxis_num}")
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
        refny = hdu_header.get(f"CRPIX{yaxis_num}")
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
    refcoord = None     # initialize
    if refx is not None and refy is not None:
        refcoord = SkyCoord(ra=u.Quantity(refx, spatialunit), dec=u.Quantity(refy, spatialunit)).to_string('hmsdms')
    
    # determine image type and reshape data
    if nx > 1 and ny > 1 and nchan > 1:
        imagetype = "datacube"
        newshape = (nstokes, nchan, nx, ny)
        data = data.reshape(newshape)
    elif nx > 1 and ny > 1 and nchan == 1:
        imagetype = "spatialmap"
        newshape = (nstokes, nchan, nx, ny)
        data = data.reshape(newshape)
    elif nchan > 1 and nx > 1 and ny == 1:
        imagetype = "pvdiagram"
        newshape = (nstokes, nchan, nx)
        if data.shape == (nstokes, nx, nchan):
            data = data[0].T[None, :, :]  # transpose if necessary
    else:
        raise Exception("Image cannot be read as 'datacube', 'spatialmap', or 'pvdiagram'")
    
    # beam size
    bmaj = hdu_header.get("BMAJ", np.nan)
    bmin = hdu_header.get("BMIN", np.nan)
    bpa =  hdu_header.get("BPA", np.nan)  # deg
    
    if spatialunit != "deg":  # convert beam size unit if necessary
        if not np.isnan(bmaj):
            bmaj = u.Quantity(bmaj, u.deg).to_value(spatialunit)
        if not np.isnan(bmin):
            bmin = u.Quantity(bmin, u.deg).to_value(spatialunit)
    
    # eliminate rounding error due to float64
    if dx is not None:
        dx = float(str(np.float32(dx)))
    if dy is not None:
        dy = float(str(np.float32(dy)))
    if not np.isnan(bmaj):
        bmaj = float(str(np.float32(bmaj)))
    if not np.isnan(bmin):
        bmin = float(str(np.float32(bmin)))
    if not np.isnan(bpa):
        bpa = float(str(np.float32(bpa)))
            
    # input information into dictionary as header information of image
    fileinfo = {"name": fitsfile,
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
                "beam": (bmaj, bmin, bpa),
                "specframe": hdu_header.get("RADESYS", ""),
                "unit": spatialunit,
                "specunit": _apu_to_headerstr(u.Unit(specunit)),
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
        return Datacube(fileinfo=fileinfo, data=data)
    elif imagetype == "spatialmap":
        return Spatialmap(fileinfo=fileinfo, data=data)
    elif imagetype == "pvdiagram":
        return PVdiagram(fileinfo=fileinfo, data=data)


def _get_hduheader(image):
    """
    This is a private function that reads metadata from the header of the original fits file 
    and modifies it to reflect the current status of the given image.
    Parameters:
        image (Datacube/Spatialmap/PVdiagram): the input image from which the header will be extracted
    Returns:
        header (astropy.io.fits.header.Header): the extracted header information.
    """
    # get fitsfile from name 
    fitsfile = fits.open(image.header["name"])[0]
    hdu_header = copy.deepcopy(fitsfile.header)
    dx = u.Quantity(image.dx, image.unit).to_value(u.deg)
    
    # get info from current image object
    if image.header["dy"] is not None:
        dy = u.Quantity(image.dy, image.unit).to_value(u.deg)
    if image.refcoord is not None:
        center = _icrs2relative(image.refcoord, unit="deg")
        centerx, centery = center[0].value, center[1].value
    else:
        centerx, centery = None, None
    projection = image.header["projection"]
    dtnow = dt.datetime.now()
    
    # start reading header information
    if image.header["imagetype"] == "pvdiagram":
        faxis = image.get_vaxis(specunit='Hz')
        if hdu_header["NAXIS"] == 4:
            hdu_header["NAXIS"] = 3
            hdu_header.pop("NAXIS4", None)  # adding 'None' as a parameter prevents key error
            hdu_header.pop("CTYPE4", None)
            hdu_header.pop("CRVAL4", None)
            hdu_header.pop("CDELT4", None)
            hdu_header.pop("CRPIX4", None)
            hdu_header.pop("CUNIT4", None)
            hdu_header.pop("PC4_1", None)
            hdu_header.pop("PC4_2", None)
            hdu_header.pop("PC4_3", None)
            hdu_header.pop("PC4_4", None)
        hdu_header["NAXIS1"] = image.nx
        hdu_header["NAXIS2"] = image.nchan
        hdu_header["NAXIS3"] = 1
        hdu_header["CTYPE1"] = "OFFSET"
        hdu_header["CRVAL1"] = np.float64(0.)
        hdu_header["CDELT1"] = np.float64(image.dx)
        hdu_header["CRPIX1"] = np.float64(image.refnx)
        hdu_header["CUNIT1"] = image.unit
        hdu_header["CTYPE2"] = 'FREQ'
        startf = faxis[0]
        if not ("CRVAL2" in hdu_header and np.isclose(startf, hdu_header["CRVAL2"])):
            hdu_header["CRVAL2"] = startf
        df = faxis[1] - faxis[0]
        if not ("CDELT2" in hdu_header and np.isclose(df, hdu_header["CDELT2"])):
            hdu_header["CDELT2"] = df
        hdu_header["CRPIX2"] = np.float64(1.)
        hdu_header["CUNIT2"] = 'Hz'
        hdu_header["ALTRVAL"] = np.float64(image.vaxis[0])  # Alternative frequency referencenpoint
        hdu_header["ALTRPIX"] = np.float64(1.)              # Alternative frequnecy reference pixel
        hdu_header["CTYPE3"] = 'STOKES'
        hdu_header["CRVAL3"] = np.float64(1.)
        hdu_header["CDELT3"] = np.float64(1.)
        hdu_header["CRPIX3"] = np.float64(1.)
        hdu_header["CDELT3"] = ''

    elif image.header["imagetype"] in ("spatialmap", "datacube"):
        hdu_header["NAXIS"] = 4
        hdu_header["NAXIS1"] = image.shape[2]
        hdu_header["NAXIS2"] = image.shape[3]
        hdu_header["NAXIS3"] = image.shape[1]
        hdu_header["NAXIS4"] = image.shape[0]
        hdu_header["CTYPE1"] = f'RA---{projection}'
        hdu_header["CRVAL1"] = np.float64(centerx)
        hdu_header["CDELT1"] = np.float64(dx)
        hdu_header["CRPIX1"] = np.float64(image.refnx)
        hdu_header["CUNIT1"] = 'deg'
        hdu_header["CTYPE2"] = f'DEC--{projection}'
        hdu_header["CRVAL2"] = np.float64(centery)
        hdu_header["CDELT2"] = np.float64(dy)
        hdu_header["CRPIX2"] = np.float64(image.refny)
        hdu_header["CUNIT2"] = 'deg'
        if image.header["imagetype"] == "datacube":
            faxis = image.get_vaxis(specunit="Hz")
            hdu_header["CTYPE3"] = 'FREQ'
            startf = faxis[0]
            if not ("CRVAL3" in hdu_header and np.isclose(startf, hdu_header["CRVAL3"])):
                hdu_header["CRVAL3"] = startf
            df = faxis[1] - faxis[0]
            if not ("CDELT3" in hdu_header and np.isclose(df, hdu_header["CDELT3"])):
                hdu_header["CDELT3"] = df
            hdu_header["CRPIX3"] = np.float64(1.)
            hdu_header["CUNIT3"] = 'Hz'
            hdu_header["ALTRVAL"] = np.float64(image.vaxis[0])  # Alternative frequency reference point
            hdu_header["ALTRPIX"] = np.float64(1.)              # Alternative frequnecy reference pixel
        hdu_header["CTYPE4"] = 'STOKES'
        hdu_header["CRVAL4"] = np.float64(1.)
        hdu_header["CDELT4"] = np.float64(1.)
        hdu_header["CRPIX4"] = np.float64(1.)
        hdu_header["CUNIT4"] = ''

    # other information    
    if np.isnan(image.bmaj) or np.isnan(image.bmin) or np.isnan(image.bpa):
        updatedparams = {"BUNIT": image.header["bunit"],
                         "DATE": "%04d-%02d-%02d"%(dtnow.year, dtnow.month, dtnow.day),
                         "DATAMAX": np.float64(np.nanmax(image.data)),
                         "DATAMIN": np.float64(np.nanmin(image.data)),
                         "BSCALE": np.float64(1.),
                         "BZERO": np.float64(1.),
                         "OBJECT": image.header["object"],
                         "INSTRUME": image.header["instrument"],
                         "DATE-OBS": image.header["obsdate"],
                         "RESTFRQ": image.header["restfreq"],
                         "HISTORY": "Exported from astroviz."
                         }
    else:
        bmaj = np.float64(u.Quantity(image.bmaj, image.unit).to_value(u.deg))   # this value can be NaN
        bmin = np.float64(u.Quantity(image.bmin, image.unit).to_value(u.deg))
        bpa = np.float64(image.bpa)
        updatedparams = {"BUNIT": image.header["bunit"],
                         "DATE": "%04d-%02d-%02d"%(dtnow.year, dtnow.month, dtnow.day),
                         "BMAJ": bmaj,
                         "BMIN": bmin,
                         "BPA": bpa, 
                         "DATAMAX": np.float64(np.nanmax(image.data)),
                         "DATAMIN": np.float64(np.nanmin(image.data)),
                         "OBJECT": image.header["object"],
                         "BSCALE": np.float64(1.),
                         "BZERO": np.float64(1.),
                         "INSTRUME": image.header["instrument"],
                         "DATE-OBS": image.header["obsdate"],
                         "RESTFRQ": image.header["restfreq"],
                         "HISTORY": "Exported from astroviz."
                         }
    
    # don't write history if already in FITS file header
    if "HISTORY" in hdu_header and hdu_header["HISTORY"] == "Exported from astroviz.":
        updatedparams.pop("HISTORY")
    
    # start updating other header info
    for key, value in updatedparams.items():
        hdu_header[key] = value
        
    # return header
    return hdu_header


def _icrs2relative(coord, ref=None, unit="arcsec"):
    """
    This is a private function to convert the absolute coordinates to relative coordinates.
    Parameters:
        coord (tuple/str): either the J2000 coordinates as string in hmsdms format 
                           or the absolute coordinates converted to an angular unit as tuple
        ref (tuple/str): the reference coordinate. Same properties as 'coord.'
        unit (str): the unit of the relative coordinate that will be returned.
    Returns:
        (ra, dec): the right ascension and declination of the converted relative coordinates
                   as tuple.
    """
    if isinstance(coord, str):
        skycoord = SkyCoord(coord, unit=(u.hourangle, u.deg), frame="icrs")
    elif len(coord) == 2 and isinstance(coord[0], str) and isinstance(coord[1], str):
        skycoord = SkyCoord(coord[0], coord[1], unit=(u.hourangle, u.deg), frame="icrs")
    
    if ref is not None:
        if isinstance(ref, str):
            skycoord_ref = SkyCoord(ref, unit=(u.hourangle, u.deg), frame="icrs")
        elif len(ref) == 2 and isinstance(ref[0], str) and isinstance(ref[1], str):
            skycoord_ref = SkyCoord(ref[0], ref[1], unit=(u.hourangle, u.deg), frame="icrs")
        relative_ra = u.Quantity(skycoord.ra - skycoord_ref.ra, unit)
        relative_dec = u.Quantity(skycoord.dec - skycoord_ref.dec, unit)
        return (relative_ra, relative_dec)
    else:
        ra = u.Quantity(skycoord.ra, unit)
        dec = u.Quantity(skycoord.dec, unit)
        return (ra, dec)


def _relative2icrs(coord, ref=None, unit="arcsec"):
    """
    This is a private function to convert the relative coordinates to aboslute coordinates.
    Parameters:
        coord (tuple): the absolute coordinates converted to an angular unit as tuple
        ref (tuple/str): the reference coordinate. Same properties as 'coord,' but could be 
                         the J2000 coordinates as string in hmsdms format 
        unit (str): the unit of the relative coordinate that will be returned.
    Returns:
        str: in hmsdms format, the absolute coordinate.
    """
    if isinstance(coord, tuple):
        coord = list(coord)
    if ref is not None:
        if isinstance(ref, str):
            skycoord_ref = SkyCoord(ref, unit=(u.hourangle, u.deg), frame="icrs")
        elif len(ref) == 2 and isinstance(ref[0], str) and isinstance(ref[1], str):
            skycoord_ref = SkyCoord(ref[0], ref[1], unit=(u.hourangle, u.deg), frame="icrs")
        ref_ra = skycoord_ref.ra
        ref_dec = skycoord_ref.dec
    else:
        ref_ra = 0*u.arcsec
        ref_dec = 0*u.arcsec
    coord[0] = u.Quantity(coord[0], unit)
    coord[1] = u.Quantity(coord[1], unit)
    return SkyCoord(coord[0]+ref_ra, coord[1]+ref_dec, frame="icrs").to_string("hmsdms")


def _to_apu(unit_string):
    """
    This is private function that fixes a 'bug': astropy's incorrect reading of string units.
    For instance, strings like "Jy/beam.km/s" would be read as u.Jy/u.beam*u.km/u.s 
    by this function.
    """
    # if it is already a unit, return itself.
    if isinstance(unit_string, u.Unit):
        return unit_string
    elif isinstance(unit_string, u.Quantity):
        return unit_string.unit
    elif isinstance(unit_string, (u.core.CompositeUnit, u.core.PrefixUnit)):
        return unit_string
    
    # Split the unit string by '.' and '/'
    units_split = unit_string.replace('.', ' * ').replace('/', ' / ').split()

    # Initialize the unit
    result_unit = 1 * u.Unit(units_split[0])

    # Apply multiplication or division for the remaining units
    for i in range(1, len(units_split)):
        if units_split[i] in ['*', '/']:
            continue
        if units_split[i - 1] == '*':
            result_unit *= u.Unit(units_split[i])
        elif units_split[i - 1] == '/':
            result_unit /= u.Unit(units_split[i])

    return result_unit.unit


def _apu_to_str(unit):
    """
    This is private function that fixes a 'bug': astropy's incorrect conversion of units to strings.
    For instance, units like u.Jy/u.beam*u.km/u.s would be read as in the correct order in 
    the latex format by this function.
    """
    if unit.is_equivalent(u.Jy/u.beam*u.km/u.s) or \
       unit.is_equivalent(u.Jy/u.rad**2*u.km/u.s) or \
       unit.is_equivalent(u.Jy/u.pixel*u.km/u.s):
        unitstr = unit.to_string(format='latex_inline')
        unitlst = list(unitstr)
        idx_left = unitlst.index("{")
        units_lst = unitstr[idx_left+1:-2].split(",")
        newlst = units_lst[:]     # copy the list
        for i, ele in enumerate(units_lst):
            cleanunitstr = ele.replace("\\", "").replace("{", "").replace("}", "")
            if u.Unit(cleanunitstr).is_equivalent(u.Jy):
                newlst[0] = ele
            elif u.Unit(cleanunitstr).is_equivalent(1/u.beam) or \
                 u.Unit(cleanunitstr).is_equivalent(1/u.rad**2) or \
                 u.Unit(cleanunitstr).is_equivalent(1/u.pixel):
                newlst[1] = ele
            elif u.Unit(cleanunitstr).is_equivalent(u.km):
                newlst[2] = ele
            elif u.Unit(cleanunitstr).is_equivalent(1/u.s):
                newlst[3] = ele
        newstr = r"$\mathrm{" + ",".join(newlst) + r"}$"
        return newstr
    elif unit.is_equivalent(u.K*u.km/u.s):
        unitstr = unit.to_string(format='latex_inline')
        unitlst = list(unitstr)
        idx_left = unitlst.index("{")
        units_lst = unitstr[idx_left+1:-2].split(",")
        newlst = units_lst[:]     # copy the list
        for i, ele in enumerate(units_lst):
            cleanunitstr = ele.replace("\\", "").replace("{", "").replace("}", "")
            if u.Unit(cleanunitstr).is_equivalent(u.K):
                newlst[0] = ele
            elif u.Unit(cleanunitstr).is_equivalent(u.km):
                newlst[1] = ele
            elif u.Unit(cleanunitstr).is_equivalent(1/u.s):
                newlst[2] = ele
        newstr = r"$\mathrm{" + ",".join(newlst) + r"}$"
        return newstr
    else:
        return f"{unit:latex_inline}"


def _apu_to_headerstr(unit):
    """
    This is private function that converts an astropy unit object to the string 
    value corresponding to the 'bunit' key in the header.
    """
    unit_str = _apu_to_str(unit)
    unit_str = unit_str.replace("$", "").replace("\\mathrm", "")[1:-1]
    unit_lst = unit_str.split(r",")
    unit_lst = (item[:-1] if item[-1] == "\\" else item for item in unit_lst)  # create generator
    headerstr = ""
    for i, item in enumerate(unit_lst):
        if "^{" in item:
            ustr, power = item.split("^{")
            power = power[:-1]
        else:
            ustr = item
            power = 1
        if float(power) == -1:
            headerstr += "/" + ustr
        elif float(power) < -1:
            if int(float(power)) == float(power):
                headerstr += "/" + ustr + str(-int(power))
            else:
                headerstr += "/" + ustr + str(-float(power))
        elif float(power) > 1:
            if i == 0:
                if int(float(power)) == float(power):
                    headerstr += ustr + str(int(power))
                else:
                    headerstr += ustr + str(float(power))
            else:
                if int(float(power)) == float(power):
                    headerstr += "." + ustr + str(int(power))
                else:
                    headerstr += "." + ustr + str(float(power))
        else:
            if i == 0:
                headerstr += ustr
            else:
                headerstr += "." + ustr
    if headerstr.startswith("/"):
        headerstr = "1" + headerstr
    return headerstr


def _unit_plt_str(mathstr):
    """
    Private function that converts latex inline string to string for plotting 
    (fixes font issue).
    """
    mathstr = mathstr.replace("math", "")
    math_lst = mathstr[5:-2].split(r"\,")
    for i, unit in enumerate(math_lst[:]):
        math_lst[i] = unit.replace("^{", "$^{").replace("}", "}$")
    mathstr = " ".join(math_lst)
    return mathstr


def _convert_Bunit(quantity, newunit, equivalencies, factors, max_depth=10):
    """
    Private function to convert brightness units.
    Used by the 'conv_bunit' method.
    """
    # helper function to recursively utilize equivalencies
    def recursive_equiv(qt, eq, i):
        if i >= len(eq):  # base case
            return None
        try:
            return qt.to(newunit, equivalencies=eq[i])
        except UnitConversionError:
            return recursive_equiv(qt, eq, i+1)
    
    # helper function to recursively multiply/divide factors
    def recursive_factors(qt, f, i):
        if i >= len(factors)*2:  # base case
            return None
        try:
            if i % 2 == 0:
                return (qt / f[i//2]).to(newunit)
            else:
                return (qt * f[i//2]).to(newunit)
        except UnitConversionError:
            return recursive_factors(qt, f, i+1)
        
    try:
        # try direct conversion
        newqt = quantity.to(newunit)
    except UnitConversionError:
        # if that doesn't work, maybe try using equivalencies?
        newqt = recursive_equiv(quantity, equivalencies, i=0)
        j = 1
        new_factors = factors
        while newqt is None:
            # if that still doesn't work, maybe try multiplying/dividing by factors?
            newqt = recursive_factors(quantity, new_factors, i=0)
            j += 1
            new_factors = [factor**j for factor in factors]
            if j >= max_depth:  # maximum depth
                break
    return newqt


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
        _fontfamily = input("mathtext.fontset: ")
        _fontfamily = input("mathtext.tt: ")


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
    def __init__(self, fitsfile=None, fileinfo=None, data=None, hduindex=0, 
                 spatialunit="arcsec", specunit="km/s", quiet=False):
        if fitsfile is not None:
            fits = importfits(fitsfile, hduindex=hduindex, spatialunit=spatialunit, 
                              specunit=specunit, quiet=False)
            self.fileinfo = fits.fileinfo
            self.data = fits.data
        elif fileinfo is not None:
            self.fileinfo = fileinfo
            self.data = data
        if self.fileinfo["imagetype"] != "datacube":
            raise TypeError("The given FITS file cannot be read as a data cube.")
        self.__updateparams()
        
        if isinstance(self.data, u.quantity.Quantity):
            self.value = Datacube(fileinfo=self.fileinfo, data=self.data.value)
        
        self._peakshifted = False
        self.__pltnax = 0
        self.__pltnchan = 0
        
    def __updateparams(self):
        self.header = self.fileinfo
        self.spatialunit = self.unit = self.axisunit = self.fileinfo["unit"]
        nx = self.nx = self.fileinfo["nx"]
        dx = self.dx = self.fileinfo["dx"]
        refnx = self.refnx = self.fileinfo["refnx"]
        ny = self.ny = self.fileinfo["ny"]
        dy = self.dy = self.fileinfo["dy"]
        refny = self.refny = self.fileinfo["refny"]
        self.xaxis, self.yaxis = self.get_xyaxes()
        self.shape = self.fileinfo["shape"]
        self.size = self.data.size
        self.restfreq = self.fileinfo["restfreq"]
        self.bmaj, self.bmin, self.bpa = self.beam = self.fileinfo["beam"]
        self.resolution = np.sqrt(self.beam[0]*self.beam[1]) if self.beam is not None else None
        self.refcoord = self.fileinfo["refcoord"]
        if isinstance(self.data, u.Quantity):
            self.bunit = self.fileinfo["bunit"] = _apu_to_headerstr(self.data.unit)
        else:
            self.bunit = self.fileinfo["bunit"]
        xmin, xmax = self.xaxis[[0, -1]]
        ymin, ymax = self.yaxis[[0, -1]]
        self.imextent = [xmin-0.5*dx, xmax+0.5*dx, 
                         ymin-0.5*dy, ymax+0.5*dy]
        self.widestfov = max(self.xaxis[0], self.yaxis[-1])
        self.specunit = self.fileinfo["specunit"]
        if self.specunit == "km/s":
            rounded_dv = round(self.fileinfo["dv"], 5)
            if np.isclose(self.fileinfo["dv"], rounded_dv):
                self.dv = self.fileinfo["dv"] = rounded_dv
            else:
                self.dv = self.fileinfo["dv"]
            specmin, specmax = self.fileinfo["vrange"]
            rounded_specmin = round(specmin, 5)
            rounded_specmax = round(specmax, 5)
            if np.isclose(specmin, rounded_specmin):
                specmin = rounded_specmin
            if np.isclose(specmax, rounded_specmax):
                specmax = rounded_specmax
            self.vrange = self.fileinfo["vrange"] = [specmin, specmax]
        else:
            self.dv = self.fileinfo["dv"]
            self.vrange = self.fileinfo["vrange"]
        self.nv = self.nchan = self.fileinfo["nchan"]        
        self.vaxis = self.get_vaxis()
        
    # magic methods to define operators
    def __add__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                print("WARNING: operation performed on two images with different units.")
            return Datacube(fileinfo=self.fileinfo, data=self.data+other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data+other)
        
    def __sub__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                print("WARNING: operation performed on two images with different units.")
            return Datacube(fileinfo=self.fileinfo, data=self.data-other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data-other)
        
    def __mul__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data*other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data*other)
    
    def __pow__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data**other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data**other)
        
    def __rpow__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data**other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data**other)
        
    def __truediv__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data/other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data/other)
        
    def __floordiv__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data//other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data//other)
    
    def __mod__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data%other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data%other)
    
    def __lt__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data<other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data<other)
    
    def __le__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data<=other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data<=other)
    
    def __eq__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data==other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data==other)
        
    def __ne__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data!=other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data!=other)

    def __gt__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data>other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data>other)
        
    def __ge__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Datacube(fileinfo=self.fileinfo, data=self.data>=other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data>=other)

    def __abs__(self):
        return Datacube(fileinfo=self.fileinfo, data=np.abs(self.data))
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return Datacube(fileinfo=self.fileinfo, data=-self.data)
    
    def __round__(self, n):
        return Datacube(fileinfo=self.fileinfo, data=np.round(self.data, n))
    
    def round(self, decimals):
        """
        This is a method alias for the round function, round(Datacube).
        """
        return Datacube(fileinfo=self.fileinfo, data=np.round(self.data, decimals))
    
    def __invert__(self):
        return Datacube(fileinfo=self.fileinfo, data=~self.data)
    
    def __getitem__(self, indices):
        try:
            try:
                return Datacube(fileinfo=self.fileinfo, data=self.data[indices])
            except:
                print("WARNING: Returning value after reshaping image data to 3 dimensions.")
                return self.data.copy[0][indices]
        except:
            return self.data[indices]
    
    def __setitem__(self, indices, value):
        newdata = self.data.copy()
        newdata[indices] = value
        return Datacube(fileinfo=self.fileinfo, data=newdata)
    
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
            return Datacube(fileinfo=self.fileinfo, data=np.round(self.data, decimals))
        
        # Apply the numpy ufunc to the data
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Return a new Datacube instance with the result if the ufunc operation was successful
        if method == '__call__' and isinstance(result, np.ndarray):
            return Datacube(fileinfo=self.fileinfo, data=result)
        else:
            return result
        
    def __array__(self, *inputs, **kwargs):
        return np.array(self.data, *inputs, **kwargs)
    
    def to(self, unit, *args, **kwargs):
        """
        This method converts the intensity unit of original image to the specified unit.
        """
        return Datacube(fileinfo=self.fileinfo, data=self.data.to(unit, *args, **kwargs))
    
    def to_value(self, unit, *args, **kwargs):
        """
        Duplicate of astropy.unit.Quantity's 'to_value' method.
        """
        image = self.copy()
        image.data = image.data.to_value(unit, *args, **kwargs)
        image.bunit = image.header["bunit"] = _apu_to_headerstr(u.Unit(unit))
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
        vaxis = np.arange(self.vrange[0], self.vrange[-1]+self.dv, self.dv)
        if specunit is None:
            if u.Unit(self.specunit).is_equivalent(u.km/u.s):
                round_mask = np.isclose(vaxis, np.round(vaxis, 5))
                vaxis[round_mask] = np.round(vaxis, 5)
            return vaxis
        try:
            # attempt direct conversion
            vaxis = u.Quantity(vaxis, self.specunit).to_value(specunit)
        except UnitConversionError:
            # if that fails, try using equivalencies
            equiv = u.doppler_radio(self.restfreq*u.Hz)
            vaxis = u.Quantity(vaxis, self.specunit).to_value(specunit, equivalencies=equiv)
        if u.Unit(specunit).is_equivalent(u.km/u.s):
            round_mask = np.isclose(vaxis, np.round(vaxis, 5))
            vaxis[round_mask] = np.round(vaxis, 5)
            return vaxis
        return vaxis
    
    def conv_specunit(self, specunit, inplace=False):
        """
        Convert the spectral axis into the desired unit.
        """
        image = self if inplace else self.copy()
        vaxis = image.get_vaxis(specunit=specunit)
        image.fileinfo["dv"] = vaxis[1]-vaxis[0]
        image.fileinfo["vrange"] = vaxis[[0, -1]].tolist()
        image.fileinfo["specunit"] = _apu_to_headerstr(u.Unit(specunit))
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
        return Datacube(fileinfo=self.fileinfo, data=newdata.reshape(self.shape))
   
    def __get_momentdata(self, moment, data=None, vaxis=None):
        """
        Private method to get the data array of the specified moment map.
        Used for the public methods 'immoments' and 'peakshift' (moment 8 used).
        """
        # intialize parameters
        vaxis = self.vaxis if vaxis is None else vaxis
        data = self.data if data is None else data
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
            reshaped_vaxis = vaxis.reshape((1, nv, 1, 1))
            vv = np.broadcast_to(reshaped_vaxis, (1, nv, self.nx, self.ny))
            momdata = np.nansum(data*vv, axis=1) / np.nansum(data, axis=1)
        
        # intensity weighted dispersion of the coordinate (unit: km/s)
        elif moment == 2:
            reshaped_vaxis = vaxis.reshape((1, nv, 1, 1))
            vv = np.broadcast_to(reshaped_vaxis, (1, nv, self.nx, self.ny))
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
        
        return momdata.reshape((1, 1, self.nx, self.ny))
    
    def immoments(self, moments=[0], vrange=None, chans=None, threshold=None):
        """
        Parameters:
            moments (list[int]): a list of moment maps to be outputted
            vrange (list[float]): a list of [minimum velocity, maximum velocity] in km/s. 
                                  Default is to use the entire velocity range.
            chans (list[float]): a list of [minimum channel, maximum channel] using 1-based indexing.
                                 Default is to use all channels.
            threshold (float): a threshold to be applied to data cube. None to not use a threshold.
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
        if threshold is not None:
            data = np.where(self.data<threshold, np.nan, self.data)
        else:
            data = self.data.copy()
        
        # get nx and ny 
        nx, ny = self.nx, self.ny
        
        # truncate channels
        if len(chans) == 2 or len(chans) == 4:
            indicies = np.array(chans)-1
            vrange = self.vaxis[indicies]
        if len(vrange) == 2:
            vmask = (vrange[0]<=self.vaxis)&(self.vaxis<=vrange[1])
            vaxis = self.vaxis[vmask]
            clean_data = data[:, vmask, :, :]
        elif len(vrange) == 4:
            vmask1 = (vrange[0]<=self.vaxis)&(self.vaxis<=vrange[1])
            vmask2 = (vrange[2]<=self.vaxis)&(self.vaxis<=vrange[3])
            vmask = vmask1 | vmask2
            vaxis = self.vaxis[vmask]
            clean_data = data[:, vmask, :, :]
        else:
            clean_data = data
            vaxis = self.vaxis
        
        # start adding output maps to a list
        maps = []
        for moment in moments:
            momdata = self.__get_momentdata(moment=moment, data=clean_data, vaxis=vaxis)
            if moment in (-1, 3, 5, 6, 7, 8, 10):
                bunit = self.bunit
            elif moment == 0:
                bunit = f"{self.bunit}.{self.specunit}"
            else:
                bunit = self.specunit
            # update header information
            newheader = copy.deepcopy(self.header)
            newheader["imagetype"] = "spatialmap"
            newheader["shape"] = momdata.shape
            newheader["vrange"] = None
            newheader["nchan"] = 1
            newheader["dv"] = None
            newheader["bunit"] = bunit
            maps.append(Spatialmap(fileinfo=newheader, data=momdata))
            
        # return output
        return maps[0] if len(maps) == 1 else maps
        
    def imview(self, contourmap=None, title=None, fov=None, ncol=5, nrow=None, cmap="inferno",
               figsize=(11.69, 8.27), center=None, vrange=None, nskip=1, vskip=None, 
               tickscale=None, tickcolor="w", txtcolor='w', crosson=True, crosslw=0.5, 
               crosscolor="w", crosssize=0.3, dpi=400, vmin=None, vmax=None, xlim=None,
               ylim=None, xlabel=None, ylabel=None, xlabelon=True, ylabelon=True, crms=None, 
               clevels=np.arange(3, 21, 3), ccolor="w", clw=0.5, vsys=None, fontsize=12, 
               decimals=2, vlabelon=True, cbarloc="right", cbarwidth="3%", cbarpad=0., 
               cbarlabel=None, cbarlabelon=True, addbeam=True, beamcolor="skyblue", 
               beamloc=(0.1225, 0.1225), nancolor="k", labelcolor="k", axiscolor="w", axeslw=0.8, 
               labelsize=10, tickwidth=1., ticksize=3., tickdirection="in", vlabelsize=12, 
               vlabelunit=False, cbaron=True, title_fontsize=14, grid=None, plot=True):
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
            ccolor (str): the color of the contour image.
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
            plot (bool): True to execute 'plt.show()'
            
        Returns:
            The image grid with the channel maps.
        """
        # initialize parameters:
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
        if vrange is None:
            vrange = [vaxis.min(), vaxis.max()]
        velmin, velmax = vrange
        if vskip is None:
            vskip = self.dv*nskip
        else:
            nskip = int(vskip/self.dv)
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
            
        # trim data along vaxis for plotting:
        vmask = (velmin <= vaxis) & (vaxis <= velmax)
        trimmed_data = self.data[:, vmask, :, :][:, ::nskip, :, :]
        trimmed_vaxis = vaxis[vmask][::nskip]
            
        # trim data along xyaxes for plotting:
        if xlim != [self.widestfov, -self.widestfov] \
           or ylim != [self.widestfov, -self.widestfov]:
            xmask = (xlim[1]<=self.xaxis) & (self.xaxis<=xlim[0])
            ymask = (ylim[0]<=self.yaxis) & (self.yaxis<=ylim[1])
            trimmed_data = trimmed_data[:, :, xmask, :]
            trimmed_data = trimmed_data[:, :, :, ymask]
        imextent = [xlim[0]-0.5*self.dx, xlim[1]+0.5*self.dx, 
                    ylim[0]-0.5*self.dy, ylim[1]+0.5*self.dy]
        
        # modify contour map to fit the same channels:
        if contourmap is not None:
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
                trimmed_cdata = contmap.data[:, :, cxmask, :]
                trimmed_cdata = trimmed_cdata[:, :, :, cymask]
            else:
                trimmed_cdata = contmap.data
            contextent = [xlim[0]-0.5*contmap.dx, xlim[1]+0.5*contmap.dx, 
                          ylim[0]-0.5*contmap.dy, ylim[1]+0.5*contmap.dy]
        
        # figure out the number of images per row/column to plot:
        nchan = trimmed_vaxis.size
        if nrow is None and ncol is not None:
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
        for i in range(nrow*ncol):
            ax = grid[i] # the axes object of this channel
            if i < nchan:
                thisvel = trimmed_vaxis[i]         # the velocity of this channel
                thisdata = trimmed_data[0, i]      # 2d data in this channel

                # color image
                imcolor = ax.imshow(thisdata, cmap=cmap, origin='lower', 
                                    extent=imextent, rasterized=True)
                # contour image
                if contourmap is not None:
                    if contmap_isdatacube:
                        thiscdata = trimmed_cdata[:, (cvaxis == thisvel), :, :][0, 0]  # the contour data of this channel
                    imcontour = ax.contour(thiscdata, colors=ccolor, origin='lower', 
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
                    vlabel = f"%.{decimals}f "%thisvel + _apu_to_str(u.Unit(self.specunit)) \
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
        if xlabelon or ylabelon or addbeam:
            bottomleft_ax = grid[(nrow-1)*ncol]
        if xlabelon:
            bottomleft_ax.set_xlabel(xlabel)
            bottomleft_ax.xaxis.label.set_color(labelcolor)
        if ylabelon:
            bottomleft_ax.set_ylabel(ylabel)
            bottomleft_ax.yaxis.label.set_color(labelcolor)
        
        # add beam
        if addbeam:
            bottomleft_ax = self.__addbeam(bottomleft_ax, xlim=xlim, ylim=ylim,
                                           beamcolor=beamcolor, beamloc=beamloc)

        # colorbar 
        if cbaron:
            cax  = grid.cbar_axes[0]
            cbar = plt.colorbar(imcolor, cax=cax)
            cax.toggle_label(True)
            cbar.ax.yaxis.set_tick_params(color=tickcolor)
            cbar.ax.spines["bottom"].set_color(axiscolor)  
            cbar.ax.spines["top"].set_color(axiscolor)
            cbar.ax.spines["left"].set_color(axiscolor)
            cbar.ax.spines["right"].set_color(axiscolor)
            if cbarlabelon:
                cbar.ax.set_ylabel(cbarlabel, color=labelcolor)
        
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
        
        # show image
        if plot:
            plt.show()
        return grid
    
    def __addbeam(self, ax, xlim, ylim, beamcolor, beamloc=(0.1225, 0.1225)):
        """
        This method adds an ellipse representing the beam size to the specified ax.
        """
        # get axes limits
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        
        # coordinate of ellipse center 
        centerx = xlim[0] + beamloc[0]*xrange
        centery = ylim[0] + beamloc[1]*yrange
        coords = (centerx, centery)
        
        # beam size
        bmaj, bmin = self.bmaj, self.bmin
        
        # add patch to ax
        beam = patches.Ellipse(xy=coords, width=bmin, height=bmaj, fc=beamcolor,
                               angle=-self.bpa, alpha=1, zorder=10)
        ax.add_patch(beam)
        return ax
        
    def trim(self, vrange, nskip=1, vskip=None, inplace=False):
        """
        This method trims the data cube along the velocity axis.
        Parameters:
            vrange (iterable): the [minimum velocity, maximum velocity], inclusively.
            nskip (int): the number of channel increments
            vskip (float): the velocity increment between channels in km/s. An alternative to 'nskip' parameter.
            inplace (True): True to modify the data cube in-place. False to return a new data cube.
        Returns:
            The trimmed data cube.
        """
        # get values
        velmin, velmax = vrange
        vaxis = self.vaxis
        
        # convert vskip to nskip, if necessary
        if vskip is not None:
            nskip = int(vskip/self.dv)
        
        # create mask and trim data
        vmask = (velmin <= vaxis) & (vaxis <= velmax)
        trimmed_data = self.data[:, vmask, :, :][:, ::nskip, :, :]
        trimmed_vaxis = vaxis[vmask][::nskip]
        
        # update header information
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["dv"] = nskip*self.dv
        newfileinfo["vrange"] = [trimmed_vaxis.min(), trimmed_vaxis.max()]
        newfileinfo["nchan"] = int(np.round((velmax-velmin)/self.dv + 1))
        
        # return trimmed image or modify image in-place
        if inplace:
            self.data = trimmed_data
            self.fileinfo = newfileinfo
            self.__updateparams()
            return self
        return Datacube(fileinfo=newfileinfo, data=trimmed_data)
        
    def conv_bunit(self, bunit, inplace=False):
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
            raise UnitConversionError(f"Failure to convert intensity unit to {bunit.to_string()}")

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
        new_fileinfo = copy.deepcopy(self.fileinfo)
        new_fileinfo["dv"] = None
        new_fileinfo["nchan"] = 1
        new_fileinfo["shape"] = (1, 1, self.nx, self.ny)
        new_fileinfo["imagetype"] = "spatialmap"
        
        # start extracting and adding maps
        for i in range(chans[0], chans[1]+1, 1):
            new_fileinfo["vrange"] = [vaxis[i], vaxis[i]]
            maps.append(Spatialmap(fileinfo=copy.deepcopy(new_fileinfo), 
                                   data=self.data[0, i].reshape(1, 1, self.nx, self.ny)))
            
        # return maps as list or as inidividual object.
        if len(maps) == 1:
            return maps[0]
        return maps
    
    def set_threshold(self, threshold=None, minimum=True, inplace=False):
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
            return Datacube(fileinfo=self.fileinfo, data=np.where(self.data<threshold, np.nan, self.data))
        return Datacube(fileinfo=self.fileinfo, data=np.where(self.data<threshold, self.data, np.nan))
    
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
        To calculate the beam area of the image.
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
    
    def conv_unit(self, unit, distance=None, inplace=False):
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
            self.header["unit"] = u.Unit(unit).to_string()
            self.__updateparams()
            return self
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
        newfileinfo["dy"] = u.Quantity(self.dy, self.unit).to_value(unit)
        newfileinfo["beam"] = newbeam
        newfileinfo["unit"] = u.Unit(unit).to_string()
        return Datacube(fileinfo=newfileinfo, data=self.data)
    
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
            q = u.Unit(self.bunit)
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
    
    def mask_region(self, region, vrange=[], exclude=False, preview=True, inplace=False, **kwargs):
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
                                    (self.nchan, self.nx, self.ny)) 
            mask = mask | ~vmask
        masked_data = np.where(mask, data[0], np.nan)
        newshape =  (1, masked_data.shape[0],  masked_data.shape[1], masked_data.shape[2])
        masked_data = masked_data.reshape(newshape)
        masked_image = Datacube(fileinfo=self.fileinfo, data=masked_data)
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
        ymask = (-length/2 <= self.yaxis) & (self.yaxis <= length/2)
        vmask = (vrange[0] <= self.vaxis) & (self.vaxis <= vrange[1])
        pv_xaxis = self.yaxis[ymask]
        pv_vaxis = self.vaxis[vmask]
        pv_data = rotated_img.data[:, :, :, ymask][:, vmask, :, :]
        if width == 1:
            xmask = (self.xaxis == np.abs(self.xaxis).min())
            pv_data = pv_data[:, :, xmask, :]
        else:
            add_idx = int(width//2)
            idx = np.where(self.xaxis == np.abs(self.xaxis).min())[0][0]
            xidx1, xidx2 = idx-add_idx, idx+add_idx+1  # plus 1 to avoid OBOB
            pv_data = pv_data[:, :, xidx1:xidx2, :]
            pv_data = np.nanmean(pv_data, axis=2)
        pv_data = np.fliplr(pv_data.reshape(pv_vaxis.size, pv_xaxis.size))
        pv_data = pv_data[None, :, :]

        # export as pv data
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["shape"] = pv_data.shape
        newfileinfo["imagetype"] = "pvdiagram"
        newfileinfo["vrange"] = [pv_vaxis.min(), pv_vaxis.max()]
        newfileinfo["dv"] = np.round(pv_vaxis[1] - pv_vaxis[0], 7)
        newfileinfo["nchan"] = pv_vaxis.size
        newfileinfo["dx"] = np.round(pv_xaxis[1] - pv_xaxis[0], 7)
        newfileinfo["nx"] = pv_xaxis.size
        newfileinfo["refnx"] = pv_xaxis.size//2 + 1
        newfileinfo["dy"] = None
        newfileinfo["ny"] = None
        newfileinfo["refny"] = None
        newfileinfo["refcoord"] = _relative2icrs(center, ref=self.refcoord, unit=self.unit)
        
        # new image
        pv = PVdiagram(fileinfo=newfileinfo, data=pv_data)
        pv.pa = pa
        
        if preview:
            pv.imview(**kwargs)
        return pv
    
    def get_spectrum(self, region=None, xlabel=None, ylabel=None, returndata=False, **kwargs):
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
        xlabel = r'Radio Velocity ($\rm km~s^{-1}$)' if xlabel is None else xlabel
        ylabel = f'Intensity ({(u.Unit(self.bunit)):latex_inline})' if ylabel is None else ylabel
        trimmed_image = self if region is None else self.mask_region(region, preview=False)
        
        # generate the data
        intensity = np.nanmean(trimmed_image.data, axis=(0, 2, 3))
        vaxis = self.vaxis.copy()     # copy to prevent modifying after returning.
        
        # plot the data
        ax = _plt_spectrum(vaxis, intensity, xlabel=xlabel, ylabel=ylabel, **kwargs)
        
        # return the objects
        if returndata:
            return vaxis, intensity
        return ax
        
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
        
        convolved_image = Datacube(data=newimage, fileinfo=new_header)
        
        if preview:
            convolved_image.imview(**kwargs)
        
        if inplace:
            return self
        return convolved_image
        
    def imregrid(self, template=None, dx=None, dy=None, imsize=[], 
                 interpolation="linear", unit="arcsec", inplace=False):
        """
        Regrid the datacube image to a new grid resolution.
        Parameters:
            template (Datacube): Optional. A template image to use for the new grid.
            dx (float): New grid resolution along the x-axis.
            dy (float): New grid resolution along the y-axis.
            imsize (list(float)): Optional. The new image size (number of pixels in x- and y-axes)
            inplace (bool): If True, modify the current image in-place. Otherwise, return a new image.
        Returns:
            Datacube: The regridded image.
        """
        # get parameters from template if a template is given:
        if template is not None:
            dx = template.dx  # note: this does not store the same memory address, so no need to copy.
            dy = template.dy
        
        # calculate dx from imsize if imsize is given:
        if len(imsize) == 2:
            dx = -(self.xaxis.max()-self.xaxis.min())/imsize[0]
            dy = (self.yaxis.max()-self.yaxis.min())/imsize[1]

        # if dx or dy is not provided, raise an error
        if dx is None or dy is None:
            raise ValueError("New grid resolution (dx and dy) must be provided " + \
                              "either directly, via imsize, or via a template.")
        
        # dx must be negative and dy must be positive
        dx = -dx if dx > 0. else dx
        dy = -dy if dy < 0. else dy

        # compute the new grid
        new_nx = int((self.xaxis[-1] - self.xaxis[0]) / dx)
        new_ny = int((self.yaxis[-1] - self.yaxis[0]) / dy)
        new_xaxis = np.linspace(self.xaxis[0], self.xaxis[-1], new_nx)
        new_yaxis = np.linspace(self.yaxis[0], self.yaxis[-1], new_ny)
        new_xx, new_yy = np.meshgrid(new_xaxis, new_yaxis)

        # interpolate the data to the new grid
        old_xx, old_yy = self.get_xyaxes(grid=True)
        mask = ~np.isnan(self.data[0])
        
        print("Regridding...")
        new_data = np.array([[griddata(np.array([old_xx[mask[i]].ravel(), old_yy[mask[i]].ravel()]).T, 
                                                 self.data[0, i][mask[i]].ravel(), (new_xx, new_yy), 
                                                 method=interpolation).reshape((new_nx, new_ny)) \
                                                 for i in range(self.nchan)]])

        # update header information
        new_header = copy.deepcopy(self.fileinfo)
        new_header['shape'] = new_data.shape
        new_header['refnx'] = new_nx/2 + 1
        new_header['refnx'] = new_ny/2 + 1
        new_header['dx'] = dx
        new_header['dy'] = dy
        new_header['nx'] = new_nx
        new_header['ny'] = new_ny

        # create new Spatialmap instance
        regridded_map = Datacube(fileinfo=new_header, data=new_data)

        # modify the current image in-place if requested
        if inplace:
            self.data = new_data
            self.fileinfo = new_header
            self.__updateparams()  # Update parameters based on new header
            return self
        return regridded_map
    
    def normalize(self, range=[0, 1], inplace=False):
        """
        Normalize the data of the data cube to a specified range.
        Parameters:
            range (list/tuple): The range [min, max] to which the data will be normalized. Default is [0, 1].
            inplace (bool): If True, modify the data in-place. 
                            If False, return a new Spatialmap instance. Default is False.
        Returns:
            The normalized image, either as a new instance or the same instance based on the 'inplace' parameter.
        """

        # Extracting min and max from the range
        min_val, max_val = range

        # Normalization formula: (data - min(data)) / (max(data) - min(data)) * (max_val - min_val) + min_val
        data_min, data_max = np.nanmin(self.data), np.nanmax(self.data)
        normalized_data = (self.data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val

        if inplace:
            self.data = normalized_data
            return self
        return Datacube(fileinfo=self.fileinfo, data=normalized_data)
        
    def imshift(self, coord, unit=None, printcc=True, inplace=False):
        """
        This method shifts the image to the desired coordinate
        Parameters:
            coord (str/tuple/list/None): the J2000 or relative coordinate
            unit (str): the unit of the coordinate if coord is not J2000
            gauss (bool): use gaussian fitting to shift peak position if available
            inplace (bool): True to modify the image in-place.
        Returns:
            Image after shifting
        """
        unit = self.unit if unit is None else unit
        if isinstance(coord, Spatialmap) or isinstance(coord, Datacube) or isinstance(coord, PVdiagram):
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
        pixelx = int(shiftx/dx)
        pixely = int(shifty/dy)
        image.data = np.nan_to_num(image.data)
        shift = np.array([self.refny-1, self.refnx-1])-[self.refny-pixely, self.refnx+pixelx]
        # list comprehension is somehow faster than specifying axis with the built-in parameter of ndimage.shift
        image.data[0] = np.array([ndimage.shift(image.data[0, i], shift=shift) for i in range(self.nchan)])
        image.__updateparams()
        return image
    
    def peakshift(self, inplace=False):
        """
        Shift the maximum value of the image to the center of the image.
        Parameter:
            inplace (bool): True to modify the current image in-place. False to return a new image.
        Returns:
            The shifted image.
        """
        mom8_data = self.__get_momentdata(8)    # maximum intensity of spectrum
        indices = np.unravel_index(np.nanargmax(mom8_data[0, 0]), mom8_data[0, 0].shape)
        xx, yy = self.get_xyaxes(grid=True, unit=None)
        coord = xx[indices], yy[indices]
        if self._peakshifted:
            print("The peak is already shifted to the center.")
            return self.copy()
        else:
            shifted_image = self.imshift(coord=coord, printcc=True, unit=self.unit, inplace=inplace)
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
            i = 1
            while os.path.exists(outname):
                if os.path.exists(outname[:-5] + f"({i})" + outname[-5:]):
                    i += 1
                else:
                    outname = outname[:-5] + f"({i})" + outname[-5:]
        
        # Write to a FITS file
        hdu = fits.PrimaryHDU(data=self.data, header=hdu_header)
        hdu.writeto(outname, overwrite=overwrite)
        print(f"File saved as '{outname}'.")


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
    def __init__(self, fitsfile=None, fileinfo=None, data=None, hduindex=0, 
                 spatialunit="arcsec", quiet=False):
        if fitsfile is not None:
            fits = importfits(fitsfile, hduindex=hduindex, spatialunit=spatialunit, quiet=False)
            self.fileinfo = fits.fileinfo
            self.data = fits.data
        elif fileinfo is not None:
            self.fileinfo = fileinfo
            self.data = data
        if self.fileinfo["imagetype"] != "spatialmap":
            raise TypeError("The given FITS file is not a spatial map.")
        self.__updateparams()
        self._peakshifted = False
        if isinstance(self.data, u.quantity.Quantity):
            self.value = Spatialmap(fileinfo=self.fileinfo, data=self.data.value)
    
    # magic methods to define operators
    def __add__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                print("WARNING: operation performed on two images with different units.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data+other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data+other)
        
    def __sub__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                print("WARNING: operation performed on two images with different units.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data-other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data-other)
        
    def __mul__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data*other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data*other)
    
    def __pow__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data**other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data**other)
        
    def __rpow__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data**other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data**other)
        
    def __truediv__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data/other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data/other)
        
    def __floordiv__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data//other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data//other)
    
    def __mod__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data%other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data%other)
    
    def __lt__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data<other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data<other)
    
    def __le__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data<=other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data<=other)
    
    def __eq__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data==other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data==other)
        
    def __ne__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data!=other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data!=other)

    def __gt__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data>other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data>other)
        
    def __ge__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return Spatialmap(fileinfo=self.fileinfo, data=self.data>=other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data>=other)

    def __abs__(self):
        return Spatialmap(fileinfo=self.fileinfo, data=np.abs(self.data))
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return Spatialmap(fileinfo=self.fileinfo, data=-self.data)
    
    def __round__(self, n):
        return Spatialmap(fileinfo=self.fileinfo, data=np.round(self.data, n))
    
    def round(self, decimals):
        """
        This is a method alias for the round function, round(Spatialmap).
        """
        return Spatialmap(fileinfo=self.fileinfo, data=np.round(self.data, decimals))
    
    def __invert__(self):
        return Spatialmap(fileinfo=self.fileinfo, data=~self.data)
    
    def __getitem__(self, indices):
        try:
            try:
                return Spatialmap(fileinfo=self.fileinfo, data=self.data[indices])
            except:
                print("WARNING: Returning value after reshaping image data to 2 dimensions.")
                return self.data.copy[0, 0][indices]
        except:
            return self.data[indices]
    
    def __setitem__(self, indices, value):
        newdata = self.data.copy()
        newdata[indices] = value
        return Spatialmap(fileinfo=self.fileinfo, data=newdata)
    
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
            return Spatialmap(fileinfo=self.fileinfo, data=np.round(self.data, decimals))
        
        # Apply the numpy ufunc to the data
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Return a new Spatialmap instance with the result if the ufunc operation was successful
        if method == '__call__' and isinstance(result, np.ndarray):
            return Spatialmap(fileinfo=self.fileinfo, data=result)
        else:
            return result
        
    def __array__(self, *inputs, **kwargs):
        return np.array(self.data, *inputs, **kwargs)
    
    def to(self, unit, *args, **kwargs):
        """
        This method converts the intensity unit of original image to the specified unit.
        """
        return Spatialmap(fileinfo=self.fileinfo, data=self.data.to(unit, *args, **kwargs))
    
    def to_value(self, unit, *args, **kwargs):
        """
        Duplicate of astropy.unit.Quantity's 'to_value' method.
        """
        image = self.copy()
        image.data = image.data.to_value(unit, *args, **kwargs)
        image.bunit = image.header["bunit"] = _apu_to_headerstr(u.Unit(unit))
        return image
    
    def copy(self):
        """
        This method creates a copy of the original image.
        """
        return copy.deepcopy(self)
        
    def __updateparams(self):
        # make axes
        self.header = self.fileinfo
        self.spatialunit = self.unit = self.axisunit = self.fileinfo["unit"]
        nx = self.nx = self.fileinfo["nx"]
        dx = self.dx = self.fileinfo["dx"]
        refnx = self.refnx = self.fileinfo["refnx"]
        ny = self.ny = self.fileinfo["ny"]
        dy = self.dy = self.fileinfo["dy"]
        refny = self.refny = self.fileinfo["refny"]
        self.xaxis, self.yaxis = self.get_xyaxes()
        self.shape = self.fileinfo["shape"]
        self.size = self.data.size
        self.restfreq = self.fileinfo["restfreq"]
        self.bmaj, self.bmin, self.bpa = self.beam = self.fileinfo["beam"]
        self.resolution = np.sqrt(self.beam[0]*self.beam[1]) if self.beam is not None else None
        self.refcoord = self.fileinfo["refcoord"]
        if isinstance(self.data, u.Quantity):
            self.bunit = self.fileinfo["bunit"] = _apu_to_headerstr(self.data.unit)
        else:
            self.bunit = self.fileinfo["bunit"]
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
        return Spatialmap(fileinfo=self.fileinfo, data=newdata.reshape(self.shape))
        
    def imshift(self, coord, unit=None, printcc=True, inplace=False, order=0):
        """
        This method shifts the image to the desired coordinate
        Parameters:
            coord (str/tuple/list/None): the J2000 or relative coordinate
            unit (str): the unit of the coordinate if coord is not J2000
            gauss (bool): use gaussian fitting to shift peak position if available
            inplace (bool): True to modify the image in-place.
        Returns:
            Image after shifting
        """
        unit = self.unit if unit is None else unit
        if isinstance(coord, Spatialmap) or isinstance(coord, Datacube) or isinstance(coord, PVdiagram):
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
        pixelx = int(shiftx/dx)
        pixely = int(shifty/dy)
        
        image.data[0, 0] = ndimage.shift(np.nan_to_num(self.data[0, 0]), 
                                    np.array([self.refny-1, self.refnx-1])-[self.refny-pixely, self.refnx+pixelx])        
        image.__updateparams()
        return image
    
    def peakshift(self, inplace=False):
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
            shifted_image = self.imshift(coord=coord, printcc=True, unit=self.unit, inplace=inplace)
            if inplace:
                self._peakshifted = True
            else:
                shifted_image._peakshifted = True
            print(f"Shifted to {coord} [{self.unit}]")
            return shifted_image
    
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
        masked_image = Spatialmap(fileinfo=self.fileinfo, data=masked_data)
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
    
    def imregrid(self, template=None, dx=None, dy=None, imsize=[], interpolation="linear", 
                 unit="arcsec", inplace=False):
        """
        Regrid the image to a new grid resolution.
        Parameters:
            template (Spatialmap): Optional. A template image to use for the new grid.
            dx (float): New grid resolution along the x-axis.
            dy (float): New grid resolution along the y-axis.
            imsize (list(float)): Optional. The new image size (number of pixels in x- and y-axes)
            inplace (bool): If True, modify the current image in-place. Otherwise, return a new image.
        Returns:
            Spatialmap: The regridded image.
        """
        if template is not None:
            dx = template.dx
            dy = template.dy
        
        if len(imsize) == 2:
            dx = -(self.xaxis.max() - self.xaxis.min())/imsize[0]
            dy = (self.yaxis.max() - self.yaxis.min())/imsize[1]

        # If dx or dy is not provided, raise an error
        if dx is None or dy is None:
            raise ValueError("New grid resolution (dx and dy) must be provided"
                              "either directly, via imsize, or via a template.")
            
        dx = -dx if dx > 0 else dx
        dy = -dy if dy < 0 else dy

        # Compute the new grid
        new_nx = int((self.xaxis[-1] - self.xaxis[0]) / dx)
        new_ny = int((self.yaxis[-1] - self.yaxis[0]) / dy)
        new_xaxis = np.linspace(self.xaxis[0], self.xaxis[-1], new_nx)
        new_yaxis = np.linspace(self.yaxis[0], self.yaxis[-1], new_ny)
        new_xx, new_yy = np.meshgrid(new_xaxis, new_yaxis)

        # Interpolate the data to the new grid
        mask = np.isnan(self.data[0, 0])
        old_xx, old_yy = self.get_xyaxes(grid=True)
        points = np.array([old_xx[~mask].ravel(), old_yy[~mask].ravel()]).T
        values = self.data[0, 0][~mask].ravel()
        new_data = griddata(points, values, (new_xx, new_yy), method=interpolation)
        new_data = new_data.reshape((1, 1, new_data.shape[0], new_data.shape[1]))

        # Update header information
        new_header = copy.deepcopy(self.fileinfo)
        new_header['shape'] = new_data.shape
        new_header['refnx'] = new_nx/2 + 1
        new_header['refnx'] = new_ny/2 + 1
        new_header['dx'] = dx
        new_header['dy'] = dy
        new_header['nx'] = new_nx
        new_header['ny'] = new_ny

        # Create new Spatialmap instance
        regridded_map = Spatialmap(fileinfo=new_header, data=new_data)

        # Modify the current image in-place if requested
        if inplace:
            self.data = new_data
            self.fileinfo = new_header
            self.__updateparams()  # Update parameters based on new data
            return self
        return regridded_map
    
    def normalize(self, range=[0, 1], inplace=False):
        """
        Normalize the data of the spatial map to a specified range.
        Parameters:
            range (list/tuple): The range [min, max] to which the data will be normalized. Default is [0, 1].
            inplace (bool): If True, modify the data in-place. 
                            If False, return a new Spatialmap instance. Default is False.
        Returns:
            The normalized image, either as a new instance or the same instance based on the 'inplace' parameter.
        """

        # Extracting min and max from the range
        min_val, max_val = range

        # Normalization formula: (data - min(data)) / (max(data) - min(data)) * (max_val - min_val) + min_val
        data_min, data_max = np.nanmin(self.data), np.nanmax(self.data)
        normalized_data = (self.data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val

        if inplace:
            self.data = normalized_data
            return self
        return Spatialmap(fileinfo=self.fileinfo, data=normalized_data)
    
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
        convolved_image = Spatialmap(data=newimage.reshape((1, 1, self.nx, self.ny)), fileinfo=new_header)
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
        fitted_model = g(xx, yy).reshape(1, 1, self.nx, self.ny)     
        model_image = Spatialmap(fileinfo=self.fileinfo, data=fitted_model)
        
        # Print results
        coord = _relative2icrs(coord=(popt[0], popt[1]), ref=self.refcoord, unit=self.unit)
        coord = SkyCoord(coord, unit=(u.hourangle, u.deg), frame='icrs')
        center_J2000 = coord.ra.to_string(unit=u.hourangle, precision=3) + \
                        " " + coord.dec.to_string(unit=u.deg, precision=2)
        total_flux = np.nansum(fitted_model)
        print(15*"#"+"2D Gaussian Fitting Results"+15*"#")
        if not shiftcenter:
            print("center: (%.4f +/- %.4f, %.4f +/- %.4f) [arcsec]"%(popt[0], perr[0], popt[1], perr[1]))
        print("center (J2000): " + center_J2000)
        print("amplitude: %.4f +/- %.4f "%(popt[2], perr[2]) + f"[{(self.bunit).replace('.', ' ')}]")
        print("total flux: %.4f "%total_flux + f"[{(self.bunit).replace('.', ' ')}]")
        print("FWHM: %.4f +/- %.4f x %.4f +/- %.4f [arcsec]"%(popt[3], perr[3], popt[4], perr[4]))
        print("P.A.: %.4f +/- %.4f [deg]"%(popt[5], perr[5]))
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
        residual_image = Spatialmap(fileinfo=self.fileinfo, data=residual_data)
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
            return Spatialmap(fileinfo=self.fileinfo, data=np.where(self.data<threshold, np.nan, self.data))
        return Spatialmap(fileinfo=self.fileinfo, data=np.where(self.data<threshold, self.data, np.nan))
    
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
        To calculate the beam area of the image.
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
    
    def conv_unit(self, unit, inplace=False):
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
            self.header["unit"] = u.Unit(unit).to_string()
            self.__updateparams()
            return self
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
        newfileinfo["dy"] = u.Quantity(self.dy, self.unit).to_value(unit)
        newfileinfo["beam"] = newbeam
        newfileinfo["unit"] = u.Unit(unit).to_string()
        return Spatialmap(fileinfo=newfileinfo, data=self.data)
    
    def conv_bunit(self, bunit, inplace=False):
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
            raise UnitConversionError(f"Failure to convert intensity unit to {bunit.to_string()}")

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
        xlabel = f"Offset ({self.unit})" if xlabel is None else xlabel
        ylabel = f"Intensity ({_apu_to_str(_to_apu(self.bunit)):latex_inline})" if ylabel is None else ylabel
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
            q = u.Unit(self.bunit)
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
            
    def imview(self, contourmap=None, title=None, fov=None, vmin=None, vmax=None, 
               scale="linear", gamma=1.5, crms=None, clevels=np.arange(3, 21, 3), tickscale=None, 
               scalebaron=True, distance=None, cbarlabelon=True, cbarlabel=None, xlabelon=True,
               ylabelon=True, center=(0., 0.), dpi=500, ha="left", va="top", titleloc=(0.1, 0.9), 
               cmap=None, fontsize=12, cbarwidth="5%", width=330, height=300,
               smooth=None, scalebarsize=None, nancolor=None, beamcolor=None,
               ccolors=None, clw=0.8, txtcolor=None, cbaron=True, cbarpad=0., tickson=True, 
               labelcolor="k", tickcolor="k", labelsize=10., ticklabelsize=10., 
               cbartick_length=3., cbartick_width=1., beamon=True, scalebar_fontsize=10,
               axeslw=1., scalecolor=None, scalelw=1., cbarloc="right", 
               xlim=None, ylim=None, cbarticks=None, beamloc=(0.1225, 0.1225), vcenter=None, 
               vrange=None, aspect_ratio=1, barloc=(0.85, 0.15), barlabelloc=(0.85, 0.075),
               decimals=2, ax=None, plot=True):
        """
        Method to plot the 2D image.
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
            onsource (float): the onsource radius (useful to estimate crms if it was set to None)
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
            ticklabelsize (float): the font size of the tick labels
            cbartick_length (float): the length of the color bar ticks
            cbartick_width (float): the width of the color bar 
            beamon (bool): True to add an ellipse that represents the beam dimensions
            scalebar_fontsize (float): the font size of the scale bar label
            axeslw (float): the line width of the borders
            scalecolor (str): the color of the scale bar
            scalelw (float): the line width of the scale bar
            orientation (str): the orientation of the color bar
            plot (bool): True to show the plot
        Returns:
            The 2D image ax object.
        """
        # set parameters to default values
        if cmap is None:
            if u.Unit(self.bunit).is_equivalent(u.Hz) or \
               u.Unit(self.bunit).is_equivalent(u.km/u.s):
                cmap = 'RdBu_r'
            else:
                cmap = "inferno"
            
        if ccolors is None:
            ccolors = "w" if cmap == "inferno" else "k"
            
        if beamcolor is None:
            beamcolor = "skyblue" if cmap == "inferno" else "magenta"
            
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
        data = self.data.copy()
        if isinstance(data, u.Quantity):
            data = data.value.copy()
        
        # create a copy of the contour map
        if isinstance(contourmap, Spatialmap):
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
            
        # change default parameters
        ncols, nrows = 1, 1
        fig_width_pt  = width*ncols
        fig_height_pt = height*nrows
        inches_per_pt = 1.0/72.27                     # Convert pt to inch
        fig_width     = fig_width_pt * inches_per_pt  # width in inches
        fig_height    = fig_height_pt * inches_per_pt # height in inches
        fig_size      = [fig_width, fig_height]
        params = {'axes.labelsize': labelsize,
                  'axes.titlesize': fontsize,
                  'font.size' : fontsize,
                  'legend.fontsize': fontsize,
                  'xtick.labelsize': ticklabelsize,
                  'ytick.labelsize': ticklabelsize,
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
        
        # set colorbar
        my_cmap = copy.deepcopy(mpl.colormaps[cmap]) 
        my_cmap.set_bad(color=nancolor) 
        
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
            plt.subplots_adjust(wspace=0.4)
        
        if contourmap is not None and crms is None:
            try:
                crms = contourmap.noise()
                bunit = self.bunit.replace(".", " ")
                print(f"Estimated base contour level (rms): {crms:.4e} [{bunit}]")
            except Exception:
                contourmap = None
                print("Failed to estimate RMS noise level of contour map.")
                print("Please specify base contour level using 'crms' parameter.")
        
        # add image
        if scale.lower() == "linear":
            climage = ax.imshow(data[0, 0], cmap=my_cmap, extent=self.imextent, 
                                vmin=vmin, vmax=vmax, origin='lower')
        elif scale.lower() in ("log", "logscale", "logarithm"):
            if vmin is None:
                vmin = 3*self.noise()
            if vmax is None:
                vmax = np.nanmax(data)
                if vmax <= 0:
                    raise Exception("The data only contains negative values. Log scale is not suitable.")
            climage = ax.imshow(data[0, 0], cmap=my_cmap, extent=self.imextent,
                                norm=colors.LogNorm(vmin=vmin, vmax=vmax), 
                                origin="lower")
            if len(cbarticks) == 0:
                cbarticks = np.linspace(vmin, vmax, 7)[1:-1]
        elif scale.lower() == "gamma":
            climage = ax.imshow(data[0, 0], cmap=my_cmap, extent=self.imextent, origin="lower",
                                norm=colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))
        else:
            raise ValueError("Scale must be 'linear', 'log', or 'gamma'.")
            
        if cbaron:
            if cbarlabelon:
                if cbarlabel is None:
                    cbarlabel = "(" + _unit_plt_str(_apu_to_str(_to_apu(self.bunit))) + ")"
                else:
                    cbarlabel = ""
            
            # determine orientation from color bar location
            if cbarloc.lower() == "right":
                orientation = "vertical"
            elif cbarloc.lower() == "top":
                orientation = "horizontal"
            else:
                raise ValueError("'cbarloc' parameter must be eitehr 'right' or 'top'.")
                
            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes(cbarloc, size=cbarwidth, pad=cbarpad)
            cb = plt.colorbar(climage, cax=ax_cb, orientation=orientation, ticklocation=cbarloc)
            if len(cbarticks) > 0:
                cb.set_ticks(cbarticks)
                if scale.lower() in ("log", "logscale", "logarithm"):
                    labels = (f"%.{decimals}f"%label for label in cbarticks)  # generator object
                    labels = [label[:-1] if label.endswith(".") else label for label in labels]
                    cb.set_ticklabels(labels)
            cb.set_label(cbarlabel, fontsize=labelsize)
            cb.ax.tick_params(labelsize=ticklabelsize, width=cbartick_width, 
                              length=cbartick_length, direction='in')
        
        if contourmap is not None:
            contour_data = contourmap.data[0, 0]
            if smooth is not None:
                contour_data = gaussian_filter(contour_data, smooth)
            ax.contour(contour_data, extent=contourmap.imextent, 
                       levels=crms*clevels, colors=ccolors, linewidths=clw, origin='lower')
            
        if title is not None:
            xrange = xlim[1]-xlim[0]
            yrange = ylim[1]-ylim[0]
            titlex = xlim[0] + titleloc[0]*xrange
            titley = ylim[0] + titleloc[1]*yrange
            ax.text(x=titlex, y=titley, s=title, ha=ha, va=va, color=txtcolor, fontsize=fontsize)
        
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
                           colors=tickcolor, labelrotation=0, labelcolor=labelcolor)
        else:
            ax.tick_params(which='both', direction='in',bottom=False, top=False, left=False, right=False,
                           colors=tickcolor, labelrotation=0, labelcolor=labelcolor)
            
        # set labels
        if xlabelon:
            ax.set_xlabel(f"Relative RA ({self.unit})", fontsize=labelsize)
        if ylabelon:
            ax.set_ylabel(f"Relative Dec ({self.unit})", fontsize=labelsize)
        
        # add beam
        if beamon:
            ax = self.__addbeam(ax, xlim=xlim, ylim=ylim, beamcolor=beamcolor, 
                                beamloc=beamloc)
            
        # add scale bar
        if scalebaron and distance is not None:
            ax = self.__add_scalebar(ax, xlim=xlim, ylim=ylim, distance=distance, 
                                     size=scalebarsize, scalecolor=scalecolor, scalelw=scalelw, 
                                     fontsize=scalebar_fontsize, barloc=barloc, txtloc=barlabelloc)
            
        # aspect ratio
        ax.set_aspect(aspect_ratio)
        
        # reset field of view
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if plot:
            plt.show()
            
        return ax
    
    # need error checking.
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
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        
        barx = xlim[0] + xrange*barloc[0]
        bary = ylim[0] + yrange*barloc[1]
            
        textx = xlim[0] + xrange*txtloc[0]
        texty = ylim[0] + yrange*txtloc[1]
        
        label = str(int(dist_au))+' au'
        ax.text(textx, texty, label, ha='center', va='bottom', 
                color=scalecolor, fontsize=fontsize)
        ax.plot([barx-size/2, barx+size/2], [bary, bary], 
                color=scalecolor, lw=scalelw)
        
        return ax
        
    def __addbeam(self, ax, xlim, ylim, beamcolor, beamloc=(0.1225, 0.1225)):
        """
        This method adds an ellipse representing the beam size to the specified ax.
        """
        # get axes limits
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        
        # coordinate of ellipse center 
        centerx = xlim[0] + beamloc[0]*xrange
        centery = ylim[0] + beamloc[1]*yrange
        coords = (centerx, centery)
        
        # beam size
        bmaj, bmin = self.bmaj, self.bmin
        
        # add patch to ax
        beam = patches.Ellipse(xy=coords, width=bmin, height=bmaj, fc=beamcolor,
                               angle=-self.bpa, alpha=1, zorder=10)
        ax.add_patch(beam)
        return ax
    
    def line_info(self, **kwargs):
        """
        This method searches for the molecular line data from the Splatalogue database
        """
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
            i = 1
            while os.path.exists(outname):
                if os.path.exists(outname[:-5] + f"({i})" + outname[-5:]):
                    i += 1
                else:
                    outname = outname[:-5] + f"({i})" + outname[-5:]
        
        # Write to a FITS file
        hdu = fits.PrimaryHDU(data=self.data, header=hdu_header)
        hdu.writeto(outname, overwrite=overwrite)
        print(f"File saved as '{outname}'.")


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
    def __init__(self, fitsfile=None, fileinfo=None, data=None, hduindex=0, 
                 spatialunit="arcsec", specunit="km/s", quiet=False):
        if fitsfile is not None:
            fits = importfits(fitsfile, hduindex=hduindex, spatialunit=spatialunit, 
                              specunit=specunit, quiet=False)
            self.fileinfo = fits.fileinfo
            self.data = fits.data
        elif fileinfo is not None:
            self.fileinfo = fileinfo
            self.data = data
        if self.fileinfo["imagetype"] != "pvdiagram":
            raise TypeError("The given FITS file is not a PV diagram.")
        self.__updateparams()
        self.pa = None  # position angle at which the PV diagram was cut (to calculate offset resolution)
        
        if isinstance(self.data, u.quantity.Quantity):
            self.value = PVdiagram(fileinfo=self.fileinfo, data=self.data.value)
        
    def __updateparams(self):
        self.header = self.fileinfo
        self.spatialunit = self.unit = self.axisunit = self.fileinfo["unit"]
        self.nx = self.fileinfo["nx"]
        self.dx = self.fileinfo["dx"]
        refnx = self.refnx = self.fileinfo["refnx"]
        self.ny = self.fileinfo["ny"]
        self.xaxis = self.get_xaxis()
        self.shape = self.fileinfo["shape"]
        self.size = self.data.size
        self.restfreq = self.fileinfo["restfreq"]
        self.bmaj, self.bmin, self.bpa = self.beam = self.fileinfo["beam"]
        self.resolution = np.sqrt(self.beam[0]*self.beam[1]) if self.beam is not None else None
        self.refcoord = self.fileinfo["refcoord"]
        if isinstance(self.data, u.Quantity):
            self.bunit = self.fileinfo["bunit"] = _apu_to_headerstr(self.data.unit)
        else:
            self.bunit = self.fileinfo["bunit"]
        self.specunit = self.fileinfo["specunit"]
        self.vrange = self.fileinfo["vrange"]
        self.dv = self.fileinfo["dv"]
        self.nv = self.nchan = self.fileinfo["nchan"]
        if self.specunit == "km/s":
            rounded_dv = round(self.fileinfo["dv"], 5)
            vmin, vmax = self.fileinfo["vrange"]
            rounded_vmin = round(vmin, 5)
            rounded_vmax = round(vmax, 5)
            if np.isclose(rounded_dv, self.fileinfo["dv"]):
                self.dv = self.fileinfo["dv"] = rounded_dv
            if np.isclose(vmin, rounded_vmin):
                vmin = rounded_vmin
            if np.isclose(vmax, rounded_vmax):
                vmax = rounded_vmax
            self.vrange = self.fileinfo["vrange"] = [vmin, vmax]
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
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                print("WARNING: operation performed on two images with different units.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data+other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data+other)
        
    def __sub__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            if self.bunit != other.bunit:
                print("WARNING: operation performed on two images with different units.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data-other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data-other)
        
    def __mul__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data*other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data*other)
    
    def __pow__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data**other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data**other)
        
    def __rpow__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data**other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data**other)
        
    def __truediv__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data/other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data/other)
        
    def __floordiv__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data//other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data//other)
    
    def __mod__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data%other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data%other)
    
    def __lt__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data<other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data<other)
    
    def __le__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data<=other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data<=other)
    
    def __eq__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data==other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data==other)
        
    def __ne__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data!=other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data!=other)

    def __gt__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data>other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data>other)
        
    def __ge__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data>=other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data>=other)

    def __abs__(self):
        return PVdiagram(fileinfo=self.fileinfo, data=np.abs(self.data))
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return PVdiagram(fileinfo=self.fileinfo, data=-self.data)
    
    def __round__(self, n):
        return PVdiagram(fileinfo=self.fileinfo, data=np.round(self.data, n))
    
    def round(self, decimals):
        """
        This is a method alias for the round function, round(PVdiagram).
        """
        return PVdiagram(fileinfo=self.fileinfo, data=np.round(self.data, decimals))
    
    def __invert__(self):
        return PVdiagram(fileinfo=self.fileinfo, data=~self.data)
    
    def __getitem__(self, indices):
        try:
            try:
                return PVdiagram(fileinfo=self.fileinfo, data=self.data[indices])
            except:
                print("WARNING: Returning value after reshaping image data to 2 dimensions.")
                return self.data.copy[:, indices[0], indices[1]]
        except:
            return self.data[indices]
    
    def __setitem__(self, indices, value):
        newdata = self.data.copy()
        newdata[indices] = value
        return PVdiagram(fileinfo=self.fileinfo, data=newdata)
    
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
            return PVdiagram(fileinfo=self.fileinfo, data=np.round(self.data, decimals))
        
        # Apply the numpy ufunc to the data
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Return a new PVdiagram instance with the result if the ufunc operation was successful
        if method == '__call__' and isinstance(result, np.ndarray):
            return PVdiagram(fileinfo=self.fileinfo, data=result)
        else:
            return result
        
    def __array__(self, *inputs, **kwargs):
        return np.array(self.data, *inputs, **kwargs)
    
    def to(self, unit, *args, **kwargs):
        """
        This method converts the intensity unit of original image to the specified unit.
        """
        return PVdiagram(fileinfo=self.fileinfo, data=self.data.to(unit, *args, **kwargs))

    def to_value(self, unit, *args, **kwargs):
        """
        Duplicate of astropy.unit.Quantity's 'to_value' method.
        """
        image = self.copy()
        image.data = image.data.to_value(unit, *args, **kwargs)
        image.bunit = image.header["bunit"] = _apu_to_headerstr(u.Unit(unit))
        return image
        
    def copy(self):
        """
        This method converts the intensity unit of original image to the specified unit.
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
        vaxis = np.arange(self.vrange[0], self.vrange[-1]+self.dv, self.dv)
        if specunit is None:
            if u.Unit(self.specunit).is_equivalent(u.km/u.s):
                round_mask = np.isclose(vaxis, np.round(vaxis, 5))
                vaxis[round_mask] = np.round(vaxis, 5)
            return vaxis
        try:
            # attempt direct conversion
            vaxis = u.Quantity(vaxis, self.specunit).to_value(specunit)
        except UnitConversionError:
            # if that fails, try using equivalencies
            equiv = u.doppler_radio(self.restfreq*u.Hz)
            vaxis = u.Quantity(vaxis, self.specunit).to_value(specunit, equivalencies=equiv)
        if u.Unit(specunit).is_equivalent(u.km/u.s):
            round_mask = np.isclose(vaxis, np.round(vaxis, 5))
            vaxis[round_mask] = np.round(vaxis, 5)
            return vaxis
        return vaxis
    
    def conv_specunit(self, specunit, inplace=False):
        """
        Convert the spectral axis into the desired unit.
        """
        image = self if inplace else self.copy()
        vaxis = image.get_vaxis(specunit=specunit)
        image.fileinfo["dv"] = vaxis[1]-vaxis[0]
        image.fileinfo["vrange"] = vaxis[[0, -1]].tolist()
        image.fileinfo["specunit"] = _apu_to_headerstr(u.Unit(specunit))
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
    
    def set_threshold(self, threshold=None, minimum=True, inplace=False):
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
            return PVdiagram(fileinfo=self.fileinfo, data=np.where(self.data<threshold, np.nan, self.data))
        return PVdiagram(fileinfo=self.fileinfo, data=np.where(self.data<threshold, self.data, np.nan))
    
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
            q = u.Unit(self.bunit)
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

    def conv_unit(self, unit, inplace=False):
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
            self.header["unit"] = _apu_to_headerstr(u.Unit(unit))
            self.__updateparams()
            return self
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
        newfileinfo["beam"] = newbeam
        newfileinfo["unit"] = _apu_to_headerstr(u.Unit(unit).to_string())
        return PVdiagram(fileinfo=newfileinfo, data=self.data)
    
    def conv_bunit(self, bunit, inplace=False):
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
            raise UnitConversionError(f"Failed to convert intensity unit to {bunit.to_string()}")

        # return and set values
        if not isinstance(self.data, u.Quantity):
            newdata = newdata.value
            
        newimage = self if inplace else self.copy()
        newimage.data = newdata
        newimage.header["bunit"] = _apu_to_headerstr(bunit)
        newimage.__updateparams()
        return newimage
    
    def _mean_velocities_at_position(self):
        # intiialize variables
        vaxis = self.vaxis.copy()
        data = self.data[0]
        nv = self.nv
        nx = self.nx

        # broadcast vaxis
        vv = vaxis[:, None]
        vv = np.broadcast_to(vv, (nv, nx))
        
        # mean velocity at each position, weighted by intensity
        mean_vel = np.nansum(vv*data, axis=0)/np.nansum(data, axis=0)  # size: nx
        
        # calculate standard error
        N = np.count_nonzero(~np.isnan(data), axis=0)  # number of observations
        weighted_sqdev = data*(vv-mean_vel[None, :])**2  # weighted deviations squared
        weighted_var = np.nansum(weighted_sqdev, axis=0)/np.nansum(data, axis=0)  # weighted variance
        weighted_sd = np.sqrt(weighted_var)
        std_err = weighted_sd / np.sqrt(N-1)
        
        return mean_vel, std_err
           
    def _mean_positions_at_velocity(self):
        # intiialize variables
        xaxis = self.xaxis.copy()
        data = self.data[0]
        nv = self.nv
        nx = self.nx

        # broadcast vaxis
        xx = xaxis[None, :]
        xx = np.broadcast_to(xx, (nv, nx))
        
        # mean velocity at each position, weighted by intensity
        mean_offset = np.nansum(xx*data, axis=1)/np.nansum(data, axis=1)  # size: nx
        
        # calculate standard error
        N = np.count_nonzero(~np.isnan(data), axis=1)  # number of observations
        weighted_sqdev = data*(xx-mean_offset[:, None])**2  # weighted deviations squared
        weighted_var = np.nansum(weighted_sqdev, axis=1)/np.nansum(data, axis=1)  # weighted variance
        weighted_sd = np.sqrt(weighted_var)
        std_err = weighted_sd / np.sqrt(N-1)
        
        return mean_offset, std_err        
    
    def __get_offset_resolution(self, pa=None):
        if pa is None:
            return np.sqrt(self.bmaj*self.bmin)  # simply take average if pa is not specified
        angle = np.deg2rad(pa-self.bpa)
        aa = np.square(np.sin(angle)/self.bmin)
        bb = np.square(np.cos(angle)/self.bmaj)
        return np.sqrt(1/(aa+bb))
        
    def imview(self, contourmap=None, cmap="inferno", vmin=None, vmax=None, nancolor="k", crms=None, 
               clevels=np.arange(3, 21, 3), ccolor="w", clw=1., dpi=500, cbaron=True, cbarloc="right", 
               cbarpad="0%", vsys=None, xlim=None, vlim=None, xcenter=0., vlineon=True, xlineon=True, 
               cbarlabelon=True, cbarwidth='5%', cbarlabel=None, fontsize=12, labelsize=10, width=330, height=300, 
               plotres=True, xlabelon=True, vlabelon=True, xlabel=None, vlabel=None, offset_as_hor=False, 
               aspect_ratio=1.15, axeslw=1., tickson=True, tickwidth=1., tickdirection="in", ticksize=3., ticklabelsize=10,
               cbartick_width=1, cbartick_length=3, xticks=None, vticks=None, title=None, titleloc=(0.05, 0.985), 
               ha="left", va="top", txtcolor="w", refline_color="w", pa=None, refline_width=None, 
               subtract_vsys=False, errbarloc=(0.1, 0.1225), ax=None, plot=True):
        """
        Display a Position-Velocity (PV) diagram.

        This method generates and plots a PV diagram, offering several customization options 
        for visual aspects like colormap, contour levels, axis labels, etc.

        Parameters:
            contourmap (Datacube, optional): A Datacube instance to use for contour mapping. Default is None.
            cmap (str): Colormap for the image data. Default is 'inferno'.
            nancolor (str): Color used for NaN values in the data. Default is 'k' (black).
            crms (float, optional): RMS noise level for contour mapping. Automatically estimated if None and contourmap is provided.
            clevels (numpy.ndarray): Contour levels for the contour mapping. Default is np.arange(3, 21, 3).
            ccolor (str): Color of the contour lines. Default is 'w' (white).
            clw (float): Line width of the contour lines. Default is 1.0.
            dpi (int): Dots per inch for the plot. Affects the quality of the image. Default is 500.
            cbaron (bool): Flag to display the color bar. Default is True.
            cbarloc (str): Location of the color bar. Default is 'right'.
            cbarpad (str): Padding for the color bar. Default is '0%'.
            vsys (float, optional): Systemic velocity. If provided, adjusts velocity axis accordingly. Default is None.
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
            plot (bool): Flag to execute the plotting. Default is True.

        Returns:
            matplotlib.axes.Axes: The Axes object of the plot if 'plot' is True, allowing further customization.
        """
        # initialize parameters:
        if not isinstance(clevels, np.ndarray):
            clevels = np.array(clevels)
        if pa is None:
            pa = self.pa
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
        contmap = contourmap.copy() if contourmap is not None else contourmap
        xlim = self.maxxlim if xlim is None else xlim
        if vlim is None:
            if subtract_vsys:
                vlim = np.array(self.maxvlim)-vsys
            else:
                vlim = self.maxvlim
        if vlabel is None:
            if subtract_vsys:
                vlabel = r"$v_{\rm obs}-v_{\rm sys}$ " + "(" + _apu_to_str(u.Unit(self.specunit)) + ")"
            else:
                vlabel = "LSR velocity " + "(" + _unit_plt_str(_apu_to_str(u.Unit(self.specunit))) + ")"
        
        if xlabel is None:
            xlabel = f'Offset ({self.unit})'
        if cbarlabel is None:
            cbarlabel = "(" + _unit_plt_str(_apu_to_str(_to_apu(self.bunit))) + ")"
        if refline_width is None:
            refline_width = clw
        vres, xres = self.dv, self.__get_offset_resolution(pa=pa)
        cmap = copy.deepcopy(mpl.colormaps[cmap]) 
        cmap.set_bad(color=nancolor)
                
        # change default matplotlib parameters
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
        
        if ax is None:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(111)
        
        # plot image
        if offset_as_hor:
            imextent = [self.imextent[2], self.imextent[3], self.imextent[0], self.imextent[1]]
            if subtract_vsys:
                imextent[3] -= vsys
                imextent[4] -= vsys
            colordata = self.data[0, :, :]
            climage = ax.imshow(colordata, cmap=cmap, extent=imextent, origin='lower', vmin=vmin, vmax=vmax)
            if contmap is not None:
                contextent = [contmap.imextent[2], contmap.imextent[3], contmap.imextent[0], contmap.imextent[1]]
                contdata = contmap.data[0, :, :]
                ax.contour(contdata, colors=ccolor, origin='lower', extent=contextent, 
                           levels=crms*clevels, linewidths=clw)
            
            ax.set_xlabel(xlabel, fontsize=fontsize)
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
            imextent = self.imextent[:]
            if subtract_vsys:
                imextent[0] -= vsys
                imextent[1] -= vsys
            colordata = self.data[0, :, :].T
            climage = ax.imshow(colordata, cmap=cmap, extent=imextent, origin='lower', vmin=vmin, vmax=vmax)
            if contmap is not None:
                contextent = contmap.imextent
                contdata = contmap.data[0, :, :].T
                ax.contour(contdata, colors=ccolor, origin='lower', extent=contextent, 
                           levels=crms*clevels, linewidths=clw)
            ax.set_xlabel(vlabel, fontsize=fontsize)
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
            ax.tick_params(which='both', direction=tickdirection, bottom=True, top=True, left=True, right=True,
                           pad=9, labelsize=labelsize)
        else:
            ax.tick_params(which='both', direction=tickdirection, bottom=False, top=False, left=False, right=False,
                           pad=9, labelsize=labelsize)
        
        # define horizontal and vertical limits
        if offset_as_hor:
            horlim = xlim
            vertlim = vlim
        else:
            horlim = vlim
            vertlim = xlim
        
        # color bar
        if cbaron:
            # determine orientation based on color bar location
            if cbarloc.lower() == "right":
                orientation = "vertical"
                ax_cb = inset_axes(ax, width=cbarwidth, height='100%', loc='lower left',
                               bbox_to_anchor=(1.0 + float(cbarpad.strip('%'))*0.01, 0., 1., 1.),
                               bbox_transform=ax.transAxes, borderpad=0)
            elif cbarloc.lower() == "top":
                orientation = "horizontal"
                ax_cb = inset_axes(ax, width='100%', height=cbarwidth, loc='lower left',
                       bbox_to_anchor=(0., 1.0 + float(cbarpad.strip('%'))*0.01, 1., 1.),
                       bbox_transform=ax.transAxes, borderpad=0)
            else:
                raise ValueError("'cbarloc' parameter must be eitehr 'right' or 'top'.")
                
            cbar = fig.colorbar(climage, cax=ax_cb, pad=cbarpad, 
                                orientation=orientation, ticklocation=cbarloc.lower())
            cbar.set_label(cbarlabel)
            cbar.ax.tick_params(labelsize=ticklabelsize, width=cbartick_width, 
                                length=cbartick_length, direction="in")
            
         # set aspect ratio
        if aspect_ratio is not None:
            hor_range = horlim[1]-horlim[0]
            vert_range = vertlim[1]-vertlim[0]    
            real_ar = float(np.abs(1/aspect_ratio*hor_range/vert_range))
            ax.set_aspect(real_ar)
        
        # plot resolution
        if plotres:
            res_x, res_y = (xres, vres) if offset_as_hor else (vres, xres)
            res_x_plt, res_y_plt = ax.transLimits.transform((res_x*0.5, res_y*0.5))-ax.transLimits.transform((0, 0))
            ax.errorbar(errbarloc[0], errbarloc[1], xerr=res_x_plt, yerr=res_y_plt, color=ccolor, capsize=3, 
                        capthick=1., elinewidth=1., transform=ax.transAxes)
           
        # plot title, if necessary
        if title is not None:
            titlex = horlim[0] + titleloc[0]*hor_range
            titley = horlim[0] + titleloc[1]*vert_range
            ax.text(x=titlex, y=titley, s=title, ha=ha, va=va, 
                    color=txtcolor, fontsize=fontsize)
        
        if plot:
            plt.show()
        
        return ax
    
    def line_info(self, **kwargs):
        """
        This method searches for the molecular line data from the Splatalogue database
        """
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
            i = 1
            while os.path.exists(outname):
                if os.path.exists(outname[:-5] + f"({i})" + outname[-5:]):
                    i += 1
                else:
                    outname = outname[:-5] + f"({i})" + outname[-5:]
        
        # Write to a FITS file
        hdu = fits.PrimaryHDU(data=self.data, header=hdu_header)
        hdu.writeto(outname, overwrite=overwrite)
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
        # recursively finds file
        def find_file(file):  
            for root, dirs, files in os.walk(os.getcwd()):
                if file in files:
                    return os.path.join(root, file)
            return None 

        if not os.path.exists(self.regionfile):
            if not quiet:
                print(f"Given directory '{fitsfile}' does not exist as a relative directory. " + \
                       "Recursively finding file...")
            maybe_filename = find_file(self.regionfile)
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
            self.shape = filelst[2].split()[0]
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
        elif self.shape == "ellipse":
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
        elif self.shape == "box":
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
            raise Exception("Not implemented yet.")

    def __readCRTF(self):
        raise Exception("Not implemented yet.")


def plt_1ddata(xdata=None, ydata=None, xlim=[], ylim=[], mean_center=False, title=None,
               legendon=True, xlabel="", ylabel="", threshold=None, linewidth=0.8,
               xtick_spacing=None, ytick_spacing=None, ticklabelsize=7, borderwidth=0.7,
               labelsize=7, fontsize=8, ticksize=3, legendsize=6, title_position=0.92,
               bbox_to_anchor=(0.6, 0.95), legendloc="best", threshold_color="gray",
               linecolor="k", figsize=(2.76, 2.76), bins="auto", hist=False,
               dpi=500, plot=True, **kwargs):
    
    fontsize = labelsize if fontsize is None else fontsize
    # Get this from LaTeX using \showthe\columnwidth
    
    params = {'axes.labelsize': labelsize,
              'axes.titlesize': labelsize,
              'font.size': fontsize,
              'legend.fontsize': legendsize,
              'xtick.labelsize': ticklabelsize,
              'ytick.labelsize': ticklabelsize,
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


def _plt_spectrum(velocity=None, intensity=None, csvfile=None, xlim=[], ylim=[], 
                  gaussfit=True, mean_center=True, title=None, legendon=True,
                  xlabel=r"Velocity offset ($\rm km~s^{-1}$)", ylabel=r"Intensity ($\rm K$)", 
                  threshold=None, linewidth=0.8, figsize=(2.76, 2.76), xtick_spacing=None,
                  ytick_spacing=None, ticklabelsize=7, borderwidth=0.7, labelsize=7,
                  fontsize=8, ticksize=3,legendsize=6, title_position=0.92,
                  bbox_to_anchor=(0.6, 0.95), legendloc="best", skiprows=5,
                  delimiter="\t", vsys=0., subtract_vsys=False, dpi=500,
                  threshold_color="gray", **kwargs):
    
    """
    Plot and analyze spectral data with optional Gaussian fitting.

    Parameters:
        velocity (array-like, optional): Array of velocity values.
        intensity (array-like, optional): Corresponding intensity values.
        csvfile (str, optional): Path to a CSV file containing velocity and intensity data.
        xlim (list, optional): Limits for the x-axis (velocity).
        ylim (list, optional): Limits for the y-axis (intensity).
        gaussfit (bool, optional): If True, fits a Gaussian model to the data.
        mean_center (bool, optional): If True, centers the plot around the mean of the data.
        title (str, optional): Title for the plot.
        legendon (bool, optional): If True, displays a legend.
        xlabel/ylabel (str, optional): Labels for the x and y axes.
        threshold (float, optional): Threshold value for plotting and analysis.
        linewidth (float, optional): Line width for plotting.
        figsize (tuple, optional): Size of the figure.
        xtick_spacing/ytick_spacing (float, optional): Spacing for x and y ticks.
        ticklabelsize, borderwidth, labelsize, fontsize (float, optional): Various formatting parameters.
        ticksize, legendsize, title_position (float, optional): Additional formatting parameters.
        bbox_to_anchor, legendloc (tuple/str, optional): Legend positioning.
        skiprows (int, optional): Number of rows to skip when reading a CSV file.
        delimiter (str, optional): Delimiter for CSV file.
        vsys (float, optional): Systemic velocity.
        subtract_vsys (bool, optional): If True, subtracts systemic velocity from the velocity data.
        dpi (int, optional): Resolution of the figure.
        threshold_color (str, optional): Color for the threshold line.
    """
    
    def gauss(x, amp, mean, fwhm):
        sigma = fwhm / 2.355
        return amp * np.exp(-np.square(x-mean)/(2*sigma**2))

    def fit_gauss(xdata, ydata, xrange=None, threshold=None, p0=None, return_params=True):
        xlim = np.nanmin(xdata), np.nanmax(xdata)
        xdata = np.array(xdata) if not isinstance(xdata, np.ndarray) else xdata
        ydata = np.array(ydata) if not isinstance(ydata, np.ndarray) else ydata

        if xrange is not None and len(xrange) == 2:
            mask = (xdata>np.nanmin(xrange)) & (xdata<np.nanmax(xrange))
            xdata = xdata[mask]
            ydata = ydata[mask]

        if threshold is not None and (isinstance(threshold, float) or isinstance(threshold, int)):
            mask = ydata > threshold
            xdata = xdata[mask]
            ydata = ydata[mask]

        if p0 is None:
            init_amp = np.nanmax(ydata)
            init_mean = np.trapz(x=xdata, y=ydata*xdata)/np.trapz(x=xdata, y=ydata)
            init_fwhm = (np.nanmax(xdata) - np.nanmin(xdata))/2
            p0 = [init_amp, init_mean, init_fwhm]

        # Perform fitting
        popt, pcov = curve_fit(f=gauss, xdata=xdata, ydata=ydata, p0=p0)
        perr = np.sqrt(np.diag(pcov))

        # Get fitting results
        amp_opt, mean_opt, fwhm_opt = popt
        amp_err, mean_err, fwhm_err = perr

        print(10*"#"+"Fitting Results"+10*"#")
        print("Amplitude = %0.2f +/- %0.2f"%(amp_opt, amp_err))
        print("Mean = %0.2f +/- %0.2f"%(mean_opt, mean_err))
        print("FWHM = %0.2f +/- %0.2f"%(fwhm_opt, fwhm_err))
        print(35*"#")

        # get best-fit curve
        xsmooth = np.linspace(xlim[0], xlim[1], 10000)
        ysmooth = gauss(xsmooth, amp_opt, mean_opt, fwhm_opt)

        return (xsmooth, ysmooth, [popt, perr]) if return_params else (xsmooth, ysmooth)
    if velocity is None or intensity is None:
        if csvfile is None:
            raise Exception("Please specify the 'velocity' and 'intensity' parameters or 'csvfile'.")
        else:
            if subtract_vsys and vsys == 0:
                print("WARNING: vsys is set to 0. vobs = vobs-vsys.")
            dataset = pd.read_csv(csvfile, skiprows=skiprows, delimiter=delimiter)
            velocity = dataset["# x"]
            velocity = velocity - vsys if subtract_vsys else velocity
            xlabel=r"Radio velocity ($\rm km~s^{-1}$)" if not subtract_vsys else r"Velocity offset ($\rm km~s^{-1}$)"
            intensity = dataset["y"]
                
    fontsize = labelsize if fontsize is None else fontsize
    params = {'axes.labelsize': labelsize,
              'axes.titlesize': labelsize,
              'font.size': fontsize,
              'figure.dpi': dpi,
              'legend.fontsize': legendsize,
              'xtick.labelsize': ticklabelsize,
              'ytick.labelsize': ticklabelsize,
              'font.family': _fontfamily,
              "mathtext.fontset": _mathtext_fontset, #"Times New Roman"
              'mathtext.tt': _mathtext_tt,
              'axes.linewidth': borderwidth,
              'xtick.major.width': borderwidth,
              'ytick.major.width': borderwidth,
              'figure.figsize': figsize,
#               'xtick.minor.width': 1.0,
#               'ytick.minor.width': 1.0,
               'xtick.major.size': ticksize,
               'ytick.major.size': ticksize,
                }
    rcParams.update(params)
    
    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False, figsize=figsize)
    plt.subplots_adjust(wspace=0.4)
    
    ax.plot(velocity, intensity, color="k", lw=linewidth, label="Observation")
    
    if gaussfit:
        xsmooth, model, [popt, perr] = fit_gauss(velocity, intensity, threshold=threshold, 
                                   return_params=True, **kwargs)
        ax.plot(xsmooth, model, color="tomato", lw=linewidth, label="Model")
        
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
        xmean = popt[1]
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
    
    return ax


# duplicate function
def exportfits(image, *args, **kwargs):
    return image.exportfits(*args, **kwargs)


def search_molecular_line(restfreq, unit="GHz", printinfo=True):
    """
    Function to search for molecular line info given a rest frequency.
    This function requires internet connection.
    """
    # import data base
    from astroquery.splatalogue import Splatalogue 
    
    # error checking for rest frequency
    if restfreq is None:
        raise ValueError("The rest frequency cannot be 'None'.")
        
    if unit != "GHz":
        restfreq = u.Quantity(restfreq, unit).to_value(u.GHz)   # convert unit to GHz
    
    # search for best-match result.
    possible_lines = Splatalogue.query_lines(restfreq*u.GHz-0.01*u.GHz, restfreq*u.GHz+0.01*u.GHz,
                                             show_upper_degeneracy=True, only_NRAO_recommended=False)
    freq1 = possible_lines["Freq-GHz(rest frame,redshifted)"]
    freq2 = possible_lines["Meas Freq-GHz(rest frame,redshifted)"]
    possible_restfreq = np.where(np.ma.is_masked(freq1), freq2, freq1)
    error = np.abs(possible_restfreq - restfreq)
    idx = np.nanargmin(error)
    best_match_result = possible_lines[idx]
    
    # find information of the line 
    species = best_match_result["Species"]
    chemical_name = best_match_result["Chemical Name"]
    freq = possible_restfreq[idx]
    freq_err = best_match_result["Meas Freq Err(rest frame,redshifted)"]
    qns = best_match_result["Resolved QNs"]
    CDMS_JPL_intensity = best_match_result["CDMS/JPL Intensity"]
    Sijmu_sq = best_match_result["S<sub>ij</sub>&#956;<sup>2</sup> (D<sup>2</sup>)"]
    Sij = best_match_result["S<sub>ij</sub>"]
    log10_Aij = best_match_result["Log<sub>10</sub> (A<sub>ij</sub>)"]
    Aij = 10**log10_Aij
    Sijmu_sq = best_match_result["S<sub>ij</sub>&#956;<sup>2</sup> (D<sup>2</sup>)"]
    Lovas_AST_intensity = best_match_result["Lovas/AST Intensity"]
    lerg = best_match_result["E_L (K)"]
    uerg = best_match_result["E_U (K)"]
    gu = best_match_result["Upper State Degeneracy"]
    source = best_match_result["Linelist"]
    
    # find species id
    species_ids = Splatalogue.get_species_ids(species.split("v=")[0])
    species_id = None
    for key, value in species_ids.items():
        str_lst = key[6:].split(" - ")
        if str_lst[0].replace(" ", "") == species and str_lst[1] == chemical_name:
            species_id = value
            break
    if species_id is None:
        print("Failed to find species ID / rotational constants. \n")
        constants = None
    else:
        # find rotational constant
        try:
            # search the web for rotational constants 
            from urllib.request import urlopen
            url = f"https://splatalogue.online/species_metadata_displayer.php?species_id={species_id}"
            page = urlopen(url)
            html = page.read().decode("utf-8")
        except:
            print(f"Failed to read webpage: {url} \n")
            print(f"Double check internet connection / installation of 'urllib' module.")
            url = None
            constants = None
        else:
            lines = np.array(html.split("\n"))
            a_mask = np.char.find(lines, "<td>A</td><td>") != -1
            b_mask = np.char.find(lines, "<td>B</td><td>") != -1
            c_mask = np.char.find(lines, "<td>C</td><td>") != -1
            if np.all(~(a_mask|b_mask|c_mask)):
                constants = None
                print("Failed to find rotational constants. \n")
            else:
                # get rotational A constant
                if np.any(a_mask):
                    a_const = float("".join(char for char in lines[a_mask][0] if char.isnumeric() or char == "."))
                else:
                    a_const = None
                # get rotational B constant
                if np.any(b_mask):
                    b_const = float("".join(char for char in lines[b_mask][0] if char.isnumeric() or char == "."))
                else:
                    b_const = None
                # get rotational C constant
                if np.any(c_mask):
                    c_const = float("".join(char for char in lines[c_mask][0] if char.isnumeric() or char == "."))
                else:
                    c_const = None
                constants = (a_const, b_const, c_const)
                
    # store data in a list to be returned and convert masked data to NaN
    data = [species, chemical_name, freq, freq_err, qns, CDMS_JPL_intensity, Sijmu_sq,
            Sij, Aij, Lovas_AST_intensity, lerg, uerg, gu, constants, source]
    
    for i, item in enumerate(data):
        if np.ma.is_masked(item):
            data[i] = np.nan
    
    # error as warning
    if not np.isclose(error[idx], 0.):
        print("WARNING: This image may not be of a molecular line. \n")
    
    # print information if needed
    if printinfo:
        print(15*"#" + "Line Information" + 15*"#")
        print(f"Species ID: {species_id}")
        print(f"Species: {data[0]}")
        print(f"Chemical name: {data[1]}")
        print(f"Frequency: {data[2]} +/- {data[3]} [GHz]")
        print(f"Resolved QNs: {data[4]}")
        if not np.isnan(data[5]) and data[5] != 0:
            print(f"CDMS/JPL Intensity: {data[5]}")        
        print(f"Sij mu2: {data[6]} [Debye2]")
        print(f"Sij: {data[7]}")
        print(f"Einstein A coefficient (Aij): {data[8]:.3e} [1/s]")
        if not np.isnan(data[9]) and data[9] != 0:
            print(f"Lovas/AST intensity: {data[9]}")
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
            print(f"Link to species data: {url} \n")
    
    # return data
    return tuple(data)


def planck_function(v, T):
    """
    Public function to calculate the planck function value
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
    I_v = continuum.conv_bunit("Jy/sr")
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


def __Qrot_linear(T, B0):
    """
    Private function to calculate rotational partition function value of a linear molecule.
    Source: https://doi.org/10.48550/arXiv.1501.01703
    Parameters:
        T: excitation temperature (K)
        B0: rotational constant of the molecule (MHz)
    Returns:
        Value of rotational partition function.
    """ 
    # constants
    k = const.k_B.cgs
    h = const.h.cgs
    
    # assign units
    if not isinstance(T, u.Quantity):
        T *= u.K
    if not isinstance(B0, u.Quantity):
        B0 *= u.MHz
        
    # warning checking
    if T <= 2.*u.K:
        print("WARNING: Approximation error of partition function is greater than 1%.")
    
    # linear diatomic molecules
    Qrot = k*T/h/B0 + 1/3 + 1/15*(h*B0/k/T) + 4/315*(h*B0/k/T)**2 + 1/315*(h*B0/k/T)**3
    return Qrot.cgs.value


def __Qrot_linear_polyatomic(T, B0):
    """
    Private function to calculate rotational partition function value of linear polyatomic molecule.
    Source: https://doi.org/10.48550/arXiv.1501.01703
    Parameters:
        T: excitation temperature (K)
        B0: rotational constant of the molecule (MHz)
    Returns:
        Value of rotational partition function.
    """ 
    # constants
    k = const.k_B.cgs
    h = const.h.cgs
    
    # assign units
    if not isinstance(T, u.Quantity):
        T *= u.K
    if not isinstance(B0, u.Quantity):
        B0 *= u.MHz
        
    # warning checking
    if T <= 3.5*u.K:
        print("WARNING: Approximation error of partition function is greater than 1%.")
    
    # linear diatomic molecules
    Qrot = k*T/h/B0 * np.exp(h*B0/3/k/T)
    return Qrot.cgs.value


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


def column_density_linear_optically_thin(moment0_map, T_ex, T_bg=2.726, R_i=1, f=1.):
    """
    Public function to calculate the column density of a linear molecule using optically thin assumption.
    Source: https://doi.org/10.48550/arXiv.1501.01703
    Parameters:
        moment0_map (Spatialmap): the moment 0 map
        T_ex (float): the excitation temperature [K]
        T_bg (float): background temperature [K]. 
                      Default is to use cosmic microwave background (2.726 K).
        R_i (float): Relative intensity of transition. 
                     Default is to consider simplest case (= 1)
        f (float): a correction factor accounting for source area being smaller than beam area.
                   Default = 1 assumes source area is larger than beam area.
    Returns:
        cd_img (Spatialmap): the column density map
    """
    # constants
    k = const.k_B.cgs
    h = const.h.cgs
    
    # assign units
    if not isinstance(T_ex, u.Quantity):
        T_ex *= u.K
    if not isinstance(T_bg, u.Quantity):
        T_bg *= u.K
    moment0_map = moment0_map.conv_bunit("K.km/s")
    
    # get info
    line_data = moment0_map.line_info(printinfo=True)
    v = line_data[2]*u.GHz  # rest frequency
    S_mu2 = line_data[6]*(1e-18**2)*(u.cm**5*u.g/u.s**2)  # S mu^2 * g_i*g_j*g_k [debye2]
    E_u = line_data[11]*u.K  # upper energy level 
    B0 = line_data[13][1]*u.MHz  # rotational constant
    Q_rot = __Qrot_linear(T=T_ex, B0=B0)  # partition function
        
    # error checking to make sure molecule is linear
    if line_data[13][0] is not None or line_data[13][2] is not None:
        raise Exception("The molecule is not linear.")
        
    # calculate column density
    aa = 3*h/(8*np.pi**3*S_mu2*R_i)
    bb = Q_rot
    cc = np.exp(E_u/T_ex) / (np.exp(h*v/k/T_ex)-1)
    dd = 1 / (J_v(v=v, T=T_ex)-J_v(v=v, T=T_bg))
    constant = aa*bb*cc*dd/f
    cd_img = constant*(moment0_map*u.K*u.km/u.s)
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
