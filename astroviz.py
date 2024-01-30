"""
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
import pandas as pd

# standard packages
import copy
import os
import sys
import datetime as dt

# scientific packages
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from astropy import units as u
from astropy.units import Unit, def_unit
from astropy import constants as const
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.coordinates import Angle, SkyCoord
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian2DKernel, Box2DKernel, convolve, convolve_fft

# data visualization packages
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm, rcParams, ticker, patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# -

def importfits(fitsfile, hduindex=0, spatialunit="arcsec", quiet=False):
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
    
    # read and export header
    ctype = tuple(hdu_header.get(f"CTYPE{i+1}") for i in range(data.ndim))
    
    # get rest frequency
    if "RESTFRQ" in hdu_header:
        restfreq = hdu_header["RESTFRQ"]
    elif "RESTFREQ" in hdu_header:
        restfreq = hdu_header["RESTFREQ"]
    elif "FREQ" in hdu_header:
        restfreq = hdu_header["FREQ"]
    else:
        restfreq = None

    # find corresponding axes
    if "FREQ" not in ctype:
        nchan = 1
    else:
        freq_axis_num = np.where(np.array(ctype)=="FREQ")[0][0]+1
        nchan = hdu_header.get(f"NAXIS{freq_axis_num}")
        
    if nchan == 1:
        df = None
        startf = None
        faxis = None
        vaxis = None
        dv = None
        vrange = None
    else:
        funit = hdu_header.get(f"CUNIT{freq_axis_num}", "Hz")
        df = u.Quantity(hdu_header.get(f"CDELT{freq_axis_num}"), funit).to_value("Hz")
        startf = u.Quantity(hdu_header.get(f"CRVAL{freq_axis_num}"), funit).to_value("Hz")
        faxis = np.arange(startf, startf+nchan*df, df)
        clight = const.c.to_value(u.km/u.s)
        vaxis = np.round(clight * (1-faxis/restfreq), 5)
        dv = np.abs(vaxis[1]-vaxis[0])
        vrange = [np.min(vaxis), np.max(vaxis)]
        
    if "STOKES" not in ctype:
        nstokes = 1
    else:
        stokes_axis_num = np.where(np.array(ctype)=="STOKES")[0][0]+1
        nstokes = hdu_header.get(f"NAXIS{stokes_axis_num}")
    
    projection = None    # initialize projection parameter, updates if finds one
    if "RA---SIN" in ctype:
        xaxis_num = np.where(np.array(ctype)=="RA---SIN")[0][0]+1
        xunit = hdu_header.get(f"CUNIT{xaxis_num}", "deg")
        refx = hdu_header.get(f"CRVAL{xaxis_num}")
        nx = hdu_header.get(f"NAXIS{xaxis_num}")
        dx = hdu_header.get(f"CDELT{xaxis_num}")
        refnx = hdu_header.get(f"CRPIX{xaxis_num}")
        projection = 'SIN'
        if dx is not None:
            dx = u.Quantity(dx, xunit).to_value(spatialunit)
        if refx is not None:
            refx = u.Quantity(refx, xunit).to_value(spatialunit)
    elif "RA---COS" in ctype:
        xaxis_num = np.where(np.array(ctype)=="RA---COS")[0][0]+1
        xunit = hdu_header.get(f"CUNIT{xaxis_num}", "deg")
        refx = hdu_header.get(f"CRVAL{xaxis_num}")
        nx = hdu_header.get(f"NAXIS{xaxis_num}")
        dx = hdu_header.get(f"CDELT{xaxis_num}")
        refnx = hdu_header.get(f"CRPIX{xaxis_num}")
        projection = 'COS'
        if dx is not None:
            dx = u.Quantity(dx, xunit).to_value(spatialunit)
        if refx is not None:
            refx = u.Quantity(refx, xunit).to_value(spatialunit)
    elif "RA---TAN" in ctype:
        xaxis_num = np.where(np.array(ctype)=="RA---TAN")[0][0]+1
        xunit = hdu_header.get(f"CUNIT{xaxis_num}", "deg")
        refx = hdu_header.get(f"CRVAL{xaxis_num}")
        nx = hdu_header.get(f"NAXIS{xaxis_num}")
        dx = hdu_header.get(f"CDELT{xaxis_num}")
        refnx = hdu_header.get(f"CRPIX{xaxis_num}")
        projection = 'TAN'
        if dx is not None:
            dx = u.Quantity(dx, xunit).to_value(spatialunit)
        if refx is not None:
            refx = u.Quantity(refx, xunit).to_value(spatialunit)
    elif "OFFSET" in ctype:
        xaxis_num = np.where(np.array(ctype)=="OFFSET")[0][0]+1
        xunit = hdu_header.get(f"CUNIT{xaxis_num}", "deg")
        refx = hdu_header.get(f"CRVAL{xaxis_num}")
        nx = hdu_header.get(f"NAXIS{xaxis_num}")
        dx = hdu_header.get(f"CDELT{xaxis_num}")
        refnx = hdu_header.get(f"CRPIX{xaxis_num}")
        if dx is not None:
            dx = u.Quantity(dx, xunit).to_value(spatialunit)
        if refx is not None:
            refx = u.Quantity(refx, xunit).to_value(spatialunit)
    else:
        nx = 1
        refx = None,
        dx = None
        refnx = None
        
    if "DEC--COS" in ctype:
        yaxis_num = np.where(np.array(ctype)=="DEC--COS")[0][0]+1
        yunit = hdu_header.get(f"CUNIT{yaxis_num}", "deg")
        refy = hdu_header.get(f"CRVAL{yaxis_num}")
        ny = hdu_header.get(f"NAXIS{yaxis_num}")
        dy = hdu_header.get(f"CDELT{yaxis_num}")
        refny = hdu_header.get(f"CRPIX{yaxis_num}")
        projection = "COS"
        if dy is not None:
            dy = u.Quantity(dy, yunit).to_value(spatialunit)
        if refy is not None:
            refy = u.Quantity(refy, yunit).to_value(spatialunit)
    elif "DEC---COS" in ctype:
        yaxis_num = np.where(np.array(ctype)=="DEC---COS")[0][0]+1
        yunit = hdu_header.get(f"CUNIT{yaxis_num}", "deg")
        refy = hdu_header.get(f"CRVAL{yaxis_num}")
        ny = hdu_header.get(f"NAXIS{yaxis_num}")
        dy = hdu_header.get(f"CDELT{yaxis_num}")
        refny = hdu_header.get(f"CRPIX{yaxis_num}")
        projection = "COS"
        if dy is not None:
            dy = u.Quantity(dy, yunit).to_value(spatialunit)
        if refy is not None:
            refy = u.Quantity(refy, yunit).to_value(spatialunit)
    elif "DEC--TAN" in ctype:
        yaxis_num = np.where(np.array(ctype)=="DEC--TAN")[0][0]+1
        yunit = hdu_header.get(f"CUNIT{yaxis_num}", "deg")
        refy = hdu_header.get(f"CRVAL{yaxis_num}")
        ny = hdu_header.get(f"NAXIS{yaxis_num}")
        dy = hdu_header.get(f"CDELT{yaxis_num}")
        refny = hdu_header.get(f"CRPIX{yaxis_num}")
        projection = "TAN"
        if dy is not None:
            dy = u.Quantity(dy, yunit).to_value(spatialunit)
        if refy is not None:
            refy = u.Quantity(refy, yunit).to_value(spatialunit)
    elif "DEC---TAN" in ctype:
        yaxis_num = np.where(np.array(ctype)=="DEC---TAN")[0][0]+1
        yunit = hdu_header.get(f"CUNIT{yaxis_num}", "deg")
        refy = hdu_header.get(f"CRVAL{yaxis_num}")
        ny = hdu_header.get(f"NAXIS{yaxis_num}")
        dy = hdu_header.get(f"CDELT{yaxis_num}")
        refny = hdu_header.get(f"CRPIX{yaxis_num}")
        projection = "TAN"
        if dy is not None:
            dy = u.Quantity(dy, yunit).to_value(spatialunit)
        if refy is not None:
            refy = u.Quantity(refy, yunit).to_value(spatialunit)
    elif "DEC--SIN" in ctype:
        yaxis_num = np.where(np.array(ctype)=="DEC--SIN")[0][0]+1
        yunit = hdu_header.get(f"CUNIT{yaxis_num}", "deg")
        refy = hdu_header.get(f"CRVAL{yaxis_num}")
        ny = hdu_header.get(f"NAXIS{yaxis_num}")
        dy = hdu_header.get(f"CDELT{yaxis_num}")
        refny = hdu_header.get(f"CRPIX{yaxis_num}")
        projection = "SIN"
        if dy is not None:
            dy = u.Quantity(dy, yunit).to_value(spatialunit)
        if refy is not None:
            refy = u.Quantity(refy, yunit).to_value(spatialunit)
    elif "DEC---SIN" in ctype:
        yaxis_num = np.where(np.array(ctype)=="DEC---SIN")[0][0]+1
        yunit = hdu_header.get(f"CUNIT{yaxis_num}", "deg")
        refy = hdu_header.get(f"CRVAL{yaxis_num}")
        ny = hdu_header.get(f"NAXIS{yaxis_num}")
        dy = hdu_header.get(f"CDELT{yaxis_num}")
        refny = hdu_header.get(f"CRPIX{yaxis_num}")
        projection = "SIN"
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
    
    # set shape 
    newshape = (nstokes, nchan, nx, ny)
    data = data.reshape(newshape)
    
    # determine image type
    if nx > 1 and ny > 1 and nchan > 1:
        imagetype = "datacube"
    elif nx > 1 and ny > 1 and nchan == 1:
        imagetype = "spatialmap"
    elif nchan > 1 and nx > 1 and ny == 1:
        imagetype = "pvdiagram"
    else:
        raise Exception("Image cannot be read as 'datacube', 'spatialmap', or 'pvdiagram'")
    
    # eliminate rounding error due to float64
    if dx is not None:
        dx = np.round(dx, 7)
    if dy is not None:
        dy = np.round(dy, 7)
    if dv is not None:
        dv = np.round(dv, 7)
        
    # beam size
    bmaj = hdu_header.get("BMAJ", np.nan)
    bmin = hdu_header.get("BMIN", np.nan)
    bpa =  hdu_header.get("BPA", np.nan)
    
    if bmaj is not None:
        bmaj = u.Quantity(bmaj, u.deg).to_value(spatialunit)
        bmin = u.Quantity(bmin, u.deg).to_value(spatialunit)

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
                "beam": tuple(np.round([bmaj, bmin, bpa], 5)),
                "specframe": hdu_header.get("RADESYS"),
                "unit": spatialunit,
                "bunit": hdu_header.get('BUNIT'),
                "projection": projection,
                "instrument": hdu_header.get('INSTRUME'),
                "observer": hdu_header.get('OBSERVER'),
                "obsdate": hdu_header.get('DATE-OBS'),
                "date": hdu_header.get('DATE'),
                "origin": hdu_header.get('ORIGIN'),
                }
    
    # return image
    if imagetype == "datacube":
        return Datacube(fileinfo=fileinfo, data=data)
    elif imagetype == "spatialmap":
        return Spatialmap(fileinfo=fileinfo, data=data)
    elif imagetype == "pvdiagram":
        return PVdiagram(fileinfo=fileinfo, data=data)
    else:
        raise Exception("ERROR: Cannot determine image type.")


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
        faxis = image.get_vaxis(freq=True)
        if hdu_header["NAXIS"] == 4:
            hdu_header["NAXIS"] = 3
            hdu_header.pop("NAXIS4")
            hdu_header.pop("CTYPE4")
            hdu_header.pop("CRVAL4")
            hdu_header.pop("CDELT4")
            hdu_header.pop("CRPIX4")
            hdu_header.pop("CUNIT4")
            hdu_header.pop("PC4_1")
            hdu_header.pop("PC4_2")
            hdu_header.pop("PC4_3")
            hdu_header.pop("PC4_4")
        hdu_header["NAXIS1"] = np.int64(image.nx)
        hdu_header["NAXIS2"] = np.int64(image.nchan)
        hdu_header["NAXIS3"] = np.int64(1)
        hdu_header["CTYPE1"] = "OFFSET"
        hdu_header["CRVAL1"] = np.float64(0.)
        hdu_header["CDELT1"] = np.float64(image.dx)
        hdu_header["CRPIX1"] = np.float64(image.refnx)
        hdu_header["CUNIT1"] = image.unit
        hdu_header["CTYPE2"] = 'FREQ'
        hdu_header["CRVAL2"] = np.float64(faxis[0])
        hdu_header["CDELT2"] = np.float64(faxis[1]-faxis[0])
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
        hdu_header["NAXIS"] = np.int64(4)
        hdu_header["NAXIS1"] = np.int64(image.shape[2])
        hdu_header["NAXIS2"] = np.int64(image.shape[3])
        hdu_header["NAXIS3"] = np.int64(image.shape[1])
        hdu_header["NAXIS4"] = np.int64(image.shape[0])
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
            faxis = image.get_vaxis(freq=True)
            hdu_header["CTYPE3"] = 'FREQ'
            hdu_header["CRVAL3"] = np.float64(faxis[0])
            hdu_header["CDELT3"] = np.float64(faxis[1]-faxis[0])
            hdu_header["CRPIX3"] = np.float64(1.)
            hdu_header["CUNIT3"] = 'Hz'
            hdu_header["ALTRVAL"] = np.float64(image.vaxis[0])  # Alternative frequency referencenpoint
            hdu_header["ALTRPIX"] = np.float64(1.)              # Alternative frequnecy reference pixel
        hdu_header["CTYPE4"] = 'STOKES'
        hdu_header["CRVAL4"] = np.float64(1.)
        hdu_header["CDELT4"] = np.float64(1.)
        hdu_header["CRPIX4"] = np.float64(1.)
        hdu_header["CUNIT4"] = ''

    # other information
    updatedparams = {"BUNIT": image.header["bunit"],
                     "DATE": "%04d-%02d-%02d"%(dtnow.year, dtnow.month, dtnow.day),
                     "BMAJ": np.float64(u.Quantity(image.bmaj, image.unit).to_value(u.deg)),
                     "BMIN": np.float64(u.Quantity(image.bmin, image.unit).to_value(u.deg)),
                     "BPA": np.float64(image.bpa), 
                     "INSTRUME": image.header["instrument"],
                     "DATE-OBS": image.header["obsdate"],
                     "RESTFRQ": np.float64(image.header["restfreq"]),
                     "HISTORY": "Exported from Astroviz."
                     }
    
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
    This is private function to fix a 'bug': astropy's incorrect reading of string units.
    For instance, strings like "Jy/beam.km/s" would be read as u.Jy/u.beam*u.km/u.s 
    by this function.
    """
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
    This is private function to fix a 'bug': astropy's incorrect conversion of units to strings.
    For instance, units like u.Jy/u.beam*u.km/u.s would be read as in the correct order in 
    the latex format by this function.
    """
    if unit.is_equivalent(u.Jy/u.beam*u.km/u.s) or unit.is_equivalent(u.Jy/u.rad**2*u.km/u.s):
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
                 u.Unit(cleanunitstr).is_equivalent(1/u.rad**2):
                newlst[1] = ele
            elif u.Unit(cleanunitstr).is_equivalent(u.km):
                newlst[2] = ele
            elif u.Unit(cleanunitstr).is_equivalent(1/u.s):
                newlst[3] = ele
        newstr = r"$\mathrm{" + ",".join(newlst) + r"}$"
        return newstr
    else:
        return f"{unit:latex_inline}"


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
    def __init__(self, fitsfile=None, fileinfo=None, data=None):
        if fitsfile is not None:
            fits = importfits(fitsfile, hduindex=hduindex, spatialunit=spatialunit, quiet=False)
            self.fileinfo = fits.fileinfo
            self.data = fits.data
        elif fileinfo is not None:
            self.fileinfo = fileinfo
            self.data = data
        if self.fileinfo["imagetype"] != "datacube":
            raise TypeError("The given fitsfile is not a data cube.")
        self.__updateparams()
        
        if isinstance(self.data, u.quantity.Quantity):
            self.value = Datacube(fileinfo=self.fileinfo, data=self.data.value)
        
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
            self.bunit = self.fileinfo["bunit"] = (self.data.unit).to_string()
        else:
            self.bunit = self.fileinfo["bunit"]
        xmin, xmax = self.xaxis[[0,-1]]
        ymin, ymax = self.yaxis[[0,-1]]
        self.imextent = [xmin-0.5*self.dx, xmax+0.5*self.dx, 
                         ymin-0.5*self.dy, ymax+0.5*self.dy]
        self.widestfov = max(self.xaxis[0], self.yaxis[-1])
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
            return Datacube(fileinfo=self.fileinfo, data=self.data+other.data)
        return Datacube(fileinfo=self.fileinfo, data=self.data+other)
        
    def __sub__(self, other):
        if isinstance(other, Datacube):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
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
        yaxis = self.dy * (np.arange(self.ny)-self.refnx+1)
        
        # convert to the specified unit if necessary
        if unit is not None:
            xaxis = u.Quantity(xaxis, self.unit).to_value(unit)
            yaxis = u.Quantity(yaxis, self.unit).to_value(unit)
        
        # make 2D grid if grid is set to True
        if grid:
            xx, yy = np.meshgrid(xaxis, yaxis)
            return xx, yy
        return xaxis, yaxis
    
    def get_vaxis(self, freq=False):
        """
        Get the spectral axis of the image.
        Parameters:
            freq (bool): True to return the spectral axis as frequnecy (unit: Hz.)
                         False to return the spectral axis as velocity (unit: km/s.)
        Returns:
            The spectral axis (ndarray).
        """
        vaxis = np.round(np.arange(self.vrange[0], self.vrange[1]+self.dv, self.dv), 6)
        if freq:
            clight = const.c.to_value(u.km/u.s)
            return self.restfreq*clight/(clight+vaxis)
        return vaxis
    
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
   
    # private method to calculate moment data.
    def __get_momentdata(self, moment, data=None, vaxis=None):
        # intialize parameters
        vaxis = self.vaxis if vaxis is None else vaxis
        data = self.data if data is None else data
        nv = vaxis.size
        dv = vaxis[1] - vaxis[0]
        reshaped_vaxis = vaxis.reshape((1, nv, 1, 1))
        vv = np.broadcast_to(reshaped_vaxis, (1, nv, self.nx, self.ny))

        # mean value of spectrum (unit: intensity)
        if moment == -1:
            momdata = np.nanmean(data, axis=1)
        
        # integrated value of the spectrum (unit: intensity*km/s)
        elif moment == 0:
            momdata = np.nansum(data, axis=1)*dv
        
        # intensity weighted coordinate (unit: km/s)
        elif moment == 1:
            momdata = np.nansum(data*vv, axis=1) / np.nansum(data, axis=1)
        
        # intensity weighted dispersion of the coordinate (unit: km/s)
        elif moment == 2:
            meanvel = np.nansum(data*vv, axis=1) / np.nansum(data, axis=1)
            momdata = np.sqrt(np.nansum(data*(vv-meanvel)**2, axis=1)/np.nansum(data, axis=1))
            
        # median value of the spectrum (unit: intensity)
        elif moment == 3:
            momdata = np.nanmedian(data, axis=1)
        
        # median coordinate (unit: km/s)
        elif moment == 4:
            func = lambda m: np.nanmedian(np.broadcast_to(reshaped_vaxis, m.shape)*m, axis=1)
            momdata = np.apply_along_axis(func, 1, data)
        
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
            maxidx = np.nanargmax(data, axis=1)
            momdata = np.take_along_axis(reshaped_vaxis, maxidx, axis=2)
            
        # minimum value of the spectrum (unit: intensity)
        elif moment == 10:
            momdata = np.nanmin(data, axis=1)
        
        # coordinate of the minimum value of the spectrum (unit: km/s)
        elif moment == 11:
            minidx = np.nanargmin(data, axis=1)
            momdata = np.take_along_axis(vv, minidx, axis=2)
        
        return momdata.reshape((1, 1, self.nx, self.ny))
    
    def immoments(self, moments=[0], vrange=[], chans=[], threshold=None):
        """
        Parameters:
            moments (list[int]): a list of moment maps to be outputted
            vrange (list[float]): a list of [minimum velocity, maximum velocity] in km/s
            threshold (float): a threshold to be applied to data cube. None to not use a threshold.
        Returns:
            A list of moment maps (Sptialmap objects).
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
                bunit = f"{self.bunit}.km/s"
            else:
                bunit = "km/s"
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
        
    def imview(self, contourmap=None, fov=None, ncol=5, nrow=None, cmap="inferno",
               figsize=(11.69, 8.27), vrange=None, nskip=1, vskip=None, tickscale=None, 
               tickcolor="w", txtcolor='w', crosson=True, crosslw=0.5, crosscolor="w", 
               crosssize=0.3, dpi=400, vmin=None, vmax=None, xlabel=None, ylabel=None, 
               xlabelon=True, ylabelon=True, crms=None, clevels=np.arange(3, 21, 3), 
               ccolor="w", clw=0.5, vsys=0., fontsize=12, decimals=2, vlabelon=True, 
               cbarloc="right", cbarwidth="3%", cbarpad=0., cbarlabel=None, cbarlabelon=True, 
               addbeam=True, beamcolor="skyblue", nancolor="k", labelcolor="k", axiscolor="w",
               axeslw=0.8, labelsize=10, tickwidth=1., ticksize=3., tickdirection="in", 
               vlabelsize=12, vlabelunit=False, cbaron=True, plot=True):
        """
        To plot the data cube's channel maps.
        Parameters:
            contourmap (Spatialmap/Datacube): The contour map to be drawn. Default is to not plot contour.
            fov (float): the field of view of the image in the same spaital unit as the data cube.
            ncol (int): the number of columns to be drawn. 
            nrow (int): the number of rows to be drawn. Default is the minimum rows needed to plot all specified channels.
            cmap (str): the color map of the color image.
            figsize (tuple(float)): the size of the figure
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
        fov = self.widestfov if fov is None else fov
        if tickscale is not None:
            ticklist = np.arange(0, fov, tickscale) 
            ticklist = np.append(-ticklist, ticklist)
        xlabel = f"Relative RA ({self.unit})" if xlabel is None else xlabel
        ylabel = f"Relative Dec ({self.unit})" if ylabel is None else ylabel
        clevels = np.array(clevels) if not isinstance(clevels, np.ndarray) else clevels
        cbarlabel = f"({(u.Unit(self.bunit)):latex_inline})" if cbarlabel is None else cbarlabel
        vaxis = self.vaxis - vsys
        vrange = [vaxis.min(), vaxis.max()] if vrange is None else vrange
        velmin, velmax = vrange
        nskip = int(vskip/self.dv) if vskip is not None else nskip
        vskip = self.dv*nskip if vskip is None else vskip
        cmap = copy.deepcopy(mpl.colormaps[cmap]) 
        cmap.set_bad(color=nancolor) 
        if (contourmap is not None and crms is None):
            crms = contourmap.noise()
            print(f"Estimated base contour level (rms): {crms:.4e}")
        
        # trim data along vaxis for plotting:
        vmask = (velmin <= vaxis) & (vaxis <= velmax)
        trimmed_data = self.data[:, vmask, :, :][:, ::nskip, :, :]
        trimmed_vaxis = vaxis[vmask][::nskip]
        
        # trim data along xyaxes for plotting:
        if fov != self.widestfov:
            xmask = (-fov<=self.xaxis) & (self.xaxis<=fov)
            ymask = (-fov<=self.yaxis) & (self.yaxis<=fov)
            trimmed_data = trimmed_data[:, :, xmask, :]
            trimmed_data = trimmed_data[:, :, :, ymask]
        imextent = [fov-0.5*self.dx, -fov+0.5*self.dx, 
                    -fov-0.5*self.dy, fov+0.5*self.dy]
            
        # modify contour map to fit the same channels:
        if contourmap is not None:
            contmap = contourmap.copy()
            contmap_isdatacube = (contmap.header["imagetype"] == "datacube")
            if contmap_isdatacube:
                cvaxis = contmap.vaxis - vsys
            cxmask = (-fov<=contmap.xaxis) & (contmap.xaxis<=fov)
            cymask = (-fov<=contmap.yaxis) & (contmap.yaxis<=fov)
            trimmed_cdata = contmap.data[:, :, cxmask, :]
            trimmed_cdata = trimmed_cdata[:, :, :, cymask]
            contextent = [fov-0.5*contmap.dx, -fov+0.5*contmap.dx, 
                         -fov-0.5*contmap.dy, fov+0.5*contmap.dy]
        
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
                  'font.family': 'Times New Roman',
                  'mathtext.fontset': 'stix', #"Times New Roman"
                  'mathtext.tt':' Times New Roman',
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
                
                # set field of view
                ax.set_xlim(fov, -fov)
                ax.set_ylim(-fov, fov)
                
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
                    vlabel = r"$\rm " + f"%.{decimals}f"%thisvel + r"~km~s^{-1}$" \
                             if vlabelunit else f"%.{decimals}f"%thisvel
                    ax.text(0.1, 0.9, vlabel, color=txtcolor, size=vlabelsize,
                            ha='left', va='top', transform=ax.transAxes)
                
                # plot central cross
                if crosson:
                    ax.plot([crosssize*fov, -crosssize*fov], [0., 0.], 
                             color=crosscolor, lw=crosslw, zorder=99)   # horizontal line
                    ax.plot([0., 0.], [-crosssize*fov, crosssize*fov], 
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
            bmin_plot, bmaj_plot = ax.transLimits.transform((self.bmin, self.bmaj)) - ax.transLimits.transform((0, 0))
            beam = patches.Ellipse(xy=(0.1, 0.11), width=bmin_plot, height=bmaj_plot, 
                                   fc=beamcolor, angle=self.bpa, transform=ax.transAxes)
            bottomleft_ax.add_patch(beam)

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
        
        # show image
        if plot:
            plt.show()
        return grid
        
    def trim(self, vrange, nskip=1, vskip=None, inplace=False):
        """
        This method trims the data cube along the velocity axis.
        Parameters:
            vrange (iterable): the [minimum velocity, maximum velocity], inclusively.
            nskip (float): the number of channel increments
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
        bunit = _to_apu(bunit)
        area_per_beam = self.beam_area(unit=None)
        area_per_pix = self.pixel_area(unit=None)
        equiv_bt = u.equivalencies.brightness_temperature(frequency=self.restfreq*u.Hz, beam_area=area_per_beam)
        equiv_pix = u.pixel_scale(u.Quantity(np.abs(self.dx), self.unit)**2/u.pixel)
        equiv_ba = u.beam_angular_area(area_per_beam)
        try:
            newdata = u.Quantity(self.data, _to_apu(self.bunit)).to(bunit)
        except:
            try:
                newdata = u.Quantity(self.data, _to_apu(self.bunit)).to(bunit, equivalencies=equiv_bt)
            except:
                try:
                    newdata = u.Quantity(self.data, _to_apu(self.bunit)).to(bunit, equivalencies=equiv_pix)
                except:
                    try: 
                        newdata = u.Quantity(self.data, _to_apu(self.bunit)).to(bunit, equivalencies=equiv_ba)
                    except:
                        try:
                            newdata = u.Quantity(self.data, _to_apu(self.bunit)).to("Jy/sr", equivalencies=equiv_ba)
                            newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_pix)
                        except:
                            try:
                                newdata = u.Quantity(self.data, _to_apu(self.bunit)).to("Jy/sr", equivalencies=equiv_pix)
                                newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_ba)
                            except:
                                try:
                                    newdata = u.Quantity(self.data, _to_apu(self.bunit)).to("Jy/sr", equivalencies=equiv_bt)
                                    newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_pix)
                                except:
                                    newdata = u.Quantity(self.data, _to_apu(self.bunit)).to("Jy/sr", equivalencies=equiv_pix)
                                    newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_bt)
        
        # return and set values
        newdata = newdata.value
        if inplace:
            self.data = newdata
            self.header["bunit"] = bunit.to_string()
            self.__updateparams()
            return self
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["bunit"] = bunit
        newimage = Datacube(fileinfo=newfileinfo, data=newdata)
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
                          None to use same unit as the positional axes.
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
        bg_image = self.set_data(clipped_data, inplace=False)
        mean = np.nanmean(clipped_data)
        rms = np.sqrt(np.nanmean(clipped_data**2))
        std = np.nanstd(clipped_data)
        
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
            print("Not Implemented Yet")
            return
        elif region.shape == "rectangle":
            print("Not Implemented Yet")
            return
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
        # raise exception if width is an even number of pixels.
        if width % 2 == 0:
            raise ValueError("The parameter 'width' must be an odd positive integer.")
        if isinstance(width, float):
            width = int(width)
            
        # initialize parameters
        vrange = [self.vaxis[0], self.vaxis[-1]] if vrange is None else vrange
        region = self.__readregion(region)
        center, pa, length = region.center, region.pa, region.length
        
        # shift and then rotate image
        shifted_img = self.imshift(center, inplace=False, printcc=False) if center != (0, 0) else self.copy()
        rotated_img = shifted_img.rotate(angle=pa, ccw=False, inplace=False) if pa != 0 else shifted_img
        
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
            xidx1, xidx2 = idx-add_idx, idx+add_idx+1 # plus 1 to avoid OBOB
            pv_data = pv_data[:, :, xidx1:xidx2, :]
            pv_data = np.nanmean(pv_data, axis=2)
        pv_data = pv_data.reshape(1, pv_vaxis.size, pv_xaxis.size, 1)
        
        # export as pv data
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["shape"] = pv_data.shape
        newfileinfo["imagetype"] = "pvdiagram"
        newfileinfo["vrange"] = [pv_vaxis.min(), pv_vaxis.max()]
        newfileinfo["dv"] = np.round(pv_vaxis[1] - pv_vaxis[0], 7)
        newfileinfo["nchan"] = pv_vaxis.size
        newfileinfo["dx"] = np.round(pv_xaxis[1] - pv_xaxis[0], 7)
        newfileinfo["nx"] = pv_xaxis.size
        newfileinfo["refnx"] = pv_xaxis.size/2 + 1
        newfileinfo["dy"] = None
        newfileinfo["ny"] = None
        newfileinfo["refny"] = None
        newfileinfo["refcoord"] = _relative2icrs(center, ref=self.refcoord, unit=self.unit)
        
        # new image
        pv = PVdiagram(fileinfo=newfileinfo, data=pv_data)
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
        ax = plt_spectrum(vaxis, intensity, xlabel=xlabel, ylabel=ylabel, **kwargs)
        
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
            dy = (self.yaxis.max() - self.yaxis.min())/imsize[0]

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
        # this method of looping is somehow faster than specifying axis with the built-in parameter.
        image.data[0] = np.array([ndimage.shift(image.data[0, i], shift=shift) for i in range(self.nchan)])
        image.__updateparams()
        return image
    
    def get_hduheader(self):
        """
        To retrieve the header of the current FITS image. This method accesses the header 
        information of the original FITS file, and then modifies it to reflect the current
        status of this image object.

        Returns:
            The FITS header of the current image object (astropy.io.fits.header.Header).
        """
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
        # update parameters before saving
        self.__updateparams()
        
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
    def __init__(self, fitsfile=None, fileinfo=None, data=None, hduindex=0, spatialunit="arcsec", quiet=False):
        if fitsfile is not None:
            fits = importfits(fitsfile, hduindex=hduindex, spatialunit=spatialunit, quiet=False)
            self.fileinfo = fits.fileinfo
            self.data = fits.data
        elif fileinfo is not None:
            self.fileinfo = fileinfo
            self.data = data
        if self.fileinfo["imagetype"] != "spatialmap":
            raise TypeError("The given fitsfile is not a spatialmap.")
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
            return Spatialmap(fileinfo=self.fileinfo, data=self.data+other.data)
        return Spatialmap(fileinfo=self.fileinfo, data=self.data+other)
        
    def __sub__(self, other):
        if isinstance(other, Spatialmap):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
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
            self.bunit = self.fileinfo["bunit"] = (self.data.unit).to_string()
        else:
            self.bunit = self.fileinfo["bunit"]
        xmin, xmax = self.xaxis[[0,-1]]
        ymin, ymax = self.yaxis[[0,-1]]
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
        yaxis = self.dy * (np.arange(self.ny)-self.refnx+1)
        
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
            print("Not Implemented Yet")
            return
        elif region.shape == "rectangle":
            print("Not Implemented Yet")
            return
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
            inplace (bool): If True, modify the current image in-place. Otherwise, return a new image.
        Returns:
            Spatialmap: The regridded image.
        """
        if template is not None:
            dx = template.dx
            dy = template.dy
        
        if len(imsize) == 2:
            dx = -(self.xaxis.max()-self.xaxis.min())/imsize[0]
            dy = (self.yaxis.max() - self.yaxis.min())/imsize[0]

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
            kernel (str): the type of convolution to be used .This parameter is case-insensitive.
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
            area = area.to(unit).value
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
            area = area.to(unit).value
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
        bunit = _to_apu(bunit)
        area_per_beam = self.beam_area(unit=None)
        area_per_pix = self.pixel_area(unit=None)
        equiv_bt = u.equivalencies.brightness_temperature(frequency=self.restfreq*u.Hz, beam_area=area_per_beam)
        equiv_pix = u.pixel_scale(u.Quantity(np.abs(self.dx), self.unit)**2/u.pixel)
        equiv_ba = u.beam_angular_area(area_per_beam)
        try:
            newdata = u.Quantity(self.data, _to_apu(self.bunit)).to(bunit)
        except:
            try:
                newdata = u.Quantity(self.data, _to_apu(self.bunit)).to(bunit, equivalencies=equiv_bt)
            except:
                try:
                    newdata = u.Quantity(self.data, _to_apu(self.bunit)).to(bunit, equivalencies=equiv_pix)
                except:
                    try: 
                        newdata = u.Quantity(self.data, _to_apu(self.bunit)).to(bunit, equivalencies=equiv_ba)
                    except:
                        try:
                            newdata = u.Quantity(self.data, _to_apu(self.bunit)).to("Jy/sr", equivalencies=equiv_ba)
                            newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_pix)
                        except:
                            try:
                                newdata = u.Quantity(self.data, _to_apu(self.bunit)).to("Jy/sr", equivalencies=equiv_pix)
                                newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_ba)
                            except:
                                try:
                                    newdata = u.Quantity(self.data, _to_apu(self.bunit)).to("Jy/sr", equivalencies=equiv_bt)
                                    newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_pix)
                                except:
                                    newdata = u.Quantity(self.data, _to_apu(self.bunit)).to("Jy/sr", equivalencies=equiv_pix)
                                    newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_bt)
        
        # return and set values
        newdata = newdata.value
        if inplace:
            self.data = newdata
            self.header["bunit"] = bunit.to_string()
            self.__updateparams()
            return self
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["bunit"] = bunit.to_string()
        newimage = Spatialmap(fileinfo=newfileinfo, data=newdata)
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
        region = self.__readregion(region)
        if region is None:
            stat_image = copy.deepcopy(self)  
        else:
            if region is None:
                fov = None
            else:
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
            
    def imview(self, contourmap=None, title="", fov=None, vmin=None, vmax=None, 
               scale="linear", gamma=None, crms=None, clevels=np.arange(3, 21, 3), tickscale=None, 
               scalebaron=True, distance=None, cbarlabelon=True, cbarlabel=None, xlabelon=True,
               ylabelon=True, dpi=500, ha="left", va="top", titlepos=0.85, 
               cmap=None, fontsize=12, cbarwidth="5%", width=330, height=300,
               imsmooth=None, scalebarsize=0.1, nancolor=None, beamcolor=None,
               ccolors=None, clw=0.8, txtcolor=None, cbaron=True, cbarpad=0., tickson=False, 
               labelcolor="k", tickcolor="k", labelsize=10., ticklabelsize=10., 
               cbartick_length=3., cbartick_width=1., beamon=True, scalebar_fontsize=10,
               axeslw=1., scalecolor=None, scalelw=1., orientation="vertical", plot=True):
        """
        Method to plot the image.
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
            titlepos (float): position of title (x, y) = (titlepos*fov, titlepos*fov)
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
        if cmap is None:
            cmap = 'RdBu_r' if self.bunit == "km/s" else 'inferno'
            
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
        
        # copy data and convert data to unitless array, if necessary
        data = self.data.copy()
        if isinstance(data, u.Quantity):
            data = data.value
            
        clevels = np.array(clevels) if not isinstance(clevels, np.ndarray) else clevels
        
        # create a copy of the contour map
        if isinstance(contourmap, Spatialmap):
            contourmap = contourmap.copy()            
        
        # set field of view to widest of image by default
        fov = self.widestfov if fov is None else fov
        
        # change default parameters
        letterratio = 1.294
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
                  'xtick.top'           : True,   # draw ticks on the top side
                  'xtick.major.top' : True,
                  'figure.figsize': fig_size,
                  'font.family': 'Times New Roman',
                  "mathtext.fontset" : 'stix', 
                  'mathtext.tt':'Times New Roman',
                  'axes.linewidth' : axeslw,
                  'xtick.major.width' : 1.0,
                  "xtick.direction": 'in',
                  "ytick.direction": 'in',
                  'ytick.major.width' : 1.0,
                  'xtick.minor.width' : 1.0,
                  'ytick.minor.width' : 1.0,
                  'xtick.major.size' : 6.,
                  'ytick.major.size' : 6.,
                  'xtick.minor.size' : 4.,
                  'ytick.minor.size' : 4.,
                  'figure.dpi': dpi,
                }
        rcParams.update(params)
        
        if tickscale is not None:
            ticklist = np.arange(-100, 100+tickscale, tickscale)
        
        # set colorbar
        my_cmap = copy.deepcopy(mpl.colormaps[cmap]) 
        my_cmap.set_bad(color=nancolor) 
        
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False)
        plt.subplots_adjust(wspace=0.4)
        
        if contourmap is not None and crms is None:
            crms = contourmap.noise()
            bunit = self.bunit.replace(".", " ")
            print(f"Estimated base contour level (rms): {crms:.4e} [{bunit}]")
        
        # add image
        climage = ax.imshow(data[0, 0], cmap=my_cmap, extent=self.imextent, 
                            vmin=vmin, vmax=vmax, origin='lower')
        if contourmap is not None:
            contour_data = contourmap.data[0, 0]
            if imsmooth is not None:
                contour_data = gaussian_filter(contour_data, imsmooth)
            ax.contour(contour_data, extent=contourmap.imextent, 
                       levels=crms*clevels, colors=ccolors, linewidths=clw, origin='lower')
        
        if scalebaron and distance is not None:
            obj_distance = distance*scalebarsize*2.06265*1e5*(np.pi/(3600*180)) 
            obj_distance = np.round(distance, 1)
            centerx = -0.7*fov
            ax.text(centerx, fov*(-0.85), str(int(distance*scalebarsize))+' au', 
                     ha='center', va='bottom',color=scalecolor, fontsize=scalebar_fontsize)
            ax.plot([centerx-scalebarsize/2, centerx+scalebarsize/2],
                     [fov*(-0.7),fov*(-0.7)], color=scalecolor,lw=scalelw)
            
        if title:
            titlex, titley = fov*titlepos, fov*titlepos
            ax.text(x=titlex, y=titley, s=title, ha=ha, va=va, color=txtcolor, fontsize=fontsize)
        
        if cbaron:
            if cbarlabelon:
                if cbarlabel is None:
                    cbarlabel = f"({_apu_to_str(_to_apu(self.bunit))})"
                else:
                    cbarlabel = ""
            ticklocation = "right"
            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes(ticklocation, size=cbarwidth, pad=cbarpad)
            cb = plt.colorbar(climage, cax=ax_cb, orientation=orientation, 
                              ticklocation=ticklocation)
            cb.set_label(cbarlabel, fontsize=labelsize)
            cb.ax.tick_params(labelsize=ticklabelsize, width=cbartick_width, 
                              length=cbartick_length, direction='in')
        
        ax.set_xlim([fov, -fov])
        ax.set_ylim([-fov, fov])
        
        if tickscale is not None:
            ax.set_xticks(ticklist)
            ax.set_yticks(ticklist)
        else:
            defaultticks = ax.get_xticks().copy()
            ax.set_yticks(-defaultticks)
            ax.set_xticks(defaultticks)
            
        if tickson:
            ax.tick_params(which='both',direction='in',bottom=True, top=True, left=True, right=True,
                    colors=tickcolor, labelrotation=0, labelcolor=labelcolor)
        else:
            ax.tick_params(which='both',direction='in',bottom=False, top=False, left=False, right=False,
                        colors=tickcolor, labelrotation=0, labelcolor=labelcolor)
            
        if xlabelon:
            ax.set_xlabel(f"Relative RA ({self.unit})", fontsize=labelsize)
        if ylabelon:
            ax.set_ylabel(f"Relative Dec ({self.unit})", fontsize=labelsize)
        
        # add beam
        if beamon:
            bmin_plot, bmaj_plot = ax.transLimits.transform((self.bmin, self.bmaj))-ax.transLimits.transform((0,0))
            beam = patches.Ellipse(xy=(0.1, 0.11), width=bmin_plot, height=bmaj_plot, fc=beamcolor, 
                                   angle=self.bpa, transform=ax.transAxes)
            ax.add_patch(beam)
        
        # set field of view
        ax.set_xlim([fov, -fov])
        ax.set_ylim([-fov, fov])
        
        if plot:
            plt.show()
        return ax
    
    def get_hduheader(self):
        """
        To retrieve the header of the current FITS image. This method accesses the header 
        information of the original FITS file, and then modifies it to reflect the current
        status of this image object.

        Returns:
            The FITS header of the current image object (astropy.io.fits.header.Header).
        """
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
        # update parameters before saving
        self.__updateparams()
        
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
    def __init__(self, fitsfile=None, fileinfo=None, data=None):
        if fitsfile is not None:
            fits = importfits(fitsfile, hduindex=hduindex, spatialunit=spatialunit, quiet=False)
            self.fileinfo = fits.fileinfo
            self.data = fits.data
        elif fileinfo is not None:
            self.fileinfo = fileinfo
            self.data = data
        if self.fileinfo["imagetype"] != "pvdiagram":
            raise TypeError("The given fitsfile is not a PV diagram.")
        self.__updateparams()
        
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
            self.bunit = self.fileinfo["bunit"] = (self.data.unit).to_string()
        else:
            self.bunit = self.fileinfo["bunit"]
        self.dv = self.fileinfo["dv"]
        self.vrange = self.fileinfo["vrange"]
        self.nv = self.nchan = self.fileinfo["nchan"]
        self.vaxis = self.get_vaxis()
        vmin, vmax = self.vaxis[[0,-1]]
        xmin, xmax = self.xaxis[[0,-1]]
        self.imextent = [vmin-0.5*self.dx, vmax+0.5*self.dx, 
                         xmin-0.5*self.dx, xmax+0.5*self.dx]
        v1, v2 = self.vaxis[0]-self.dv*0.5, self.vaxis[-1]+self.dv*0.5
        x1 = self.xaxis[0]-self.dx*0.5 if self.xaxis[0] == self.xaxis.min() \
             else self.xaxis[0]+self.dx*0.5
        x2 = self.xaxis[-1]+self.dx*0.5 if self.xaxis[-1] == self.xaxis.max() \
             else self.xaxis[-1]-self.dx*0.5
        self.imextent = [v1, v2, x1, x2]
        self.maxxlim = [self.xaxis[0], self.xaxis[-1]]
        self.maxvlim = [self.vaxis[0], self.vaxis[-1]]
        
    # magic methods to define operators
    def __add__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
            return PVdiagram(fileinfo=self.fileinfo, data=self.data+other.data)
        return PVdiagram(fileinfo=self.fileinfo, data=self.data+other)
        
    def __sub__(self, other):
        if isinstance(other, PVdiagram):
            if self.resolution is not None and other.resolution is not None:
                if np.round(self.resolution, 1) != np.round(other.resolution, 1):
                    print("WARNING: operation performed on two images with significantly different beam sizes.")
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
                return self.data.copy[:, indices[0], :, indices[1]]
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
            The spectral axis (ndarray).
        """
        # create axes in arcsec
        xstart, xend = -self.nx/2*self.dx, self.nx/2*self.dx
        xaxis = np.round(np.arange(xstart, xend, self.dx), 6)
        
        # convert units to specified
        if unit is not None:
            xaxis = u.Quantity(xaxis, self.unit).to_value(unit)
        return xaxis
    
    def get_vaxis(self, freq=False):
        """
        Get the spectral axis of the image.
        Parameters:
            freq (bool): True to return the spectral axis as frequnecy (unit: Hz.)
                         False to return the spectral axis as velocity (unit: km/s.)
        Returns:
            The spectral axis (ndarray).
        """
        vaxis = np.round(np.arange(self.vrange[0], self.vrange[1]+self.dv, self.dv), 6)
        if freq:
            clight = const.c.to_value(u.km/u.s)
            return self.restfreq*clight/(clight+vaxis)
        return vaxis
    
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
        
        clipped_data = sigma_clip(self.data, sigma=sigma, maxiters=10000, masked=False, axis=(0, 1, 2, 3))
        bg_image = self.set_data(clipped_data, inplace=False)
        mean = np.nanmean(clipped_data)
        rms = np.sqrt(np.nanmean(clipped_data**2))
        std = np.nanstd(clipped_data)
        
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
            self.header["unit"] = u.Unit(unit).to_string()
            self.__updateparams()
            return self
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["dx"] = u.Quantity(self.dx, self.unit).to_value(unit)
        newfileinfo["beam"] = newbeam
        newfileinfo["unit"] = u.Unit(unit).to_string()
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
        area_per_beam = self.beam_area(unit=None)
        area_per_pix = self.pixel_area(unit=None)
        equiv_bt = u.equivalencies.brightness_temperature(frequency=self.restfreq*u.Hz, beam_area=area_per_beam)
        equiv_pix = u.pixel_scale(u.Quantity(np.abs(self.dx), self.unit)**2/u.pixel)
        equiv_ba = u.beam_angular_area(area_per_beam)
        try:
            newdata = u.Quantity(self.data, self.bunit).to(bunit)
        except:
            try:
                newdata = u.Quantity(self.data, self.bunit).to(bunit, equivalencies=equiv_bt)
            except:
                try:
                    newdata = u.Quantity(self.data, self.bunit).to(bunit, equivalencies=equiv_pix)
                except:
                    try: 
                        newdata = u.Quantity(self.data, self.bunit).to(bunit, equivalencies=equiv_ba)
                    except:
                        try:
                            newdata = u.Quantity(self.data, self.bunit).to("Jy/sr", equivalencies=equiv_ba)
                            newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_pix)
                        except:
                            try:
                                newdata = u.Quantity(self.data, self.bunit).to("Jy/sr", equivalencies=equiv_pix)
                                newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_ba)
                            except:
                                try:
                                    newdata = u.Quantity(self.data, self.bunit).to("Jy/sr", equivalencies=equiv_bt)
                                    newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_pix)
                                except:
                                    newdata = u.Quantity(self.data, self.bunit).to("Jy/sr", equivalencies=equiv_pix)
                                    newdata = u.Quantity(newdata).to(bunit, equivalencies=equiv_bt)
        
        # return and set values
        newdata = newdata.value
        if inplace:
            self.data = newdata
            self.header["bunit"] = bunit
            self.__updateparams()
            return self
        newfileinfo = copy.deepcopy(self.fileinfo)
        newfileinfo["bunit"] = bunit
        newimage = PVdiagram(fileinfo=newfileinfo, data=newdata)
        return newimage
    
    def imview(self, contourmap=None, cmap="inferno", nancolor="k", crms=None, clevels=np.arange(3, 21, 3),
               ccolor="w", clw=1., dpi=500, cbaron=True, cbarloc="right", cbarpad="0%", vsys=None, 
               xlim=None, vlim=None, xcenter=0., vlineon=True, xlineon=True, cbarlabelon=True, cbarwidth='5%',
               cbarlabel=None, fontsize=18, labelsize=18, figsize=(11.69, 8.27), plotres=True, 
               xlabelon=True, vlabelon=True, xlabel=None, vlabel=None, offset_as_hor=False, 
               aspect_ratio=1.1, axeslw=1.3, tickwidth=1.3, tickdirection="in", ticksize=5., 
               xticks=None, vticks=None, title=None, titlepos=0.85, ha="left", va="top", txtcolor="w", 
               refline_color="w", refline_width=None, plot=True):
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
        clevels = clevels if isinstance(clevels, np.ndarray) else np.array(clevels)
        crms = contourmap.noise() if (crms is None and contourmap is not None) else crms
        contmap = contourmap.copy() if contourmap is not None else contourmap
        xlim = self.maxxlim if xlim is None else xlim
        vlim = self.maxvlim if vlim is None else vlim
        if vlabel is None:
            vlabel = r'$\mathrm{LSR\ velocity\ (km\ s^{-1})}$' \
                        if vsys is None else r'$v_{\rm obs}-v_{\rm sys}$ ($\rm km~s^{-1}$)'
        xlabel = f'Offset ({self.unit})' if xlabel is None else xlabel
        cbarlabel = f"({u.Unit(self.bunit):latex_inline})" if cbarlabel is None else cbarlabel
        refline_width = clw if refline_width is None else refline_width
        vres, xres = self.dv, np.sqrt(self.bmaj*self.bmin)
        cmap = copy.deepcopy(mpl.colormaps[cmap]) 
        cmap.set_bad(color=nancolor)
                
        # change default matplotlib parameters
        params = {'axes.labelsize': labelsize,
                  'axes.titlesize': labelsize,
                  'font.size': fontsize,
                  'legend.fontsize': labelsize,
                  'xtick.labelsize': labelsize,
                  'ytick.labelsize': labelsize,
                  'xtick.top': True,   # draw ticks on the top side
                  'xtick.major.top': True,
                  'figure.figsize': figsize,
                  'figure.dpi': dpi,
                  'font.family': 'Times New Roman',
                  'mathtext.fontset': 'stix', #"Times New Roman"
                  'mathtext.tt':' Times New Roman',
                  'axes.linewidth': axeslw,
                  'xtick.major.width': tickwidth,
                  'xtick.major.size': ticksize,
                  'xtick.direction': tickdirection,
                  'ytick.major.width': tickwidth,
                  'ytick.major.size': ticksize,
                  'ytick.direction': tickdirection,
                  }
        rcParams.update(params)
        
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111)
        
        # plot image
        if offset_as_hor:
            imextent = [self.imextent[2], self.imextent[3], self.imextent[0], self.imextent[1]]
            colordata = self.data[0, :, :, 0]
            climage = ax.imshow(colordata, cmap=cmap, extent=imextent, origin='lower')
            if contmap is not None:
                contextent = [contmap.imextent[2], contmap.imextent[3], contmap.imextent[0], contmap.imextent[1]]
                contdata = contmap.data[0, :, :, 0].reshape(contmap.nv, contmap.nx)
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
                ax.plot([xlim[0], xlim[1]], [vsys, vsys], color=refline_color, 
                        ls='dashed', lw=refline_width)
        else:
            imextent = self.imextent
            colordata = self.data[0, :, :, 0].T
            climage = ax.imshow(colordata, cmap=cmap, extent=imextent, origin='lower')
            if contmap is not None:
                contextent = contmap.imextent
                contdata = contmap.data[0, :, :, 0].T
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
                ax.plot([vsys, vsys], [xlim[0], xlim[1]], color=refline_color, 
                        ls='dashed', lw=refline_width)
            if xlineon:
                ax.plot([vlim[0], vlim[1]], [xcenter, xcenter], color=refline_color, 
                        ls='dashed', lw=refline_width)
        # tick parameters
        ax.tick_params(which='both', direction=tickdirection, bottom=True, top=True, left=True, right=True,
                       pad=9, labelsize=labelsize)
        
        # set aspect ratio
        aspect = (1/aspect_ratio)*(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
        ax.set_aspect(float(np.abs(aspect)))
        
        # color bar
        if cbaron:
            axin_cb = inset_axes(ax, width=cbarwidth, height='100%',
                                 loc='lower left', bbox_to_anchor=(1.0 + float(cbarpad.strip('%'))*0.01, 0., 1., 1.),
                                 bbox_transform=ax.transAxes, borderpad=0)
            cbar = fig.colorbar(climage, cax=axin_cb, pad=cbarpad)
            cbar.set_label(cbarlabel)
        
        # plot resolution
        if plotres:
            res_x, res_y = (xres, vres) if offset_as_hor else (vres, xres)
            res_x_plt, res_y_plt = ax.transLimits.transform((res_x*0.5, res_y*0.5))-ax.transLimits.transform((0, 0))
            ax.errorbar(0.1, 0.1, xerr=res_x_plt, yerr=res_y_plt, color=ccolor, capsize=3, 
                        capthick=1., elinewidth=1., transform=ax.transAxes)
            
        # plot title, if necessary
        if title is not None:
            hormin, hormax = ax.get_xlim()
            xfov = (hormax - hormin)/2
            xmidpt = (hormin + hormax)/2
            vertmin, vertmax = ax.get_ylim()
            yfov = (vertmax - vertmin)/2
            ymidpt = (vertmax+vertmin)/2
            titlex, titley = xmidpt-xfov*titlepos, ymidpt+yfov*titlepos
            ax.text(x=titlex, y=titley, s=title, ha=ha, va=va, color=txtcolor, fontsize=fontsize)

        if plot:
            plt.show()
        
        return ax
    
    def get_hduheader(self):
        """
        To retrieve the header of the current FITS image. This method accesses the header 
        information of the original FITS file, and then modifies it to reflect the current
        status of this image object.

        Returns:
            The FITS header of the current image object (astropy.io.fits.header.Header).
        """
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
        # update parameters before saving
        self.__updateparams()
        
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
                 semimajor=1., semiminor=1., pa=0., center=(0., 0.), 
                 unit="arcsec", start=None, end=None, length=None, 
                 relative=True, quiet=False):
        self.regionfile = regionfile
        shape = shape.lower()
        unit = unit.lower()
        isJ2000 = (isinstance(center, str) and not center.isnumeric()) \
                    or (len(center)==2 and isinstance(center[0], str) and isinstance(center[1], str)) 
        
        relative = False if isJ2000 else relative
        
        if length is not None or start is not None or end is not None:
            shape = "line"
        
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
              'font.family': 'Times New Roman',
              "mathtext.fontset": 'stix', #"Times New Roman"
              'mathtext.tt':' Times New Roman',
              'axes.linewidth': borderwidth,
              'xtick.major.width': borderwidth,
              'ytick.major.width': borderwidth,
#               'xtick.minor.width': 1.0,
#               'ytick.minor.width': 1.0,
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


def plt_spectrum(velocity=None, intensity=None, csvfile=None, xlim=[], ylim=[], 
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
              'font.family': 'Times New Roman',
              "mathtext.fontset": 'stix', #"Times New Roman"
              'mathtext.tt':' Times New Roman',
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
