from .common import *


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
        from .datacube import Datacube
        return Datacube(header=header, data=data)
    elif imagetype == "spatialmap":
        from .spatialmap import Spatialmap
        return Spatialmap(header=header, data=data)
    elif imagetype == "pvdiagram":
        from .pvdiagram import PVdiagram
        return PVdiagram(header=header, data=data)
    else:
        raise Exception("Cannot determine image type.")


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
    from .datacube import Datacube
    from .spatialmap import Spatialmap
    from .pvdiagram import PVdiagram

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
    from .datacube import Datacube
    from .spatialmap import Spatialmap
    from .pvdiagram import PVdiagram
    
    if isinstance(image, (Datacube, Spatialmap, PVdiagram)):
        return image.imview(*args, **kwargs)
    elif isinstance(image, str):
        image = importfits(image)
        return image.imview(*args, **kwargs)
    else:
        raise ValueError("'image' parameter must be a directory to a FITS file or an image instance.")