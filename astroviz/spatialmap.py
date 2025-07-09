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


from .common import *


class Spatialmap:
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
            vmin, vmax = _clip_percentile(data=clipped_data, area=percentile)
            
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