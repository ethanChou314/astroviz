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


from .common import *


class Datacube:
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
            vmin, vmax = _clip_percentile(data=trimmed_data, area=percentile)
            
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
        from .spatialmap import Spatialmap

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
        
    def pvextractor(self, region, vrange=None, width=1, preview=True, 
                    parallel=True, **kwargs):
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
        from .pvdiagram import PVdiagram

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
        shifted_img = self.imshift(center, inplace=False, printcc=False, parallel=parallel) \
                      if center != (0, 0) else self.copy()
        rotated_img = shifted_img.rotate(angle=-pa_prime, ccw=False, inplace=False) \
                      if pa_prime != 0 else shifted_img
        
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
        from .spatialmap import Spatialmap

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
        from .spatialmap import Spatialmap
        from .pvdiagram import PVdiagram

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
        from .io import to_casa
        
        to_casa(self, *args, **kwargs)

