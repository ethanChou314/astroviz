from .common import *


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
            vmin, vmax = _clip_percentile(data=trimmed_data, area=percentile)
        
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
        from .io import to_casa
        
        to_casa(self, *args, **kwargs)