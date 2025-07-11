"""
Utilities module for astroviz.
"""

# numerical packages
import numpy as np
import pandas as pd

# typing 
from typing import *
from numpy.typing import ArrayLike

# other standard modules
import copy
import os
import warnings
import datetime as dt

# scientific packages 
from astropy.io import fits
from astropy import units as u, constants as const
from astropy.units import Unit, def_unit, UnitConversionError
from astropy.coordinates import Angle, SkyCoord
from scipy.interpolate import griddata
from scipy import ndimage

# parallel processing
from concurrent.futures import ProcessPoolExecutor

# plotting packages 
import matplotlib as mpl
from matplotlib import cm, rcParams, ticker, patches, colors, pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _type_checker(vars: List[Any], names: List[str], types: List[type]) -> None:
    for v, n, t in zip(vars, names, types):
        if not isinstance(v, t):
            raise TypeError(f"Variable '{n}' is expected to be of type {t.__name__}, " + \
                            f"but got {type(v).__name__}.")


def _best_match_line(restfreq, species_id=None, return_table=False):
    # module
    from astroquery.splatalogue import Splatalogue

    # get best match results
    df = _casa_query_line([restfreq-1, restfreq+1])
    if return_table:
        return df
    frequencies = df["FREQUENCY"]
    errors = np.abs(frequencies-restfreq)
    least_err_position = np.argmin(errors)
    if not np.isclose(errors[least_err_position], 0.):
        warnings.warn("This image may not be of a molecular line. \n")
    line_info = dict(df.iloc[least_err_position])

    # search for species ID
    species = line_info["SPECIES"]
    chemical_name = line_info["CHEMICAL_NAME"]

    if species_id is None:
        if "v" in species and "=" in species:
                v_idx = species.rindex("v")
                name = species[:v_idx]
                version = species[v_idx:]
                version = " " + version.replace("=", " = ")
        else:
            name = species
            version = ""

        # do a first find
        species_ids = Splatalogue.get_species_ids()
        species_ids = species_ids.find(name)
        species_ids = species_ids.find(chemical_name)

        if len(species_ids) == 1:
            species_id = tuple(species_ids.values())[0]
        else:
            # do additional search (of versions) if needed
            species_keys = tuple(key[6:] for key in species_ids.keys())
            organized_species_ids = dict(zip(species_keys, species_ids.values()))
            search_key = name + version + " - " + chemical_name

            try:
                species_id = organized_species_ids[search_key]
            except KeyError:
                print("Failed to find species ID.")
                print("Possible species IDs:")
                for key, value in organized_species_ids.items():
                    print(f"[{value}] {key}")
                species_id = None

    line_info["SPECIES_ID"] = species_id

    return line_info


def _casa_query_line(freqrange: List) -> pd.DataFrame:
    # modules
    from casatasks import slsearch
    from casatools import table 
    import tempfile

    with tempfile.TemporaryDirectory() as parent_temp_dir:
        # Generate a unique directory path inside the parent directory
        temp_dir = os.path.join(parent_temp_dir, 'casa_temp_table')

        # Perform the spectral line search and output to the unique directory
        slsearch(freqrange=freqrange, outfile=temp_dir)

        # Open the CASA table using the table tool
        tb = table()
        tb.open(temp_dir, nomodify=False)

        # Get column names and data from the table
        columns = tb.colnames()
        data_in_col = {col: tb.getcol(col) for col in columns}

        # Convert to pandas DataFrame
        df = pd.DataFrame(data_in_col)

        # Close the table
        tb.close()

    return df


def _normalize_location(coord, xlim, ylim):
    normalized_x = (coord[0]-xlim[0])/(xlim[1] - xlim[0])
    normalized_y = (coord[1]-ylim[0])/(ylim[1] - ylim[0])
    return (normalized_x, normalized_y)


def _unnormalize_location(coord, xlim, ylim):
    unnormalized_x = coord[0]*(xlim[1]-xlim[0]) + xlim[0]
    unnormalized_y = coord[1]*(ylim[1]-ylim[0]) + ylim[0]
    return (unnormalized_x, unnormalized_y)


def _get_contour_beam_loc(color_beam_dims, contour_beam_dims, 
                          color_beamloc, xlim, beam_separation=0.04):
    hor_color_beam_dim = abs(_get_beam_dim_along_pa(*color_beam_dims, pa=90))
    hor_contour_beam_dim = abs(_get_beam_dim_along_pa(*contour_beam_dims, pa=90))
    norm_color_beam_dim = hor_color_beam_dim / (max(xlim)-min(xlim))
    norm_contour_beam_dim = hor_contour_beam_dim / (max(xlim)-min(xlim))
    x = color_beamloc[0] + norm_color_beam_dim/2 + beam_separation + norm_contour_beam_dim/2
    y = color_beamloc[1]
    return (x, y)

def _get_beam_dim_along_pa(bmaj, bmin, bpa, pa):
    """
    Calculates the beam dimension along a specified position angle.

    Args:
        bmaj (float): The major axis of the beam (FWHM) in the same units as `bmin`.
        bmin (float): The minor axis of the beam (FWHM) in the same units as `bmaj`.
        bpa (float): The position angle of the beam in degrees, measured counter-clockwise from the north.
        pa (float): The position angle along which the beam dimension is to be calculated, in degrees, measured counter-clockwise from the north.

    Returns:
        float: The effective beam dimension along the specified position angle `pa`.

    Notes:
        The position angle `pa` and the beam position angle `bpa` are given in degrees and should be measured counter-clockwise from the north.
        This calculation assumes that the beam shape is elliptical.

    Example:
        >>> bmaj = 2.5
        >>> bmin = 1.5
        >>> bpa = 45
        >>> pa = 60
        >>> _get_beam_dim_along_pa(bmaj, bmin, bpa, pa)
        1.750219722145759
    """
    angle = np.deg2rad(pa-bpa)
    aa = np.square(np.sin(angle)/bmin)
    bb = np.square(np.cos(angle)/bmaj)
    return np.sqrt(1/(aa+bb))

def _is_prime(n):
    """
    Helper function to determine if a number is prime.

    This helper function checks if the input number `n` is a prime number.
    A prime number is only divisible by 1 and itself, and has no other divisors.

    Parameters:
    n (int): The number to be checked for primality.

    Returns:
    bool: True if `n` is a prime number, False otherwise.
    """
    # the square root of n (rounded down to the nearest integer)
    for i in range(2, int(n**0.5)+1):
        # If n is divisible by any of these numbers, return False
        if n % i == 0:
            return False
    # If n is not divisible by any of these numbers, return True
    return True


def _get_optimal_columns(n, min_col=5, max_col=10):
    """
    Calculate the optimal number of columns for displaying a given number of items.

    This function determines the optimal number of columns based on the number
    of items `n` to be displayed, ensuring that the number of columns falls
    within the specified range of `min_col` to `max_col`. If `n` is a small
    number (less than or equal to 7), it simply returns `n`. If `n` is prime,
    it approximates the square root of `n` as the number of columns. If `n` is
    not prime, it finds the largest divisor of `n` within the range.

    Parameters:
    n (int): The number of items to be displayed.
    min_col (int, optional): The minimum number of columns allowed. Default is 5.
    max_col (int, optional): The maximum number of columns allowed. Default is 10.

    Returns:
    int: The optimal number of columns for displaying `n` items.
    """

    if n <= 7:
        return n
    if _is_prime(n):
        columns = int(n**0.5)
        if columns * (columns + 1) < n:
            columns += 1
        if columns < min_col:
            columns = min_col
        elif columns > max_col:
            columns = max_col
        return columns
        
    for i in range(n, 0, -1):
        if n % i == 0 and i <= max_col:
            columns = i
            break
    
    if columns < min_col:
        columns = min_col
            
    return columns


def _match_limits(image, template_image, inplace=True):
    new_shape = (1, image.shape[1], template_image.shape[2], template_image.shape[3])
    dx = image.dx
    dy = image.dy
    old_xlim = [image.xaxis.min(), image.xaxis.max()]
    old_ylim = [image.yaxis.min(), image.yaxis.max()]
    new_xlim = [template_image.xaxis.min(), template_image.xaxis.max()]
    new_ylim = [template_image.yaxis.min(), template_image.yaxis.max()]
    
    pix_below = _true_round_whole((old_ylim[0] - new_ylim[0]) / dy)
    pix_above = _true_round_whole((new_ylim[1] - old_ylim[1]) / dy)
    pix_left = _true_round_whole((new_xlim[1] - old_xlim[1]) / -dx)
    pix_right = _true_round_whole((old_xlim[0] - new_xlim[0]) / -dx)
    
    # initialize
    new_data = image.data.copy()  
   
    print("Matching limits of spatial coordinates...", end="")
    
     # modify top pixels
    pad_width = [0, 0, 0, 0]  # (bottom, top, left, right)
    if pix_above > 0:  # pad
        pad_width[1] = pix_above
    else:  # trim
        start_idx = abs(pix_above)
        new_data = new_data[:, :, start_idx:, :]
        
    # bottom pixels
    if pix_below > 0:  # pad
        pad_width[0] = pix_below
    else:  # trim
        end_idx = new_data.shape[2] - abs(pix_below)
        new_data = new_data[:, :, :end_idx, :]
        
    # modify left pixels
    if pix_left > 0:  # pad
        pad_width[3] = pix_left
    else:
        end_idx = new_data.shape[3] - abs(pix_left)
        new_data = new_data[:, :, :, :end_idx]
    
    # modify right pixels
    if pix_right > 0:  # pad
        pad_width[2] = pix_right
    else:
        start_idx = abs(pix_right)
        new_data = new_data[:, :, :, start_idx:]
    
    # pad data
    print(pad_width)
    pad_width = ((0, 0), tuple(pad_width[:2]), tuple(pad_width[2:]))
    print(pad_width)
    new_data = np.pad(new_data[0], pad_width=pad_width,
                      mode="constant", constant_values=np.nan)
    new_data = new_data[np.newaxis, :, :, :]
    
    print("done!")
            
    new_image = image if inplace else image.copy()
    new_image.data = new_data
    new_nx, new_ny = new_shape[2:]
    new_refnx = new_nx // 2 + 1
    new_refny = new_ny // 2 + 1
    
    new_image.overwrite_header(shape=new_shape,
                               nx=new_nx,
                               refnx=new_refnx,
                               ny=new_ny,
                               refny=new_refny
                               )
    return new_image


def _customizable_scale(ax, xdata, ydata, 
                        scale=("linear", "linear"),
                        xticks=None, yticks=None, 
                        plot_type="scatter",
                        linewidth=None, linestyle=None,
                        linecolor=None, label=None,
                        marker=None, color=None,
                        plot_ebars=False,
                        linear_ticks=True,
                        **kwargs):
    if isinstance(scale, str):
        scale = (scale, scale)

    xscale, yscale = scale
    
    # make parameters case-insensitive
    if not xscale.islower():
        xscale = xscale.lower()
    
    if not yscale.islower():
        yscale = yscale.lower()
        
    if not plot_type.islower():
        plot_type = plot_type.lower()
        
    # first plot scale in linear
    if plot_type in ("line", "linear"):
        ax.plot(xdata, ydata, lw=linewidth, ls=linestyle, 
                color=linecolor, label=label, **kwargs)
    elif plot_type == "scatter":
        if plt_ebars:
            ax.scatter(xdata, ydata, marker=marker, color=color, **kwargs)
        else:
            ax.scatter(xdata, ydata, marker=marker, color=color, 
                       label=label, **kwargs)
    else:
        raise ValueError("'plot_type' must be 'linear' or 'scatter'. " + \
                         "For histograms, set 'interpolation' to 'nearest'.")
    
    # get current ticks (in linear scale)
    if xticks is None:
        xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()
        
    if yticks is None:
        yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()
        
    print("xscale:", xscale)
    print("yscale:", yscale)

    # change scale
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    # keep ticks
    if linear_ticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    
    return ax


# helper function used for parallel processing
def _regrid_parallel_helper(data, old_xx, old_yy, xi, mask, method):
    points = np.array([old_yy[mask].ravel(), old_xx[mask].ravel()]).T
    values = data[mask].ravel()
    return griddata(points=points, values=values, xi=xi, method=method)


def _change_cell_size(image, dx, dy, interpolation="linear", parallel=True, inplace=False):
    # create copy if user doesn't want to modify image in-place
    image = image if inplace else image.copy()

    # get attributes
    nchan = image.shape[1]  # this will not chnage
    old_data = image.data
    old_xaxis = image.xaxis
    old_yaxis = image.yaxis
    old_xx, old_yy = image.get_xyaxes(grid=True)

    # parse cell sizes
    if dx > 0:
        dx = -dx  # dx must be negative
    if dy < 0:
        dy = -dy  # dy must be positive

    # calculate new shape
    x_first, x_last = image.xaxis[[0, -1]]
    y_first, y_last = image.yaxis[[0, -1]]

    # new parameters
    new_nx = _true_round_whole((x_last-x_first)/dx)
    new_ny = _true_round_whole((y_last-y_first)/dy)
    new_refnx = new_nx // 2 + 1
    new_refny = new_ny // 2 + 1

    new_xaxis = np.linspace(x_first, x_last, new_nx)
    new_yaxis = np.linspace(y_first, y_last, new_ny)
    new_xx, new_yy = np.meshgrid(new_xaxis, new_yaxis)  # create new grid
    new_shape = (1, nchan, new_ny, new_nx) # calculate new shape
   
    # start regridding
    print("Creating new grid...", end="")
    mask = ~np.isnan(image.data[0])
    
    if parallel and nchan > 1: 
        xi = (new_yy, new_xx)
        with ProcessPoolExecutor() as executor:  # execute parallel processing
            results = list(executor.map(_regrid_parallel_helper, 
                                        (old_data[0, i] for i in range(nchan)),
                                        (old_xx for _ in range(nchan)),
                                        (old_yy for _ in range(nchan)),
                                        (xi for _ in range(nchan)),
                                        (mask[i] for i in range(nchan)),
                                        (interpolation for _ in range(nchan))
                                        )
                          )
        new_data = np.array(results)[np.newaxis]

    else:
        new_data = np.empty(new_shape)  # initialize
        xi = (new_yy, new_xx) 
        for i in range(nchan):  # i = 0 only, if 'image' is a spatialmap, without any vaxis.
            points = np.array([old_yy[mask[i]].ravel(), old_xx[mask[i]].ravel()]).T
            values = image.data[0, i][mask[i]].ravel()
            new_data[0, i] = griddata(points=points, values=values,
                                      xi=xi, method=interpolation)

    print("done!")

    # update data
    image.data = new_data

    # update header
    image.overwrite_header(dx=dx, dy=dy, nx=new_nx, ny=new_ny, 
                           refnx=new_refnx, refny=new_refny,
                           shape=new_shape)
    
    return image
    

def _trim_data(four_dim_data, yaxis, xaxis, vaxis=None, dy=0, dx=0, dv=0,
               vlim=None, ylim=None, xlim=None, vsys=None, nskip=None, vskip=None):
    """
    Trim a four-dimensional dataset based on specified limits for the x, y, and velocity axes.

    Parameters:
    four_dim_data (np.ndarray): The 4D dataset to be trimmed. The dimensions are expected to be
                                in the order of (istokes, vaxis, yaxis, xaxis).
    vaxis (np.ndarray): The velocity axis values.
    yaxis (np.ndarray): The y-axis values.
    xaxis (np.ndarray): The x-axis values.
    dv (float): The velocity increment. Used to slightly extend the trimming bounds to prevent
                floating point errors.
    dy (float): The y-axis increment. Used to slightly extend the trimming bounds to prevent
                floating point errors.
    dx (float): The x-axis increment. Used to slightly extend the trimming bounds to prevent
                floating point errors.
    vlim (iterable of length 2, optional): The limits for the velocity axis to trim the data. 
                                           If None, no trimming is performed on the velocity axis.
    ylim (iterable of length 2, optional): The limits for the y-axis to trim the data. 
                                           If None, no trimming is performed on the y-axis.
    xlim (iterable of length 2, optional): The limits for the x-axis to trim the data. 
                                           If None, no trimming is performed on the x-axis.
    vsys (float, optional.): The systemic velocity. vlim will be added by this value. 
    
    Returns:
    tuple: A tuple containing the trimmed data and the corresponding trimmed axes:
           (trimmed_data, trimmed_vaxis, trimmed_yaxis, trimmed_xaxis).

    Raises:
    ValueError: If any of `xlim`, `ylim`, or `vlim` are provided and are not iterables of length 2.

    Notes:
    The trimming bounds are slightly extended by 10% of the corresponding increments (`dv`, `dy`, `dx`)
    to prevent issues related to floating point errors.

    """
    # parse parameters:
    if vlim is not None:
        if not isinstance(vlim, np.ndarray):
            if not hasattr(vlim, "__iter__"):
                raise ValueError("'vlim' must be an iterable of length 2.")
            vlim = np.array(vlim)
            if vlim.ndim > 1:
                vlim = vlim.flatten()
            if vlim.size != 2:
                raise ValueError("'vlim' must be an iterable of length 2.")
        vmin = vlim.min() - 0.1*abs(dv)  # minimum velocity (minus small value to prevent floating point error)
        vmax = vlim.max() + 0.1*abs(dv)  # maximum velocity (adding small value to prevent floating point error)
        if vsys:
            vmin += vsys
            vmax += vsys

    if ylim is not None:
        if not isinstance(ylim, np.ndarray):
            if not hasattr(ylim, "__iter__"):
                raise ValueError("'ylim' must be an iterable of length 2.")
            ylim = np.array(ylim)
            if ylim.ndim > 1:
                ylim = ylim.flatten()
            if ylim.size != 2:
                raise ValueError("'ylim' must be an iterable of length 2.")
        ymin = ylim.min() - 0.1*abs(dy)  # minimum y value (minus small value to prevent floating point error)
        ymax = ylim.max() + 0.1*abs(dy)  # maximum y value (adding small value to prevent floating point error)

    if xlim is not None:
        if not isinstance(xlim, np.ndarray):
            if not hasattr(xlim, "__iter__"):
                raise ValueError("'xlim' must be an iterable of length 2.")
            xlim = np.array(xlim)
            if xlim.ndim > 1:
                xlim = xlim.flatten()
            if xlim.size != 2:
                raise ValueError("'xlim' must be an iterable of length 2.")
        xmin = xlim.min() - 0.1*abs(dx)  # minimum x value (minus small value to prevent floating point error)
        xmax = xlim.max() + 0.1*abs(dx)  # maximum x value (adding small value to prevent floating point error)

    # initialize arrays:
    trimmed_data = four_dim_data
    if vaxis is None:
        trimmed_vaxis = None
    else:
        trimmed_vaxis = vaxis
        if nskip is not None:
            trimmed_data = trimmed_data[:, ::nskip, :, :]
            trimmed_vaxis = trimmed_vaxis[::nskip]
        elif vskip is not None:
            nskip = _true_round_whole(vskip/dv)
            trimmed_data = trimmed_data[:, ::nskip, :, :]
            trimmed_vaxis = trimmed_vaxis[::nskip]
    trimmed_xaxis = xaxis
    trimmed_yaxis = yaxis

    # create masks:
    if vlim is not None:
        vmask = (vmin <= vaxis) & (vaxis <= vmax)
        trimmed_vaxis = trimmed_vaxis[vmask]
        trimmed_data = trimmed_data[:, vmask, :, :]

    if ylim is not None:
        ymask = (ymin <= yaxis) & (yaxis <= ymax)
        trimmed_yaxis = trimmed_yaxis[ymask]
        trimmed_data = trimmed_data[:, :, ymask, :]

    if xlim is not None:
        xmask = (xmin <= xaxis) & (xaxis <= xmax)
        trimmed_xaxis = trimmed_xaxis[xmask]
        trimmed_data = trimmed_data[:, :, :, xmask]

    return trimmed_data, trimmed_vaxis, trimmed_yaxis, trimmed_xaxis


def _plt_cmap(image_obj, ax, two_dim_data, imextent, cmap, 
              vmin=None, vmax=None, scale="linear", gamma=1.5,
              linthresh=1, linscale=1):
    """
    Plots a color map on the given axes with specified scaling and color mapping.

    Parameters:
    - image_obj (object): The image object containing metadata and methods such as noise and bunit.
    - ax (matplotlib.axes.Axes): The axes on which to plot the color map.
    - two_dim_data (array-like): The 2D data array to be plotted.
    - imextent (tuple): The bounding box in data coordinates (left, right, bottom, top).
    - cmap (str or Colormap): The colormap used for mapping data values to colors.
    - vmin (float, optional): The minimum data value that corresponds to the colormap's minimum color. 
                              Default is None.
    - vmax (float, optional): The maximum data value that corresponds to the colormap's maximum color. 
                              Default is None (maximum value of dataset).
    - scale (str, optional): The scale to use for mapping data values to colors. 
                             Options are 'linear', 'log', 'symlog', or 'gamma'. Default is 'linear'.
    - gamma (float, optional): The gamma value to use if scale is 'gamma'. Default is 1.5.
    - linthresh (float, optional): The linear range (-linthresh to linthresh) to use for 'symlog' scale. Default is 1.0.
    - linscale (float, optional): The scale factor for the linear range in 'symlog' scale. Default is 1.0.

    Description:
    - Converts the scale to lowercase to ensure case insensitivity.
    - For 'linear' scale, plots the data using a linear color scale.
    - For 'log' scale:
      - Computes `vmin` and `vmax` if not provided, ensuring they are positive.
      - Raises an exception if data contains only non-positive values.
      - Provides warnings and recommendations if `vmin` is adjusted.
    - For 'symlog' scale, plots the data using a symmetrical logarithmic scale with the specified linear threshold and scale.
    - For 'gamma' scale, plots the data using a gamma color scale with the specified gamma value.
    - Raises a ValueError if the specified scale is not 'linear', 'log', 'symlog', or 'gamma'.

    Raises:
    - ValueError: If the specified scale is not 'linear', 'log', 'symlog', or 'gamma'.
    - Exception: If data only contains non-positive values when 'log' scale is used, 
                 or if `vmin` is not set appropriately for 'log' scale.

    Notes:
    - For 'log' scale, `vmin` and `vmax` must be positive. 
    - If `vmin` is not provided and the minimum value of data is non-positive, 
      it defaults to three times the estimated noise.
    - Provides warnings and recommendations if `vmin` is adjusted for 'log' scale.
    - For 'symlog' scale, `linthresh` sets the range within which the data is scaled linearly, 
      and `linscale` adjusts the size of the linear region.
    """
    # parse scale:
    if not scale.islower():
        scale = scale.lower  # make it case-insensitive
        
    # create normalization scales:
    if scale == "linear":
        # plot color map:
        climage = ax.imshow(two_dim_data, cmap=cmap, extent=imextent, origin="lower", 
                            vmin=vmin, vmax=vmax)
    elif scale in ("log", "logscale", "logarithm"):
        # compute vmax and check if log scale is suitable:
        if vmax is None:
            vmax = np.nanmax(two_dim_data)
            if vmax <= 0:
                raise Exception("The data only contains negative values, " + \
                                "so log scale is not suitable.")

        # compute vmin and check if log scale is suitable:
        if vmin is None:
            vmin = np.nanmin(two_dim_data)
            if vmin <= 0:
                vmin = 3*image_obj.noise()
                warnings.warn("Default minimum value is negative, " + \
                              "which is not allowed for log scale. \n The vmin will be assigned " + \
                              f"as three times the estimated rms: {vmin} [{image_obj.bunit}] \n" + \
                              "You are recommended to set the minimum value.")
                
        # create log scale norm object:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        # plot color map:
        climage = ax.imshow(two_dim_data, cmap=cmap, extent=imextent, 
                            norm=norm, origin="lower")
    elif scale == "symlog":
        # create symetric log scale norm object:
        norm = colors.SymLogNorm(linthresh=linthresh, linscale=linscale, 
                                 vmin=vmin, vmax=vmax)
        # plot color map:
        climage = ax.imshow(two_dim_data, cmap=cmap, extent=imextent, 
                            norm=norm, origin="lower")
    elif scale in ("gamma", "power"):
        # create gamma (power) scale norm object:
        norm = colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

        # plot color map:
        climage = ax.imshow(two_dim_data, cmap=cmap, extent=imextent, 
                            norm=norm, origin="lower")
    elif scale in ("symgamma", "symmetric gamma", "sympower", "symmetrical gamma"):
        # create symmetric gamma scale norm object (custom-made):
        forward_function = lambda x: np.sign(x) * np.abs(x) ** gamma
        reverse_function = lambda y: np.sign(y) * np.abs(y) ** (1/gamma)
        norm = colors.FuncNorm(functions=(forward_function, reverse_function), 
                               vmin=vmin, vmax=vmax)

        # plot color map:
        climage = ax.imshow(two_dim_data, cmap=cmap, extent=imextent, 
                            norm=norm, origin="lower")
    else:
        raise ValueError("Scale must be 'linear', 'symlog', 'log', 'gamma', or 'symlog'.")
        
    return ax, climage


def _prevent_overwriting(directory: str) -> str:
    """
    Modify the file path to prevent overwriting an existing file by appending a number to the filename.

    Parameters:
        directory (str): The original file path.

    Returns:
        str: The modified file path with a number appended if the file already exists.
    """
    # Split the directory into base and extension
    base, file_extension = os.path.splitext(directory)
    
    # If the file already exists, append a number to the filename
    if os.path.exists(directory):
        i = 1
        while True:
            new_directory = f"{base}({i}){file_extension}"
            if not os.path.exists(new_directory):
                return new_directory
            i += 1
    
    # If the file does not exist, return the original directory
    return directory


def _plt_cbar_and_set_aspect_ratio(ax, climage, cbarlabelon=True, cbarlabel="", cbarloc="right", 
                                   cbaron=True, cbarwidth="5%", cbarpad=0., cbarticks=None, 
                                   fontsize=10., labelsize=10., cbartick_width=1., cbartick_length=1., 
                                   cbartick_direction="in", aspect_ratio=1., is_logscale=False, 
                                   decimals=2):
    """
    Adds a color bar to the plot and sets the aspect ratio.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to which the color bar will be added.
        climage (matplotlib.image.AxesImage): The image object to which the color bar corresponds.
        cbarlabelon (bool): Flag to display the label on the color bar. Default is True.
        cbarlabel (str): Label for the color bar. Default is an empty string.
        cbarloc (str): Location of the color bar ('right' or 'top'). Default is 'right'.
        cbarwidth (str): Width of the color bar as a percentage. Default is '5%'.
        cbarpad (float or str): Padding between the color bar and the image. Can be a float or a string with a percentage. Default is 0.
        cbarticks (list, optional): List of tick positions for the color bar. Default is None.
        fontsize (float): Font size for the color bar label. Default is 10.
        labelsize (float): Font size for the color bar ticks. Default is 10.
        cbartick_width (float): Width of the color bar ticks. Default is 1.
        cbartick_length (float): Length of the color bar ticks. Default is 1.
        cbartick_direction (str): Direction of the color bar ticks ('in' or 'out'). Default is 'in'.
        aspect_ratio (float): Desired aspect ratio for the plot. Default is 1.
        is_logscale (bool): Flag to indicate if the color bar scale is logarithmic. Default is False.
        decimals (int): Number of decimal places for tick labels if log scale is used. Default is 2.

    Returns:
        ax (matplotlib.axes.Axes): The modified axes object with the color bar added and aspect ratio set.
    """
    # get horizontal and vertical limits:
    horizontal_limit = ax.get_xlim()
    horizontal_range = horizontal_limit[1] - horizontal_limit[0]
    vertical_limit = ax.get_ylim()
    vertical_range = vertical_limit[1] - vertical_limit[0]
    
    # adjust width and pad to match aspect ratio needed
    if aspect_ratio:
        real_aspect_ratio = abs(1./aspect_ratio*horizontal_range/vertical_range)
        ax.set_aspect(real_aspect_ratio)
        
    if not cbaron:
        return ax  # early stopping if user does not wish to plot color bar
    
    # check and adjust parameters:
    if not cbarloc.islower():  # make it case-insensitive
        cbarloc = cbarloc.lower()
    
    # convert to float so that calculation can be done 
    if isinstance(cbarpad, str):
        cbarpad = float(cbarpad.strip("%")) * 0.01  
    
    # determine orientation, width, and height based on location of the color bar
    if cbarloc == "right":
        orientation = "vertical"
        width = cbarwidth
        height = "100%"
        bbox_to_anchor = (1.+cbarpad, 0., 1., 1.)
    elif cbarloc == "top":
        orientation = "horizontal"
        width = "100%"
        height = cbarwidth
        bbox_to_anchor = (0, 1.+cbarpad, 1., 1.)
    else:
        raise ValueError("'cbarloc' must be either 'right' or 'top'.")
    
    # create color bar using 'inset axes'
    ax_cb = inset_axes(ax, width=width, height=height, 
                       loc="lower left", borderpad=0.,
                       bbox_to_anchor=bbox_to_anchor, 
                       bbox_transform=ax.transAxes)
    
    # plot color bar -> creates mappable instance
    cbar = plt.colorbar(climage, cax=ax_cb, pad=cbarpad, 
                        orientation=orientation, ticklocation=cbarloc)
    
    # set label on color bar
    if cbarlabelon and cbarlabel:
        cbar.set_label(cbarlabel, fontsize=fontsize)
        
    # set ticks and tick parameters
    if is_logscale and (len(cbarticks) == 0 or cbarticks is None):
        vmin, vmax = climage.get_clim()
        cbarticks = np.linspace(vmin, vmax, 7)[1:-1]
    
    # color bar ticks
    if len(cbarticks) != 0:
        cbar.set_ticks(cbarticks)
        if is_logscale:
            labels = (f"%.{decimals}f"%label for label in cbarticks)  # generator object
            labels = [label[:-1] if label.endswith(".") else label for label in labels]
            cbar.set_ticklabels(labels)
        
    # adjust color bar parameters
    cbar.ax.tick_params(labelsize=labelsize, width=cbartick_width, 
                        length=cbartick_length, direction=cbartick_direction)
    
    # disable minor ticks
    cbar.ax.minorticks_off()

    return ax


def _find_file(file: str) -> Union[str, None]: 
    """
    Private function that finds the file location in a relative directory.
    Parameters:
        file (str): directory in another subdirectory (not absolute directory)
    Returns:
        Absolute directory if found. None if 
    """
    for root, dirs, files in os.walk(os.getcwd()):
        if file in files:
            return os.path.join(root, file)
    return None 


def _true_round_whole(x: Union[float, ArrayLike]) -> Union[int, np.ndarray]:
    """
    Private function that performs true rounding 
    (always rounds >=0.5 up and <0.5 down).
    ------
    Parameters:
        x (float/ArrayLike[float]): the number/array to be rounded
    Returns:
        rounded_x (float/ArrayLike[int]): the rounded number/array
    """
    x = np.array(x)  # always convert to array first
    rounded_x: np.ndarray = np.where(x<0, np.ceil(x-0.5), np.floor(x+0.5))  # correct rounding logic
    if rounded_x.size == 1:
        return rounded_x.astype(int).item()  # return number if input is number
    return rounded_x.astype(int)  # return array if input is array


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
    if image.header["filepath"]:
        if os.path.isfile(image.header["filepath"]):
            fitsfile = fits.open(image.header["filepath"])[0]
            hdu_header = copy.deepcopy(fitsfile.header)
        elif os.path.exists(image.header["filepath"]):
            import casatasks
            temp_fitsname = str(dt.datetime.now()).replace(" ", "") + ".fits"
            temp_fitsname = _prevent_overwriting(temp_fitsname)
            casatasks.exportfits(image.header["filepath"], temp_fitsname, history=False)
            fitsfile = fits.open(temp_fitsname)[0]
            os.remove(temp_fitsname)
            hdu_header = copy.deepcopy(fitsfile.header)
        else:
            hdu_header = fits.Header()
    else:
        hdu_header = fits.Header()  # create empty header if file path cannot be located
    
    # get info from current image object
    dx = u.Quantity(image.dx, image.unit).to_value(u.deg)
    if image.header["dy"] is not None:
        dy = u.Quantity(image.dy, image.unit).to_value(u.deg)
    if image.refcoord is not None:
        center = _icrs2relative(image.refcoord, unit="deg")
        centerx, centery = center[0].value, center[1].value
    else:
        centerx, centery = None, None
    projection = image.header["projection"]
    dtnow = str(dt.datetime.now()).replace(" ", "T")
    
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
                         "DATE": dtnow,
                         "DATAMAX": np.float64(np.nanmax(image.data)),
                         "DATAMIN": np.float64(np.nanmin(image.data)),
                         "BSCALE": np.float64(1.),
                         "BZERO": np.float64(1.),
                         "OBJECT": image.header["object"],
                         "INSTRUME": image.header["instrument"],
                         "DATE-OBS": image.header["obsdate"],
                         "RESTFRQ": image.header["restfreq"],
                         "ORIGIN": image.header["origin"],
                         }
    else:
        bmaj = np.float64(u.Quantity(image.bmaj, image.unit).to_value(u.deg))   # this value can be NaN
        bmin = np.float64(u.Quantity(image.bmin, image.unit).to_value(u.deg))
        bpa = np.float64(image.bpa)
        updatedparams = {"BUNIT": image.header["bunit"],
                         "DATE": dtnow,
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
                         "ORIGIN": image.header["origin"],
                         }
    
    # write history
    if "HISTORY" in hdu_header:
        if hdu_header["HISTORY"][-1] != "Exported from astroviz.":
            hdu_header.add_history("Exported from astroviz.")
    else:
        hdu_header["HISTORY"] = "Exported from astroviz."
    
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
    if unit == u.arcmin:  # special case
        return "arcmin"
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
    # deal with special case (having solar mass/luminosity as unit)
    headerstr = headerstr.replace("_{\\odot}", "sun")
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
    mathstr = mathstr.replace(r"_{\odot}$", r"$_{\odot}$")
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


def _Qrot_linear(T, B0):
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
        warnings.warn("Approximation error of partition function is greater than 1%.")
    
    # linear diatomic molecules
    Qrot = k*T/h/B0 + 1/3 + 1/15*(h*B0/k/T) + 4/315*(h*B0/k/T)**2 + 1/315*(h*B0/k/T)**3
    return Qrot.cgs.value


def _Qrot_linear_polyatomic(T, B0):
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
        warnings.warn("Approximation error of partition function is greater than 1%.")
    
    # linear diatomic molecules
    Qrot = k*T/h/B0 * np.exp(h*B0/3/k/T)
    return Qrot.cgs.value


def _get_moment_map(moment: int, data: np.ndarray, 
                    vaxis: np.ndarray, ny: int, 
                    nx: int, keep_nan: bool,
                    bunit: str, specunit: str,
                    header: dict) -> np.ndarray:
    """
    Private method to get the data array of the specified moment map.
    Used for the public method 'immoments'.
    """
    from .spatialmap import Spatialmap


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


def _clip_percentile(data, area=0.95):
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
