# utils package (standard tools)
from .utils import (_unnormalize_location, _normalize_location, _change_cell_size, _trim_data, _plt_cmap, 
                   _prevent_overwriting, _plt_cbar_and_set_aspect_ratio, _find_file, _get_contour_beam_loc,
                   _true_round_whole, _get_hduheader, _icrs2relative, _customizable_scale,
                   _relative2icrs, _to_apu, _apu_to_str, _apu_to_headerstr, _best_match_line,
                   _unit_plt_str, _convert_Bunit, _Qrot_linear, _get_beam_dim_along_pa,
                   _Qrot_linear_polyatomic, _match_limits, _get_optimal_columns, _get_moment_map,
                   _clip_percentile, _best_match_line)

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
from typing import List, Dict, Union, Optional

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

# global configuration
from .global_config import _fontfamily, _mathtext_fontset, _mathtext_tt

# molecular analysis
from .molecules import (search_molecular_line, planck_function,
                        H2_column_density, J_v, 
                        column_density_linear_optically_thin,
                        column_density_linear_optically_thick)


__all__ = [
    # ---- utils ----
    "_unnormalize_location",
    "_normalize_location",
    "_change_cell_size",
    "_trim_data",
    "_plt_cmap",
    "_prevent_overwriting",
    "_plt_cbar_and_set_aspect_ratio",
    "_find_file",
    "_get_contour_beam_loc",
    "_true_round_whole",
    "_get_hduheader",
    "_icrs2relative",
    "_customizable_scale",
    "_relative2icrs",
    "_to_apu",
    "_apu_to_str",
    "_apu_to_headerstr",
    "_best_match_line",
    "_unit_plt_str",
    "_convert_Bunit",
    "_Qrot_linear",
    "_get_beam_dim_along_pa",
    "_Qrot_linear_polyatomic",
    "_match_limits",
    "_get_optimal_columns",
    "_get_moment_map",
    "_clip_percentile",

    # ---- numerical ----
    "np",
    "pd",

    # ---- standard ----
    "copy",
    "os",
    "sys",
    "dt",
    "string",
    "inspect",
    "warnings",
    "itertools",

    # ---- scientific ----
    "ndimage",
    "curve_fit",
    "linregress",
    "interpolate",
    "u",
    "const",
    "Unit",
    "def_unit",
    "UnitConversionError",
    "fits",
    "WCS",
    "sigma_clip",
    "Angle",
    "SkyCoord",
    "models",
    "fitting",
    "Gaussian2DKernel",
    "Box2DKernel",
    "convolve",
    "convolve_fft",

    # ---- parallel ----
    "ProcessPoolExecutor",

    # ---- visualization ----
    "mpl",
    "cm",
    "rcParams",
    "ticker",
    "patches",
    "colors",
    "plt",
    "ImageGrid",
    "inset_axes",

    # ---- global config ----
    "_fontfamily",
    "_mathtext_fontset",
    "_mathtext_tt",

    # ---- molecules ----
    "search_molecular_line",
    "planck_function",
    "H2_column_density",
    "J_v",
    "column_density_linear_optically_thin",
    "column_density_linear_optically_thick",

    # ---- typing ----
    "List", 
    "Dict", 
    "Union", 
    "Optional",
]
