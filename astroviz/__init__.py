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


# core classes
from .datacube import Datacube
from .spatialmap import Spatialmap
from .pvdiagram import PVdiagram
from .region import Region
from .imagematrix import ImageMatrix
from .plot2d import Plot2D

# molecular line analysis tools
from .molecules import (search_molecular_line, planck_function,
                        H2_column_density, J_v, 
                        column_density_linear_optically_thin,
                        column_density_linear_optically_thick)

# I/O functions
from .io import importfits, exportfits, to_casa, imview

# bring in any global config functions
from .global_config import set_font


# official public API
__all__ = [
    "Datacube",
    "Spatialmap",
    "PVdiagram",
    "Region",
    "ImageMatrix",
    "Plot2D",
    "importfits",
    "exportfits",
    "to_casa",
    "imview",
    "set_font",
    "search_molecular_line",
    "planck_function",
    "H2_column_density",
    "J_v",
    "column_density_linear_optically_thin",
    "column_density_linear_optically_thick",
]