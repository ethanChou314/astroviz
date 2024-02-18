# astroviz
The Python module, **astroviz**, is a comprehensive tool for data analysis and visualization in radio astronomy, including functionalities for importing FITS files; handling and manipulating data cubes, spatial maps, and position-velocity (PV) diagrams; and intuitive operations on image objects. It has been tested specifically for ALMA FITS files but may work on other wavelengths (not gauranteed).

**Requirements** 
* numpy
* astropy 
* matplotlib
* astroquery (only needed for .line_info() method)
* urllib (only needed for .line_info() method)
