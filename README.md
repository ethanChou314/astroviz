# astroviz
The Python module, **astroviz**, is a comprehensive tool for data analysis and visualization in radio astronomy, including functionalities for importing FITS files; handling and manipulating data cubes, spatial maps, and position-velocity (PV) diagrams; and intuitive operations on image objects. It has been tested specifically for ALMA FITS files and may work on other wavelengths (not gauranteed). It is designed to be intuitive-to-use and flexible.

**Requirements** 
* numpy
* astropy 
* matplotlib
* astroquery (only needed for .line_info() method)
* urllib (only needed for .line_info() method)

# Examples
FITS files can be read using the 'importfits' function or via the constructors of the 'PVdiagram', 'Spatialmap,' and 'Datacube' classes. Astroviz is capable of reading data up to three dimensions (PV Diagrams, Continuum Maps, Moment Maps, and Data Cubes). Here are some different ways of importing a FITS file:
```python
import astroviz as av

# importing via 'importfits' function (adapts to whichever image type the FITS file contains):
datacube = av.importfits("test_fitsfiles/datacube.fits")
continuum = av.importfits("test_fitsfiles/continuum.fits")
pv = av.importfits("test_fitsfiles/pvdiagram.fits")

# importing via constructors:
datacube = av.Datacube("test_fitsfiles/datacube.fits")
continuum = av.Spatialmap("test_fitsfiles/continuum.fits")
pv = av.PVdiagram("test_fitsfiles/pvdiagram.fits")

# it still works even if the directory exists in a subdirectory! 
pv = av.importfits("pvdiagram.fits")
```

Then, you can do a lot with these instances. For example, you can view the header information:
```python
header = datacube.header  # this returns a dictionary, which is a summary of the full header
hdu_header = datacube.get_hduheader()  # this returns the full header
```

Viewing images is extremely simple! Check the docstrings for complete documentation of the parameters.
```python
# Fonts can be set globally via the 'set_font' function. The default is Times New Roman.
av.set_font("Arial")

# Examples of viewing images:
continuum.imview(fov=2, distance=360)
datacube.imview(ncol=8, nskip=5)
pv.imview(vsys=1.4, vlim=[0, 2.5])
```

The 'Region' class allows users to import a region via its constructor, which can then be used in various methods of the other instances. Users can also specify a DS9 directory to be imported.
```python
# View what the region on the map looks like:
ellipse = av.Region(center=(1, 3), semimajor=0.3, semiminor=0.1)
datacube.view_region(ellipse)

# Extracting PV diagrams:
line = av.Region("test_fitsfiles/line")  # supports reading of DS9 regions.
extracted_pv = datacube.pvextractor(line)  # extract the pv diagram

# Performing 2D ellipcial Gaussian fitting is really simple with the 'imfit' method:
circle = av.Region(center=(0, 0), radius=1)
continuum.imfit(circle, plt_residual=True)  # it can help you plot the residual!

# View statistics in that region such as extrema, RMS, total flux, etc.:
continuum.imstat(circle)

# extract spectral profiles from data cubes:
aperture = av.Region(center=(0, 0), length=5, width=3, pa=15, shape="box")  # create a box
datacube.view_region(aperture, dpi=100, fov=5, ncol=10, nskip=3)  # see what it looks like on the map

# Masking regions on 'Spatialmaps' and 'Datacube' instances:
ellipse = av.Region(center=(0, 0), semimajor=1, semiminor=0.1)
continuum.mask_region(ellipse, exclude=True, fov=2)

# Viewing region information:
print(region.header)
```

Unit conversion is also really simple, supporting various equivalencies (e.g., intensity to brightness temperature, Jy/pixel to Jy/beam, etc.):
```python
# convert intensity units:
continuum_bt = continuum.conv_bunit("K")  # equivalencies of brightness temperature/pixel/beam is supported!

# other units like integrated intensity also work!
moment_map = moment_map.conv_bunit("K.km/s")  # now in K.km/s
moment_map = moment_map.conv_bunit("Jy/sr.m/s")  # can directly convert as long as the units are equivalent!

# convert spatial units:
continuum_bt = continuum_bt.conv_unit("arcmin")

# convert spectral units:
pv_mps = pv.conv_specunit("m/s")
```

Noise estimation (with iterative sigma clipping) works on all image objects:
```python
# The .noise() method returns the RMS noise level:
rms = continuum.noise(printstats=True,  # print out noise statistics
                      plthist=True,     # plot noise on a histogram (number of pixels VS intensity)
                      shownoise=True,   # show noise distribution
                      gaussfit=True,    # Gaussian fitting on noise
                      fixmean=False,    # don't fix mean of gaussian fitting to 0.
                     )

# using noise statistics for other tasks
clean_image = continuum.set_threshold()  # default is to set threshold to 3 sigma
clean_image = continuum.set_threshold(5*continuum.noise())

# plotting contour maps:
continuum.imview(continuum,
                 fov=1, 
                 crms=continuum.noise(),         # base contour level. Default (None) is also the noise estimated from contour map.
                 clevels=[5, 10, 15, 20, 25],    # relative contour levels
                 scale="gamma",                  # change color map to gamma scale
                 gamma=0.8
                 )
```

You can also make moment maps and extract spectral profiles from data cubes!
```python
# list of moment maps to be outputted (using CASA definition)
mom_maps = datacube.immoments(moments=moments[-1, 0, 1, 2, 4, 9, 11], threshold=3*datacube.noise())  # returns a list
mom0 = mom_maps[1]
mom9 = mom_maps[5]

# show the maps:
mom0.imview()
mom0.imview()
```

Here are other ways you can manipulate images (these work primarily on the classes 'Datacube' and 'Spatialmap'):
```python
# Convolution:
convolved_continuum = continuum.imsmooth(bmaj=0.8, bmin=0.5, bpa=45)  # elliptical Gaussian convolution
convolved_continuum = continuum.imsmooth(width=3, kernel="box")  # box convolution

# Shifting images:
shifted_continuum = continuum.imshift("5:55:38.1255589097 2:11:37.3590883560") # This coordinate is now (0, 0)
shifted_continuum = continuum.imshift((3, 2))  # the coordinate (3, 2) is now (0, 0)
shifted_continuum = continuum.peakshift()  # this shifts the pixel with the maximum value to the center

# Rotating images:
rotated_img = continuum.rotate(-45)

# Regridding images:
mom0_regrid = .imregrid(continuum)  # using continuum as template
mom0_regrid = mom0.imregrid(continuum)  # using continuum as template
```

Viewing molecular line info is extremely easy with .line_info(), which finds data from the Splatalogue database:
```python
datacube.line_info()

"""
Below is the output after calling the method:

###############Line Information###############
Species ID: 245
Species: C18O
Chemical name: Carbon Monoxide
Frequency: 219.560358 +/- 1e-06 [GHz]
Resolved QNs: 2-1
Sij mu2: 0.02455 [Debye2]
Sij: 2.0
Einstein A coefficient (Aij): 6.049e-07 [1/s]
Lovas/AST intensity: 3.5
Lower energy level: 5.26877 [K]
Upper energy level: 15.80595 [K]
Upper state degeneracy (gu): 5.0
Rotational constants:
    B0 = 54891.42 [MHz]
Source: SLAIM
##############################################
Link to species data: https://splatalogue.online/species_metadata_displayer.php?species_id=245 
"""
```

All image objects can be manipulated intuitively via built-in python operators!
```python
# plotting a pv diagram after multipling it by two:
manipulated_pv = (pv*2 + 3) / 4
manipulated_pv.imview(vsys=1.4)  # notice the intensity is now twice the value.

# works with two different image!
mom8 = datacube.immoments(moments=8)
combined_image = mom0 + mom8
combined_image = combined_image.peakshift()
combined_image.imview(fov=3)
```

Image objects can also work with numpy arrays and astropy units!
```python
# works with numpy functions
print(np.nanmax(mom1[0, 0]))
print(np.nanargmin(mom1))
print(np.where(mom1>3))
print(np.round(mom1))

# astropy units can be incorporated flawlessly!
mom1_withunits = mom1*u.km/u.s  # note: assigning units leads to the unit in the image header being overwritten
mom1_withunits = mom1_withunits.to(u.m/u.s)
mom1_withunits.imview(vcenter=1400, vrange=3000)
```

Saving the image back into a FITS file:
```python
mom8.exportfits("test_fitsfiles/moment8_image.fits", overwrite=True)
```
