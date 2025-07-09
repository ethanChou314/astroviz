from .common import *


class Region:
    """
    A class for handling regions.

    This class provides functionality to load and read DS9 files and can be inputted into 
    various methods in Datacube, Spatialmap, and PVdiagram.
    """
    def __init__(self, regionfile=None, shape="ellipse", radius=None,
                 semimajor=1., semiminor=1., pa=0., center=(0., 0.), start=None, 
                 end=None, length=None, width=None, height=None, unit="arcsec",
                 relative=True, quiet=False):
        """
        Initializes a new instance of the Region class.

        Parameters:
            regionfile (str, optional): Path to the DS9 or CRTF region file.
            shape (str): Shape of the region. Supported shapes are 'circle', 'ellipse', 'box', 'line'.
            radius (float, optional): Radius of the circle. Required if shape is 'circle'.
            semimajor (float): Semi-major axis of the ellipse or box. Defaults to 1.0.
            semiminor (float): Semi-minor axis of the ellipse or box. Defaults to 1.0.
            pa (float): Position angle of the shape in degrees. Defaults to 0.
            center (tuple): Central position of the shape. Given as (x, y) coordinates.
            start (tuple, optional): Starting point of the line. Required if shape is 'line'.
            end (tuple, optional): End point of the line. Required if shape is 'line'.
            length (float, optional): Length of the line. Calculated if start and end are provided.
            width (float, optional): Width of the box. Required if shape is 'box'.
            height (float, optional): Height of the box. Required if shape is 'box'.
            unit (str): Unit of the coordinates (e.g., 'arcsec', 'degree'). Defaults to 'arcsec'.
            relative (bool): If True, coordinates are interpreted as relative to the center. Defaults to True.
            quiet (bool): If False, outputs more detailed information during processing. Defaults to False.

        Raises:
            ValueError: If required parameters for the specified shape are not provided.

        Note:
            - Some parameters are optional for shapes. For instance, if 'center', 'length', and 'pa'
              of line are specified, then the 'start' and 'end' are not required.
            - If 'regionfile' is provided, the shape and other parameters are read from the file.
            - The position angle 'pa' is relevant for shapes 'ellipse', 'box', and 'line'.
        """

        self.regionfile = regionfile
        shape = shape.lower()
        unit = unit.lower()
        isJ2000 = (isinstance(center, str) and not center.isnumeric()) \
                    or (len(center)==2 and isinstance(center[0], str) and isinstance(center[1], str)) 
        
        relative = False if isJ2000 else relative
        
        if length is not None or start is not None or end is not None:
            shape = "line"
            
        if width is not None or height is not None:
            shape = "box"
        
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
            if shape in ["box", "rectangle", "square"]:
                if shape == "square":
                    if width is not None:
                        self.width = self.height = width
                    elif height is not None:
                        self.width = self.height = height
                    elif length is not None:
                        self.width = self.height = length
                    elif semimajor is not None:
                        self.width = self.height = semimajor*2
                    else:
                        raise ValueError("Must specify the 'width', 'height', or 'length' parameter.")
                else:
                    if width is not None:
                        self.width = width
                    elif semimajor is not None:
                        self.width = semimajor*2
                        
                    if height is not None:
                        self.height = height
                    elif semiminor is not None:
                        self.height = semiminor*2
                    elif length is not None:
                        self.height = length
                    else:
                        raise ValueError("Must specify the 'height'.")
                self.shape = "box"
                self.pa = pa
                self.center = center
                self.unit = unit
                self.header = {"shape": self.shape,
                               "center": self.center,
                               "width": self.width,
                               "height": self.height,
                               "pa": self.pa,
                               "unit": self.unit,
                              }
                    
            self.relative = self.header["relative"] = relative
        else:
            self.__readfile(quiet=quiet)
    
    def __readfile(self, quiet=False):
        if not os.path.exists(self.regionfile):
            if not quiet:
                print(f"Given directory '{fitsfile}' does not exist as a relative directory. " + \
                       "Recursively finding file...")
            maybe_filename = _find_file(self.regionfile)
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
            self.shape = filelst[1].split()[0]
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
            self.pa = pa
            self.header = {"shape": shape,
                           "start": start,
                           "end": end,
                           "center": center,
                           "length": length,
                           "pa": pa,
                           "unit": unit,
                           "relative": relative,}
        elif shape == "ellipse":
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
        elif shape == "box":
            self.unit = unit = "deg"
            coord_tup = self.__filelst[3].replace(self.shape+"(", "")
            coord_tup = coord_tup.split(")")[0]  # string
            elements = coord_tup.split(", ")
            center_ra, center_dec, width_str, height_str, pa_str = elements
            self.center = center = float(center_ra), float(center_dec)
            width_str = width_str.replace('"', "arcsec").replace("'", "arcmin").replace("∘", "deg")
            height_str = height_str.replace('"', "arcsec").replace("'", "arcmin").replace("∘", "deg")
            self.width = width = float(u.Quantity(width_str).to_value(u.deg))
            self.height = height = float(u.Quantity(height_str).to_value(u.deg))
            self.pa = pa = float(pa_str)
            
            self.relative = relative = False
            self.header = {"shape": shape,
                           "center": center,
                           "width": width,
                           "height": height,
                           "pa": pa,
                           "unit": unit,
                           "relative": relative
                           }
        else:
            raise Exception("Region not supported in this current version.")

    def __readCRTF(self):
        shape = self.shape
        unit = self.unit = "deg"
        if shape == "line":
            coord_lst = self.__filelst[1].split()[1:5]
            
            # start coordinate
            start_ra = coord_lst[0][2:-1]
            start_dec = coord_lst[1][:-2]
            start_dec = start_dec.split(".")
            start_dec = start_dec[0] + ":" + start_dec[1] + ":" + start_dec[2] + "." + start_dec[3]
            start_str = start_ra + " " + start_dec
            start_coord = SkyCoord(start_str, unit=(u.hourangle, u.deg), frame="icrs")
            self.start = start = start_coord.to_string("hmsdms")
            
            # end coordinate
            end_ra = coord_lst[2][1:-1]
            end_dec = coord_lst[3][:-2]
            end_dec = end_dec.split(".")
            end_dec = end_dec[0] + ":" + end_dec[1] + ":" + end_dec[2] + "." + end_dec[3]
            end_str = end_ra + " " + end_dec
            end_coord = SkyCoord(end_str, unit=(u.hourangle, u.deg), frame="icrs")
            self.end = end = end_coord.to_string("hmsdms")
            
            # central coordinate
            center_ra = (end_coord.ra + start_coord.ra)/2
            center_dec = (end_coord.dec + start_coord.dec)/2
            self.center = center = SkyCoord(ra=center_ra, dec=center_dec, frame="icrs").to_string("hmsdms")
            
            # position angle 
            dx = start_coord.ra - end_coord.ra
            dy = end_coord.dec - start_coord.dec
            self.length = length = np.sqrt(dx**2+dy**2).to_value(self.unit)
            pa = -np.arctan(dx/dy).to_value(u.deg) if dy != 0 else (90. if dx>0 else -90.)
            self.pa = pa
            
            self.relative = relative = False
            self.header = {"shape": shape,
                           "start": start,
                           "end": end,
                           "center": center,
                           "length": length,
                           "pa": pa,
                           "unit": unit,
                           "relative": relative}
        elif shape == "ellipse":
            raise Exception("Not implemented yet.")
        elif shape == "box":
            raise Exception("Not implemented yet.")
        else:
            raise Exception("Not implemented yet.")