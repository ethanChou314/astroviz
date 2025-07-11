from .common import *
from .datacube import Datacube
from .spatialmap import Spatialmap
from .pvdiagram import PVdiagram
from .plot2d import Plot2D
from .region import Region


class ImageMatrix:
    """
    A class for handling and plotting multiple images with customizable annotations.
    """
    def __init__(self, figsize=(11.69, 16.57), axes_padding='auto', 
                 dpi=600, labelsize=10., fontsize=12., axeslw=1., 
                 tickwidth=None, ticksize=3., tick_direction="in", 
                 cbarloc="right", **kwargs):
        """
        Initialize the ImageMatrix instance with default parameters.

        Parameters:
        - figsize (tuple of float or int, optional): The size of the figure in inches. Default is (11.69, 8.27).
        - axes_padding (tuple of float, optional): The padding between axes. Default is (0.2, 0.2).
        - dpi (float or int, optional): The dots per inch (DPI) setting for the figure. Default is 600.
        - **kwargs: Additional keyword arguments to set or update the plotting parameters.

        Description:
        - This method initializes an ImageMatrix instance with default plotting parameters.
        - It sets up attributes such as `images`, `figsize`, `shape`, `size`, `axes_padding`, and `dpi`.
        - Default plotting parameters are defined in `self.default_params`.
        - Additional keyword arguments can be passed to customize plotting parameters, which are set using the `set_params` method.
        - Initializes internal dictionaries to store special parameters, shapes, lines, and text annotations.
        - Initializes `fig` and `axes` attributes for the Matplotlib figure and axes.

        Attributes:
        - images (list): A list to store images.
        - figsize (tuple): The size of the figure in inches.
        - shape (tuple): The shape of the image matrix (initially (0, 0)).
        - size (int): The size of the image matrix (initially 0).
        - axes_padding (tuple): The padding between axes.
        - dpi (float or int): The dots per inch (DPI) setting for the figure.
        - default_params (dict): A dictionary of default plotting parameters.
        - other_kwargs (dict): A dictionary for additional keyword arguments for plotting parameters.
        - specific_kwargs (dict): A dictionary for specific keyword arguments for individual images.
        - fig (matplotlib.figure.Figure or None): The Matplotlib figure object.
        - axes (list or None): The list of Matplotlib axes objects.
        """
        # set default parameters
        self.images: list = []  # empty list
        self.figsize: Tuple[Union[float, int]] = figsize
        self.shape: Tuple[int] = (0, 0)  # empty tuple
        self.size: int = 0   # empty shape
        
        # parse 'axes padding' parameter 
        if isinstance(axes_padding, (int, np.integer, float, np.floating)):
            self.axes_padding = (axes_padding, axes_padding)
        elif hasattr(axes_padding, "__iter__") and len(axes_padding) == 2:
            self.axes_padding = axes_padding
        elif axes_padding == "auto":
            self.axes_padding = axes_padding
        else:
            raise ValueError("'axes_padding' must be a number, a tuple of two numbers, or 'auto'")
            
        # parse dpi
        self.dpi: Union[float, int] = dpi
        
        # default parameters
        if tickwidth is None:
            tickwidth = axeslw  # set it to be the same as axeslw if not specified
            
        # parse 'cbarloc'
        if not cbarloc.islower():
            cbarloc = cbarloc.lower()
        if cbarloc not in ("right", "top"):
            raise ValueError("'cbarloc' must be either 'right' or 'top'.")
            
        self.default_params: dict = {'axes.labelsize': labelsize,
                                     'axes.titlesize': fontsize,
                                     'font.size': fontsize,
                                     'xtick.labelsize': labelsize,
                                     'ytick.labelsize': labelsize,
                                     'xtick.top': True,  # draw ticks on the top side
                                     'xtick.major.top': True,
                                     'figure.figsize': figsize,
                                     'font.family': _fontfamily,
                                     'mathtext.fontset': _mathtext_fontset,
                                     'mathtext.tt': _mathtext_tt,
                                     'axes.linewidth': axeslw, 
                                     'xtick.major.width': tickwidth,
                                     'xtick.major.size': ticksize,
                                     'xtick.direction': tick_direction,
                                     'ytick.major.width': tickwidth,
                                     'ytick.major.size': ticksize,
                                     'ytick.direction': tick_direction,
                                     }
            
        self.other_kwargs: dict = {"labelsize": labelsize, 
                                   "fontsize": fontsize, 
                                   "cbarloc": cbarloc}
        self.specific_kwargs: dict = {}
        self.set_params(new_params=kwargs)
        
        # initialize private attributes:
        self.__label_abc: bool = False
        self.__lines: dict = {}
        self.__shapes: dict = {}
        self.__texts: dict = {}
        self.__show_mag: list = []
        self.__set_positions: dict = {}
        self.__cbarloc = cbarloc
                    
        self.fig = None
        self.axes = None
        
    def __ravel_index(self, multi_dim_index):
        """
        Converts a multi-dimensional index to a single-dimensional index or 
        validates a single-dimensional index.

        Parameters:
            - multi_dim_index (int or iterable): The index to be converted or validated. 
                                               This can either be a single integer 
                                               representing a flat index or an iterable 
                                               (like a tuple or list) of length 2 
                                               representing a 2D index.

        Returns:
            - (int) The single-dimensional index corresponding to the provided multi-dimensional index.

        Raises:
            - ValueError
                - If `multi_dim_index` is an int and is out of bounds for the array size.
                - If `multi_dim_index` is an iterable and its length is not 2.
                - If `multi_dim_index` is an iterable and contains indices out of bounds 
                  for the array shape.
                - If `multi_dim_index` is neither an int nor an iterable of length 2.
        """
        if isinstance(multi_dim_index, (int, np.integer)):
            # check if it can be parsed as a one-dim index
            if multi_dim_index > self.size - 1:
                raise ValueError(f"Index {multi_dim_index} is out of bounds for size {self.size}")
            # return itself if it is an integer
            return multi_dim_index
        elif hasattr(multi_dim_index, "__iter__") and hasattr(multi_dim_index, "__getitem__"):
            # check if it can be parsed as a two-dim index
            if len(multi_dim_index) != 2:
                raise ValueError("'multi_dim_index' must be an iterable of length 2")
            row, col = multi_dim_index
            if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
                raise ValueError(f"Index {multi_dim_index} is out of bounds for shape {self.shape}")
            return np.ravel_multi_index(multi_dim_index, self.shape)
        else:
            raise ValueError("multi_dim_index must be an int or an iterable of length 2")
            
    def set_params(self, new_params=None, **kwargs):
        """
        Set or update the plotting parameters for the image matrix.

        Parameters:
        - new_params (dict, optional): A dictionary of new parameters to set or update. Default is None.
        - **kwargs: Additional keyword arguments to set or update the plotting parameters.

        Returns:
        - self (ImageMatrix): The updated ImageMatrix instance with the new parameters.

        Description:
        - This method sets or updates the default plotting parameters for the image matrix.
        - The parameters can be provided either as a dictionary (`new_params`) or as keyword arguments.
        - If a parameter key is found in `rcParams`, it updates the corresponding entry in `self.default_params`.
        - Otherwise, it updates the entry in `self.other_kwargs`.
        - This allows for flexible customization of plotting parameters for the image matrix.
        """
        # set new dictionary
        if new_params is None:
            new_params = kwargs
        
        # iterate over 'new_params' dict to replace default values
        for key, value in new_params.items():
            if key in rcParams:
                self.default_params[key] = value
            else:
                self.other_kwargs[key] = value

        return self
    
    def copy(self):
        """
        Create a deep copy of the ImageMatrix instance.

        Returns:
        - ImageMatrix: A deep copy of the current ImageMatrix instance.

        Description:
        - This method creates and returns a deep copy of the current ImageMatrix instance.
        - A deep copy ensures that all nested objects and attributes within the ImageMatrix are also copied, 
          preventing any changes to the original instance from affecting the copy.

        Example:
        - To create a copy of an existing ImageMatrix instance:
          >>> matrix_copy = original_matrix.copy()
        """
        return copy.deepcopy(self)
        
    def reshape(self, *shape, inplace=True):
        """
        Reshape the image matrix into a different 2D shape.

        Parameters:
        - shape (tuple of int): The new shape for the image matrix. It should be a tuple containing two integers.
        - inplace (bool, optional): Whether to modify the image matrix in place. Default is True.

        Returns:
        - matrix (ImageMatrix): The reshaped image matrix.

        Raises:
        - ValueError: If the length of `shape` is greater than 2 or if any element in `shape` is not an integer.

        Description:
        - This method changes the shape of the image matrix to the specified `shape`.
        - If `inplace` is False, a copy of the image matrix is created and modified.
        - If the length of `shape` is 1, it is converted to a 2D shape with one row.
        - The shape and size of the image matrix are updated to match the new shape.
        """
        # old shape
        old_shape = self.shape
        
        # create copy if inplace is set to True
        matrix = self if inplace else self.copy()
        
        # parse args
        if len(shape) == 1:
            shape = shape[0]
        
        # check if shape satisfies length requirement
        if len(shape) == 1:
            shape = (1, shape[0])
        elif len(shape) > 2:
            raise ValueError("Length of 'shape' is greater than 2.")
        
        # change items in shape to int:
        if any(not isinstance(item, (int, np.integer)) for item in shape):
            shape = tuple(int(item) for item in shape)

        # raise value error
        new_size = np.multiply(*shape)
        length_of_images = len(self.images)
        if new_size < length_of_images:
            warnings.warn(f"Not all {length_of_images} images will be plotted with shape {shape}.")

        # modify shape and size
        matrix.shape = shape
        matrix.size = new_size
                
        return matrix
        
    def add_image(self, image=None, **kwargs) -> None:
        """
        Add an image to the ImageMatrix.

        Parameters:
        - image (Spatialmap, PVdiagram, Plot2D, or None): The image to be added to the matrix.
        - plot (bool, optional): Whether to plot the image after adding it. Default is False.
        - **kwargs: Additional keyword arguments to be passed to the plot method.

        Returns:
        - self (ImageMatrix): The updated ImageMatrix instance.

        Raises:
        - ValueError: If the image added is not of an acceptable type (Spatialmap, PVdiagram, Plot2D, or None).

        Description:
        - Checks if the provided image is of an acceptable type.
        - Adds the image to the image matrix.
        - Reshapes the matrix if the number of images exceeds the current size.
        - Stores specific keyword arguments for the added image.
        - Optionally plots the image if the `plot` parameter is set to True.
        """
        # check type of input
        acceptable_types = (Spatialmap, PVdiagram, Plot2D, type(None))
        if not isinstance(image, acceptable_types):
            raise ValueError("The image added is not of an acceptable type.")
        
        # add image
        self.images.append(image)
        
        
        if self.size == 0:
            self.reshape((1, 1), inplace=True)
        elif self.size == 1:
            self.reshape((1, 2), inplace=True)
        elif self.size == 2:
            self.reshape((1, 3), inplace=True)
        elif self.size == 3:
            self.reshape((2, 2), inplace=True)
        elif len(self.images) > self.size:
            new_shape = (self.shape[0]+1, self.shape[1])  # add a new row if it exceeds current size
            self.reshape(new_shape, inplace=True)
            
        # set specific keyword arguments
        self.specific_kwargs[len(self.images)-1] = kwargs

    def __create_panel_labels(self) -> List[str]:
        """
        Generates a list of panel labels for the instance.

        This method creates panel labels in a sequential pattern, starting from 'a', 'b', 'c', ..., 'z',
        and then 'aa', 'ab', 'ac', ..., 'az', 'ba', 'bb', ..., 'zz', and so on. The length of the list
        of labels is determined by the number of items in `self.images`.

        Returns:
            List[str]: A list of panel labels.
        """
        # create generator object
        def labels() -> iter:
            i = 1
            while True:
                for label in itertools.product(string.ascii_lowercase, repeat=i):
                    yield ''.join(label)
                i += 1

        # allocate memory for labels by creating a list
        generator = labels()
        return list(next(generator) for _ in range(len(self.images)))
                
    def plot(self, plot=True):
        """
        Plot the images in the image matrix along with any added annotations.

        Parameters:
        - plot (bool, optional): Whether to display the plot. Default is True.

        Returns:
        - fig (matplotlib.figure.Figure): The figure object containing the plots.
        - axes (list of matplotlib.axes.Axes): The axes objects of the plots.

        Description:
        - This method sets the default plotting parameters and creates a figure with subplots
          according to the shape of the image matrix.
        - Each image in the image matrix is plotted in its respective subplot.
        - If an image is 'None' or the subplot index exceeds the number of images, the subplot is turned off.
        - Additional keyword arguments for plotting are passed to the `imview` method of each image.
        - If `self.__label_abc` is True, each subplot is labeled with a sequential letter (a, b, c, ...).
        - The figure's padding and dpi are adjusted according to the instance's attributes.
        - After plotting the images, any lines, shapes, or texts stored in the instance are drawn on the figure.
        - The figure is displayed if `plot` is True, and the figure and axes objects are returned.

        Example:
        - To plot the image matrix and display it:
          >>> fig, axes = image_matrix.plot()

        - To plot the image matrix without displaying it (e.g., for saving to a file):
          >>> fig, axes = image_matrix.plot(plot=False)
        """
        rcParams.update(self.default_params)  # set default parameters
        if self.axes_padding == "auto":
            fig, axes = plt.subplots(*self.shape, figsize=self.figsize, 
                                     constrained_layout=True)  # set fig, axes, etc.
        else:
            fig, axes = plt.subplots(*self.shape, figsize=self.figsize)  # set fig, axes, etc.
            
        if isinstance(axes, mpl.axes._axes.Axes):
            axes = [axes]
        else: 
            axes = axes.flatten()  # flatten axes
        other_kwargs = self.other_kwargs  # other keyword arguments
        
        label_idx: int = 0
        all_labels: List[str] = self.__create_panel_labels()
        for i, ax in enumerate(axes):
            # plot blank images if image is 'None' or the ax is out of range.
            if i >= len(self.images) or (image := self.images[i]) is None:  
                ax.spines["bottom"].set_color('none')
                ax.spines["top"].set_color('none')
                ax.spines["left"].set_color('none')
                ax.spines["right"].set_color('none')
                ax.axis('off')
            else:
                specific_kwargs = self.specific_kwargs[i]
                this_kwarg = copy.deepcopy(other_kwargs)
                
                # check if all parameters are valid
                valid_args = tuple(inspect.signature(image.imview).parameters.keys())
                this_kwarg = dict((key, value) for key, value in this_kwarg.items() if key in valid_args)
                this_kwarg.update(specific_kwargs)
                
                # add figure labels if necessary
                if "title" in this_kwarg:
                    title = this_kwarg["title"]
                    this_kwarg.pop("title")
                else:
                    title = ""
                    
                # add panel labels if necessary
                if self.__label_abc:
                    title = f"({all_labels[label_idx]}) " + title
                    label_idx += 1
                    
                # plot using 'imview' method
                if hasattr(image, "imview"):
                    ax = image.imview(ax=ax, plot=False, title=title, **this_kwarg)
                else:
                    ax = image.plot(ax=ax, plot=False, title=title, **this_kwarg)
                  
        # adjust padding
        if self.axes_padding != "auto":
            wspace, hspace = self.axes_padding
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
        
        # set dpi
        fig.set_dpi(self.dpi)
        
        # set instances
        self.fig = fig
        self.axes = axes
        
        # draw other annotations
        fig, axes = self.__draw_lines(fig, axes)
        fig, axes = self.__draw_shapes(fig, axes)
        fig, axes = self.__add_texts(fig, axes)
        fig, axes = self.__plot_mag(fig, axes)
        
        # set horizontal positions
        fig, axes = self.__set_horizontal_positions(fig, axes)
        
        if plot:
            plt.show()
        
        return fig, axes
        
    def label_panels(self) -> None:
        """
        Enable labeling of panels with sequential letters (a, b, c, ...).

        Description:
        - This method sets an internal flag to enable the labeling of panels with sequential letters.
        - When this flag is set to True, each panel in the image matrix will be 
          labeled with a sequential letter (a, b, c, ...).
        """
        self.__label_abc = True
        
    def show_magnification(self, zoomed_out_idx, zoomed_in_idx, 
                           linestyle="dashed", linecolor="skyblue", 
                           linewidth=1.0, edgecolor=None, 
                           facecolor="none", **kwargs) -> None:
        """
        Highlights a zoomed-in region within a zoomed-out image and draws connecting lines.

        This method adds a rectangle to the zoomed-out image, highlighting the region that is 
        zoomed in on another subplot. Additionally, it draws lines connecting the corners of 
        the zoomed-in region to the corresponding region in the zoomed-out image.

        Parameters:
            zoomed_out_idx (int or tuple): The index or coordinates of the zoomed-out image 
                                           in the image matrix.
            zoomed_in_idx (int or tuple): The index or coordinates of the zoomed-in image in 
                                          the image matrix.
            linestyle (str, optional): The style of the connecting lines. Default is "dashed".
            linecolor (str, optional): The color of the connecting lines. Default is "skyblue".
            linewidth (float, optional): The width of the connecting lines. Default is 1.0.
            edgecolor (str, optional): The edge color of the rectangle. If None, defaults to 
                                       the value of `linecolor`. Default is None.
            facecolor (str, optional): The face color of the rectangle. Default is "none".
            **kwargs: Additional keyword arguments to be passed to the `Rectangle` patch.

        Returns:
            None

        Description:
        - This method transforms the provided indices into raveled indices and stores the 
          parameters required for highlighting and connecting the zoomed regions.
        - The actual drawing of the rectangle and lines is handled in the `__plot_mag` method, 
          which is called during the plotting process.
        """
        # ravel indicies:
        zoomed_out_idx = self.__ravel_index(zoomed_out_idx)
        zoomed_in_idx = self.__ravel_index(zoomed_in_idx)
        
        if zoomed_out_idx == zoomed_in_idx:
            raise ValueError("Same indicies for zoomed in/out images were provided.")
        
        # default value of edge color:
        if edgecolor is None:
            edgecolor = linecolor
            
        # set keyword arguments as dictionary:
        set_kwargs = {"linestyle": linestyle, "linecolor": linecolor,
                      "linewidth": linewidth, "facecolor": facecolor,
                      "edgecolor": edgecolor}
        
        # append to private variable
        self.__show_mag.append((zoomed_out_idx, zoomed_in_idx, 
                                set_kwargs, kwargs))
        
    def __plot_mag(self, fig, axes):
        if not self.__show_mag:
            return fig, axes
        
        for (out_idx, in_idx, set_kwargs, kwargs) in self.__show_mag:
            # skip loop if one of the images is a blank one
            if self.images[out_idx] is None or self.images[in_idx] is None:
                continue
            
            # get values
            out_ax = axes[out_idx]  # zoomed-out axis
            out_xlim = out_ax.get_xlim()
            out_xrange = abs(out_xlim[1] - out_xlim[0])
            out_ylim = out_ax.get_ylim()
            out_yrange = abs(out_ylim[1] - out_ylim[0])
            
            in_ax = axes[in_idx]  # zoomed-in axis
            in_xlim = in_ax.get_xlim()
            in_xrange = abs(in_xlim[1] - in_xlim[0])
            in_ylim = in_ax.get_ylim()
            in_yrange = abs(in_ylim[1] - in_ylim[0])
            
            # swap out and in indicies if they are opposite
            if in_xrange > out_xrange and in_yrange > out_yrange:
                out_idx, in_idx = in_idx, out_idx  # python swap!
                out_ax = axes[out_idx]  # zoomed-out axis
                out_xlim = out_ax.get_xlim()
                out_xrange = abs(out_xlim[1] - out_xlim[0])
                out_ylim = out_ax.get_ylim()
                out_yrange = abs(out_ylim[1] - out_ylim[0])

                in_ax = axes[in_idx]  # zoomed-in axis
                in_xlim = in_ax.get_xlim()
                in_xrange = abs(in_xlim[1] - in_xlim[0])
                in_ylim = in_ax.get_ylim()
                in_yrange = abs(in_ylim[1] - in_ylim[0])
            
            # add rectangle in zoomed-out image:
            br_xy = (min(in_xlim), min(in_ylim))  # bottom right xy coord
            rect = patches.Rectangle(xy=br_xy, 
                                     width=in_xrange, 
                                     height=in_yrange, 
                                     angle=0,
                                     linewidth=set_kwargs["linewidth"], 
                                     linestyle=set_kwargs["linestyle"], 
                                     edgecolor=set_kwargs["edgecolor"], 
                                     facecolor=set_kwargs["facecolor"], 
                                     **kwargs)
            out_ax.add_patch(rect)
            
            fig.canvas.draw()
            transFigure = fig.transFigure.inverted()
            
            # Get the locations of the axes in the grid
            out_loc = np.unravel_index(out_idx, self.shape)
            in_loc = np.unravel_index(in_idx, self.shape)
            
            # Calculate the coordinates for the lines based on the location of the axes
            if out_loc[0] == in_loc[0]:  # same row -> plot side to side
                if out_loc[1] > in_loc[1]:  # right is zoomed out, left is zoomed in
                    in_coords = [(in_xlim[1], in_ylim[0]), (in_xlim[1], in_ylim[1])]
                    out_coords = [(in_xlim[0], in_ylim[0]), (in_xlim[0], in_ylim[1])]
                else:  # left is zoomed out, right is zoomed in
                    in_coords = [(in_xlim[0], in_ylim[0]), (in_xlim[0], in_ylim[1])]
                    out_coords = [(in_xlim[1], in_ylim[0]), (in_xlim[1], in_ylim[1])]
            else:  # different rows -> plot one above the other
                if out_loc[0] > in_loc[0]:  # top is zoomed out, bottom is zoomed in
                    in_coords = [(in_xlim[0], in_ylim[0]), (in_xlim[1], in_ylim[0])]
                    out_coords = [(in_xlim[0], in_ylim[1]), (in_xlim[1], in_ylim[1])]
                else:  # bottom is zoomed out, top is zoomed in
                    in_coords = [(in_xlim[0], in_ylim[1]), (in_xlim[1], in_ylim[1])]
                    out_coords = [(in_xlim[0], in_ylim[0]), (in_xlim[1], in_ylim[0])]
            
            # plot
            for in_coord, out_coord in zip(in_coords, out_coords):
                in_fig_coord = transFigure.transform(in_ax.transData.transform(in_coord))
                out_fig_coord = transFigure.transform(out_ax.transData.transform(out_coord))
                
                line = plt.Line2D((in_fig_coord[0], out_fig_coord[0]), 
                                  (in_fig_coord[1], out_fig_coord[1]),
                                  transform=fig.transFigure, 
                                  color=set_kwargs["linecolor"],
                                  linestyle=set_kwargs["linestyle"], 
                                  linewidth=set_kwargs["linewidth"])
                fig.lines.append(line)

        return fig, axes
            
    def add_text(self, img_idx, x, y, s, **kwargs) -> None:
        """
        Add text to the specified image in the image matrix.

        Parameters:
        - img_idx (int): The index of the image to which the text will be added.
        - x (float): The x-coordinate for the text position.
        - y (float): The y-coordinate for the text position.
        - s (str): The text string to be displayed.
        - **kwargs: Additional keyword arguments to be passed to the `text` method of the axes.

        Returns:
        - None

        Raises:
        - ValueError: If the specified image index is invalid.

        Description:
        - This method creates a text object with specified properties and adds it to the specified image in the image matrix.
        - The text properties such as position, string, and other text attributes can be customized through `kwargs`.
        - The text specifications are stored in `self.__texts` and will be drawn when the image matrix is plotted.
        """
        img_idx = self.__ravel_index(img_idx)
        
        kwargs.update({"s": s, "x": x, "y": y})
        
        if img_idx in self.__texts:
            self.__texts[img_idx].append(kwargs)
        else:
            self.__texts[img_idx] = [kwargs]
            
    def __add_texts(self, fig, axes) -> None:
        """
        Draw stored text objects on the specified axes in the image matrix.

        Parameters:
        - fig (matplotlib.figure.Figure): The figure object containing the plots.
        - axes (list of matplotlib.axes.Axes): The axes objects of the plots.

        Returns:
        - fig (matplotlib.figure.Figure): The updated figure object with the text objects drawn.
        - axes (list of matplotlib.axes.Axes): The updated axes objects.

        Description:
        - This method draws text objects stored in `self.__texts` on the specified figure and axes.
        - Each text object is associated with an image index and is added to the corresponding subplot.
        - If there are no text objects to be drawn (`self.__texts` is empty), the method returns the figure and axes unmodified.

        Text Drawing Process:
        - The method first checks if there are any text objects to be drawn. If not, it returns the figure and axes unmodified.
        - For each text object defined in `self.__texts`, the method retrieves the image index and the text properties.
        - The text is added to the corresponding axis using the `text` method with the specified properties.
        - The method ensures that the text objects are drawn when the figure is displayed by updating the axes with the text objects.

        Raises:
        - None
        """
        if not self.__texts:
            return fig, axes
        
        for img_idx, kwargs_lst in self.__texts.items():
            for kwargs in kwargs_lst:
                axes[img_idx].text(**kwargs)
                
        return fig, axes
    
    def add_arrow(self, img_idx, xy1, xy2=None, dx=None, dy=None, 
                  width=1., color="tomato", double_headed=False, **kwargs):
        """
        Add an arrow to a specific image in the image matrix.

        Parameters:
        - img_idx (int or tuple): The index or coordinates of the image to which the arrow will be added. 
          If a tuple is provided, it should contain two elements representing coordinates.
        - xy1 (tuple): The starting point of the arrow (x, y).
        - xy2 (tuple, optional): The ending point of the arrow (x, y). If not provided, dx and dy must be specified.
        - dx (float, optional): The length of the arrow along the x-axis. Required if xy2 is not provided.
        - dy (float, optional): The length of the arrow along the y-axis. Required if xy2 is not provided.
        - width (float, optional): The width of the arrow. Default is 1.0.
        - color (str, optional): The color of the arrow. Default is "tomato".
        - **kwargs: Additional keyword arguments to pass to `patches.Arrow`.

        Raises:
        - ValueError: If neither xy2 nor both dx and dy are provided.

        Description:
        - Converts the img_idx to a single-dimensional index using the `__ravel_index` method.
        - Validates that either xy2 or both dx and dy are provided; raises a ValueError if not.
        - Computes dx and dy from xy1 and xy2 if xy2 is provided.
        - Creates an Arrow patch with the specified properties and additional keyword arguments.
        - Adds the Arrow patch to the list of shapes for the specified image index.
        """
        # ravel index and check if the index is valid
        img_idx = self.__ravel_index(img_idx)
        
        # check if all info is provided:
        if xy2 is None and (dx is None or dy is None):
            raise ValueError("Either 'xy2' or both 'dx' and 'dy' must be provided.")
        
        # get x1 and y1
        x1, y1 = xy1
        
        # get dx and dy if xy2 is provided (this overwrites the provided 'dx' and 'dy')
        if xy2 is not None:
            x2, y2 = xy2
            dx = x2 - x1
            dy = y2 - y1

        if double_headed:
            # forward arrow
            self.add_arrow(img_idx=img_idx, xy1=(x1+dx/2, y1+dy/2), dx=dx/2, dy=dy/2,
                           width=width, color=color, double_headed=False, 
                           **kwargs)

            # backward arrow
            self.add_arrow(img_idx=img_idx, xy1=(x1+dx/2, y1+dy/2), dx=-dx/2, dy=-dy/2,
                           width=width, color=color, double_headed=False, 
                           **kwargs)
        else:
            # create new patch object:
            patch = patches.Arrow(x=x1, y=y1, dx=dx, dy=dy, color=color, 
                                  width=width, **kwargs)
            
            # save the patch to the private instance:
            if img_idx in self.__shapes:
                self.__shapes[img_idx].append(patch)
            else:
                self.__shapes[img_idx] = [patch]

    def add_rectangle(self, img_idx, xy=(0, 0), width=1, height=None, angle=0, 
                      linewidth=1., edgecolor="skyblue", facecolor='none', 
                      linestyle="dashed", center=True, **kwargs) -> None:
        """
        Add a rectangle to the specified image in the image matrix.

        Parameters:
        - img_idx (int): The index of the image to which the rectangle will be added.
        - xy (tuple, optional): The (x, y) bottom left corner coordinates of the rectangle. Default is (0, 0).
        - width (float, optional): The width of the rectangle. Default is 1.
        - height (float, optional): The height of the rectangle. If None, it is set equal to the width. Default is None.
        - angle (float, optional): The rotation angle of the rectangle in degrees. Default is 0.
        - linewidth (float, optional): The width of the rectangle edge. Default is 1.
        - edgecolor (str, optional): The edge color of the rectangle. Default is "skyblue".
        - facecolor (str, optional): The fill color of the rectangle. Default is 'none'.
        - linestyle (str, optional): The line style of the rectangle edge. Default is "dashed".
        - **kwargs: Additional keyword arguments to be passed to the `Rectangle` constructor.

        Returns:
        - None

        Raises:
        - ValueError: If the specified image index is invalid.

        Description:
        - This method creates a rectangle patch and adds it to the specified image in the image matrix.
        - The rectangle's properties such as width, height, angle, colors, and line styles can be customized.
        - The rectangle is stored in `self.__shapes` and will be drawn when the image matrix is plotted.
        """
        # ravel index and check if the index is valid
        img_idx = self.__ravel_index(img_idx)
        
        # set height to be same as width if height is not specified:
        if height is None:
            height = width
        
        # calculate bottom right if xy specified should be the center:
        if center:
            xy = (xy[0]-width/2, xy[1]-height/2)
         
        # create new patch object:
        patch = patches.Rectangle(xy, width=width, height=height, angle=angle,
                                  linewidth=linewidth, linestyle=linestyle, 
                                  edgecolor=edgecolor, facecolor=facecolor, 
                                  **kwargs)
        
        # save the patch to the private instance:
        if img_idx in self.__shapes:
            self.__shapes[img_idx].append(patch)
        else:
            self.__shapes[img_idx] = [patch]
        
    def add_ellipse(self, img_idx, xy=(0, 0), width=1, height=None, angle=0, 
                    linewidth=1, facecolor='none', edgecolor="skyblue", 
                    linestyle="dashed", center=True, **kwargs) -> None:
        """
        Add an ellipse to the specified image in the image matrix.

        Parameters:
        - img_idx (int): The index of the image to which the ellipse will be added.
        - xy (tuple, optional): The (x, y) center coordinates of the ellipse. Default is (0, 0).
        - width (float, optional): The width of the ellipse. Default is 1.
        - height (float, optional): The height of the ellipse. If None, it is set equal to the width. Default is None.
        - angle (float, optional): The rotation angle of the ellipse in degrees. Default is 0.
        - linewidth (float, optional): The width of the ellipse edge. Default is 1.
        - facecolor (str, optional): The fill color of the ellipse. Default is 'none'.
        - edgecolor (str, optional): The edge color of the ellipse. Default is "skyblue".
        - linestyle (str, optional): The line style of the ellipse edge. Default is "dashed".
        - **kwargs: Additional keyword arguments to be passed to the `Ellipse` constructor.

        Returns:
        - None

        Raises:
        - ValueError: If the specified coordinates or image index are invalid.

        Description:
        - This method creates an ellipse patch and adds it to the specified image in the image matrix.
        - The ellipse's properties such as width, height, angle, colors, and line styles can be customized.
        - The ellipse is stored in `self.__shapes` and will be drawn when the image matrix is plotted.
        """
        # check whether coordinate specified is correct
        img_idx = self.__ravel_index(img_idx)
            
        if height is None:
            height = width
        
        if center:
            xy = (xy[0]-width/2, xy[1]-height/2)
        
        patch = patches.Ellipse(xy=xy, width=width, height=height, facecolor=facecolor,
                                angle=angle, linestyle=linestyle, **kwargs)
        if img_idx in self.__shapes:
            self.__shapes[img_idx].append(patch)
        else:
            self.__shapes[img_idx] = [patch]
        
    def add_patch(self, patch, img_idx=None) -> None:
        """
        Add a patch (shape) to the specified image(s) in the image matrix.

        Parameters:
        - patch (matplotlib.patches.Patch): The patch (shape) to be added to the image(s).
        - img_idx (int or iterable of int, optional): The index or indices of the image(s) 
          to which the patch will be added. If None, the patch will be added to all non-None 
          images in the image matrix. Default is None.

        Returns:
        - None

        Description:
        - This method adds a patch (shape) to the specified image(s) in the image matrix.
        - If `img_idx` is None, the patch will be added to all non-None images.
        - If `img_idx` is an integer, the patch will be added to the corresponding image.
        - If `img_idx` is an iterable, the patch will be added to each specified image.
        - The patches are stored in `self.__shapes`, which is used by the `__draw_shapes` 
          method to draw the patches on the figure.
        """
        if img_idx is None:
            img_idx = [idx for idx in range(self.size) if self.images[idx] is not None]
        
        if hasattr(img_idx, "__iter__"):
            for idx in img_idx:
                if img_idx in self.__shapes:
                    self.__shapes[img_idx].append(patch)
                else:
                    self.__shapes[img_idx] = [patch]
        elif isinstance(img_idx, int):
            if img_idx in self.__shapes:
                self.__shapes[img_idx].append(patch)
            else:
                self.__shapes[img_idx] = [patch]
        
    def __draw_shapes(self, fig, axes):
        """
        Draw stored shapes on the specified axes in the image matrix.

        Parameters:
        - fig (matplotlib.figure.Figure): The figure object containing the plots.
        - axes (list of matplotlib.axes.Axes): The axes objects of the plots.

        Returns:
        - fig (matplotlib.figure.Figure): The updated figure object with the shapes drawn.
        - axes (list of matplotlib.axes.Axes): The updated axes objects.

        Description:
        - This method draws shapes stored in `self.__shapes` on the specified figure and axes.
        - Each shape is associated with an image index and is added as a patch to the corresponding subplot.
        - If there are no shapes to be drawn (`self.__shapes` is empty), the method returns the figure and axes unmodified.

        Shape Drawing Process:
        - The method first checks if there are any shapes to be drawn. If not, it returns the figure and axes unmodified.
        - For each shape defined in `self.__shapes`, the method retrieves the image index and the patches (shapes) to be drawn.
        - Each patch is added to the corresponding axis using the `add_patch` method.
        - The method ensures that the shapes are drawn when the figure is displayed by updating the axes with the patches.
        """
        if not self.__shapes:
            return fig, axes
        
        for img_idx, patches in self.__shapes.items():
            for patch in patches:
                # make a copy to prevent run-time errors when plotting the same figure the second time:
                patch = copy.deepcopy(patch)
                axes[img_idx].add_patch(patch)
                
        return fig, axes
    
    def add_line(self, coord1, coord2, color="skyblue", linewidth=1.0, 
                 linestyle='dashed', **kwargs) -> None:
        """
        Add a line between two coordinates on the image matrix.

        Parameters:
        - coord1 (tuple): The starting coordinate of the line in the format 
          (image index, x coordinate, y coordinate).
        - coord2 (tuple): The ending coordinate of the line in the format 
          (image index, x coordinate, y coordinate).
        - color (str, optional): Color of the line. Default is 'k' (black).
        - linewidth (float, optional): Width of the line. Default is 1.0.
        - linestyle (str, optional): Style of the line. Default is '-' (solid line).
        - **kwargs: Additional keyword arguments to be passed to the 'plot' method.

        Returns:
        - None

        Raises:
        - ValueError: If `coord1` or `coord2` are not in the correct format or the indices are invalid.

        Description:
        - This method draws a line between two specified coordinates on the image matrix.
        - The coordinates must include the image index and the x and y positions within that image.
        """
        if len(coord1) != 3 or len(coord2) != 3:
            raise ValueError("Length of coordinates specified must be 3 " +\
                             "with the format (image index, x, y)")
            
        img_idx1, *_ = coord1
        img_idx2, *_ = coord2
        if img_idx1 > self.size - 1 or img_idx2 > self.size - 1:
            raise ValueError("Image index specified exceeds size of ImageMatrix.")

        kwargs.update({"color": color, 
                       "linewidth": linewidth,
                       "linestyle": linestyle})
        self.__lines[(coord1, coord2)] = kwargs
        
    def __draw_lines(self, fig, axes):
        """
        Draw stored lines across different axes in the image matrix.

        Parameters:
        - fig (matplotlib.figure.Figure): The figure object containing the plots.
        - axes (list of matplotlib.axes.Axes): The axes objects of the plots.

        Returns:
        - fig (matplotlib.figure.Figure): The updated figure object with the lines drawn.
        - axes (list of matplotlib.axes.Axes): The updated axes objects.

        Description:
        - This method draws lines stored in `self.__lines` on the specified figure and axes.
        - Each line is defined by two coordinates, where each coordinate includes the image index and
          the (x, y) position within that image.
        - The method transforms data coordinates to figure coordinates to draw lines that span across
          different subplots.
        - If there are no lines to be drawn (`self.__lines` is empty), the method returns the figure and axes unmodified.

        Line Drawing Process:
        - The method first checks if there are any lines to be drawn. If not, it returns the figure and axes unmodified.
        - It then draws the canvas and uses the `transFigure` transformation to convert data coordinates to figure coordinates.
        - For each line defined in `self.__lines`, the method retrieves the starting and ending coordinates, transforms them, 
          and creates a `Line2D` object to draw the line across the figure.
        - The `Line2D` object is appended to the figure's lines, ensuring the line is drawn when the figure is displayed.
        """
        # don't do anything if nothing needs to be plotted
        if not self.__lines:
            return fig, axes
        
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        
        for (coord1, coord2), line_kwargs in self.__lines.items():
            idx1, x1, y1 = coord1
            idx2, x2, y2 = coord2
            
            ax1 = axes[idx1]
            ax2 = axes[idx2]
            
            coord1_fig = transFigure.transform(ax1.transData.transform([x1, y1]))
            coord2_fig = transFigure.transform(ax2.transData.transform([x2, y2]))
            
            line = plt.Line2D((coord1_fig[0], coord2_fig[0]), (coord1_fig[1], coord2_fig[1]),
                              transform=fig.transFigure, **line_kwargs)
            fig.lines.append(line)
        
        return fig, axes
            
    def clear_annotations(self, inplace=True) -> None:
        """
        Clear all stored annotations (lines, shapes, and texts) from the image matrix.
        
        Parameters:
        - inplace (bool): Whether to modify the image matrix in place. Default is True.
        
        Description:
        - This method clears all lines, shapes, and text annotations that have been added to the image matrix.
        - After calling this method, the image matrix will have no annotations.
        - This can be useful if you want to reset the annotations and start fresh.
        """
        matrix = self if inplace else self.copy()
        matrix.__lines = {}
        matrix.__shapes = {}
        matrix.__texts = {}
        matrix.__label_abc = False
        
    def savefig(self, fname, format="pdf", bbox_inches="tight", transparent=True, 
                overwrite=False, **kwargs) -> None:
        """
        Save the current figure to a file.

        Parameters:
        - fname (str): The name of the file to save the figure to. If no extension is provided, '.pdf' will be added.
        - format (str, optional): The format to save the figure in. Default is 'pdf'.
        - bbox_inches (str, optional): Bounding box in inches. Default is 'tight'.
        - transparent (bool, optional): If True, the background of the figure will be transparent. Default is True.
        - overwrite (bool, optional): Whether to overwrite an existing file with the same name. Default is False.
        - **kwargs: Additional keyword arguments to pass to `fig.savefig`.

        Returns:
        - None

        Raises:
        - Exception: If no figure is available to save.

        Description:
        - Checks if a figure is available to save; raises an exception if not.
        - Adds a file extension to the filename if it is not provided.
        - If `overwrite` is False and a file with the specified name exists, 
          modifies the filename to avoid overwriting by appending a number.
        - Saves the figure with the specified parameters.
        """
        fig = self.fig
        
        # check if figure needs to be plotted:
        if fig is None:
            raise Exception("No figure available to save. Plot the figure using '.plot()' before saving.")
            
        # check if format is correct:
        if not format.islower():  # make it case insensitive
            format = format.lower()
        supported_formats = mpl.figure.FigureCanvasBase.get_supported_filetypes()
        if format not in supported_formats:
            supported_formats = str(tuple(supported_formats))[1:-1]
            raise ValueError(f"{format} is not a supported filetype: {supported_formats}.")
        
        # add file extension if not provided:
        extension = "." + format
        len_ext = len(extension)
        if not fname.endswith(extension):
            fname += extension
        
        # add numbers if file exists:
        if not overwrite:
            fname = _prevent_overwriting(fname)
        
        # save file:
        fig.savefig(fname, format=format, bbox_inches=bbox_inches, 
                    transparent=transparent, **kwargs)
        
        # print directory:
        print(f"Image successfully saved as '{fname}'")

    def align_center(self, row) -> None:
        ncol: int = self.shape[1]
        begin_idx: int = self.__ravel_idx((row, 0))
        end_idx: int = self.__ravel_idx((row, ncol))  # not included
        number_of_nonblank_images: int = sum(image is not None for image in self.images[begin_idx, end_idx])
        positions: np.ndarray = np.linspace(0, 1, number_of_nonblank_images+2)

        i: int = 0  # increases by one when image is not None
        for idx in enumerate(range(begin_idx, end_idx)):
            if self.images[idx] is None:
                continue
            self.set_horizontal_position(self, idx, center=positions[i])
            i += 1  # increment

    def set_horizontal_position(self, img_idx, center=0.5) -> None:
        # warning message
        if self.axes_padding == "auto":
            warnings.warn("Setting 'axes_padding' to 'auto' may result in different subplot sizes.")

        # parse and ravel image index:
        img_idx = self.__ravel_index(img_idx)
        
        # parse central coordinate:
        if not isinstance(center, (float, np.floating, int, np.integer)):
            raise ValueError("'center' must be an iterable or a float, int, or their numpy equivalents")
            
        if not (0 <= center <= 1):
            raise ValueError("'center' must be within the range [0, 1]")
        
        self.__set_positions[img_idx] = center
        
    def __set_horizontal_positions(self, fig, axes):
        if not self.__set_positions:
            return fig, axes
        
        for idx, center in self.__set_positions.items():
            # get ax
            ax = axes[idx]
            
            # get current position to calculate width and height
            points = ax.get_position().get_points()
            x1, y1 = points[0]
            x2, y2 = points[1]
            width = abs(x2-x1)
            height = abs(y2-y1)
            
            # calculate new position
            left = center - width / 2
            bottom = y1 - height / 2  
            new_position = [left, bottom, width, height]
            
            # set new position
            ax.set_position(new_position)
        return fig, axes
        
    def clean_labels(self, only_one=False) -> None:
        """
        Adjusts the labels and color bar labels in a grid of subplots.

        This method iterates over a grid of subplots and ensures that:
            - X-axis and Y-axis labels are only displayed on the bottom-left subplot.
            - Color bar labels are only displayed on the rightmost subplots.

        The method updates the `specific_kwargs` dictionary for each subplot, which is used to configure
        the display properties of each subplot in the grid.
        """
        if only_one:
            for i in range(self.shape[0]):  # row 
                for j in range(self.shape[1]):  # column
                    # turn on labels only at bottom left corner
                    if i == self.shape[0] - 1 and j == 0:
                        xlabelon = True
                        ylabelon = True
                    else:
                        xlabelon = False
                        ylabelon = False

                    # turn on color bar labels only on right/top columns
                    if self.__cbarloc == "right":
                        cbarlabelon = (j == self.shape[1] - 1)
                    else:
                        cbarlabelon = (i == 0)

                    # set parameters
                    ravelled_idx = self.__ravel_index((i, j))
                    if ravelled_idx > len(self.images) - 1:
                        break
                    self.specific_kwargs[ravelled_idx]["xlabelon"] = xlabelon
                    self.specific_kwargs[ravelled_idx]["ylabelon"] = ylabelon
                    self.specific_kwargs[ravelled_idx]["cbarlabelon"] = cbarlabelon
        else:
            for i in range(self.shape[0]):  # row 
                for j in range(self.shape[1]):  # column
                    # turn on labels only at bottom left corner
                    xlabelon = False
                    ylabelon = False
                    if i == self.shape[0] - 1:
                        xlabelon = True
                    if j == 0:
                        ylabelon = True

                    # turn on color bar labels only on right/top columns
                    if self.__cbarloc == "right":
                        cbarlabelon = (j == self.shape[1] - 1)
                    else:
                        cbarlabelon = (i == 0)

                    # set parameters
                    ravelled_idx = self.__ravel_index((i, j))
                    if ravelled_idx > len(self.images) - 1:
                        break
                    self.specific_kwargs[ravelled_idx]["xlabelon"] = xlabelon
                    self.specific_kwargs[ravelled_idx]["ylabelon"] = ylabelon
                    self.specific_kwargs[ravelled_idx]["cbarlabelon"] = cbarlabelon
        
    def pop(self, image_location=-1, inplace=True):
        """
        Remove and return the image at the specified location from the image matrix.

        Parameters:
        - image_location (int or tuple, optional): The index or coordinates of the image to remove.
          Default is -1, which removes the last image.
        - inplace (bool, optional): Whether to modify the image matrix in place.
          Default is True.

        Returns:
        - image (object): The removed image.

        Raises:
        - ValueError: If the specified location is invalid or exceeds the size of the image matrix.

        Description:
        - If `inplace` is False, creates a copy of the image matrix to modify.
        - If `image_location` is a tuple, it should contain two elements representing coordinates.
          The location is then flattened to a single index.
        - If the specified `image_location` exceeds the size of the image matrix, a ValueError is raised.
        - The image at the specified location is removed and replaced with `None`.
        """
        # parse index
        image_location = self.__ravel_index(image_location)
        
        # make copy if user doesn't want to modify in-place
        matrix = self if inplace else self.copy()
        
        # remove image by replacing it with a blank one
        image = matrix.images[image_location]
        matrix.images[image_location] = None
        
        return image


def plt_1ddata(xdata=None, ydata=None, xlim=None, ylim=None, mean_center=False, title=None,
               legendon=True, xlabel="", ylabel="", threshold=None, linewidth=0.8,
               xtick_spacing=None, ytick_spacing=None, borderwidth=0.7,
               labelsize=7, fontsize=8, ticksize=3, legendsize=6, title_position=0.92,
               bbox_to_anchor=(0.6, 0.95), legendloc="best", threshold_color="gray",
               linecolor="k", figsize=(2.76, 2.76), bins="auto", hist=False,
               dpi=600, plot=True, **kwargs):
    
    fontsize = labelsize if fontsize is None else fontsize
    
    if xlim is None:
        xlim = (np.nanmin(xdata), np.nanmax(xdata))
    if ylim is None:
        ylim = []
    
    params = {'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'font.size': fontsize,
              'legend.fontsize': legendsize,
              'xtick.labelsize': labelsize,
              'ytick.labelsize': labelsize,
              'figure.figsize': figsize,
              'figure.dpi': dpi,
              'font.family': _fontfamily,
              "mathtext.fontset": _mathtext_fontset, #"Times New Roman"
              'mathtext.tt': _mathtext_tt,
              'axes.linewidth': borderwidth,
              'xtick.major.width': borderwidth,
              'ytick.major.width': borderwidth,
              'xtick.major.size': ticksize,
              'ytick.major.size': ticksize,
              }
    rcParams.update(params)
    
    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False)
    plt.subplots_adjust(wspace=0.4)
    
    if hist:
        ax.hist(xdata, bins=bins, lw=linewidth, **kwargs)
    else:
        ax.plot(xdata, ydata, color=linecolor, lw=linewidth, **kwargs)
        
    if legendon:
        ax.legend(frameon=False, loc=legendloc, bbox_to_anchor=bbox_to_anchor)
        
    ax.tick_params(which='both',direction='in',bottom=True, top=True, left=True, right=True,
                   colors="k", labelrotation=0, labelcolor="k")
    
    if len(xlim) == 2:
        ax.set_xlim(xlim)
    if len(ylim) == 2:
        ax.set_ylim(ylim)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    if xtick_spacing is not None:
        ax.set_xticks(np.arange(-100, 100, xtick_spacing))
        ax.set_xlim(xlim)
    if ytick_spacing is not None:
        ax.set_yticks(np.arange(-100, 100, ytick_spacing))
        ax.set_ylim(ylim)
        
    if mean_center:
        xlim = ax.get_xlim()
        half_range = (np.max(xlim) - np.min(xlim))/2
        xmean = np.trapz(x=xdata, y=xdata*ydata)/np.trapz(x=xdata, y=ydata)
        xlim = (xmean-half_range, xmean+half_range)
        ax.set_xlim(xlim)
    
    if title is not None:
        xlim = ax.get_xlim() 
        ylim = ax.get_ylim() 
        xrange = np.max(xlim) - np.min(xlim)
        yrange = np.max(ylim) - np.min(ylim)
        xposition = xlim[0]+(1-title_position)*xrange
        yposition = ylim[0]+title_position*yrange
        ax.text(x=xposition, y=yposition, 
                s=title, color="k", ha="left", va="top", fontsize=fontsize)
    
    if threshold is not None:
        ax.axhline(y=threshold, linestyle="dashed", linewidth=linewidth, color=threshold_color)
        
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    
    if plot:
        plt.plot()
    
    return ax