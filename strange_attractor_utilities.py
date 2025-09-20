"""
Strange Attractor Utilities

This module contains utility functions for generating Peter de Jong attractors.
Inspired by the structure of the Paint-Pours project for consistency and modularity.
"""

import matplotlib
# Set non-interactive backend for worker processes to prevent GUI issues
import multiprocessing as mp
if mp.current_process().name != 'MainProcess':
    matplotlib.use('Agg')  # Non-interactive backend for worker processes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import numpy as np
import time
from numba import njit
import os


@njit(fastmath=True)                
def attractor(x, y, num_steps, a, b, c, d):
    """
    Calculate the Peter de Jong attractor sequence.
    
    coefficients
    ----------
    x : np.ndarray
        Array of x coordinates
    y : np.ndarray
        Array of y coordinates
    num_steps : int
        Number of iteration steps
    a, b, c, d : float
        Attractor coefficients
        
    Returns
    -------
    x, y : tuple of np.ndarray
        Updated coordinate arrays
    """
    for i in range(1, int(num_steps)):
        x[i] = np.sin(a * y[i - 1]) - np.cos(b * x[i - 1])
        y[i] = np.sin(c * x[i - 1]) - np.cos(d * y[i - 1]) 
    return x, y


def darken_cmap(function, cmap):
    """
    Apply a function to darken an existing colormap.
    
    From https://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
    
    coefficients
    ----------
    function : callable
        Function that operates on vectors of shape 3: [r, g, b]
    cmap : matplotlib.colors.Colormap
        Input colormap to transform
        
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Transformed colormap
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # First get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT

    def reduced_cmap(step):
        return np.array(cmap(step)[0:3])

    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red', 'green', 'blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


def pick_random_colormap(print_choice=False, show_plot=False):
    """
    Pick a random colormap from matplotlib, avoiding visually unappealing ones.
    
    Adapted from the Paint-Pours project for consistency.
    
    coefficients
    ----------
    print_choice : bool, optional
        If True, print the chosen colormap name (default is False).
    show_plot : bool, optional
        If True, display the chosen colormap (default is False).

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The randomly chosen matplotlib colormap.
    """
    # Some colormaps that don't work well for this kind of art
    bad_cmaps = ['flag', 'Accent', 'gist_stern', 'gist_grey', 'gist_yerg', 'grey', 'Greys', 'twilight_shifted', 
                 'Paired', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20c', 'tab20b', 'binary', 
                 'Pastel1', 'Pastel2', 'gist_yarg', 'gist_gray', 'brg', 'CMRmap', 'gist_ncar', 'gist_rainbow', 
                 'hsv', 'terrain', 'gnuplot2', 'nipy_spectral', 'prism']
    
    # Generate a list of all colormaps that don't contain "_r" in their name
    non_reversed_colormaps = [x for x in plt.colormaps() if '_r' not in x]

    # Pick a random colormap
    cmap = plt.cm.get_cmap(np.random.choice(non_reversed_colormaps))

    # Re-pick if it chose an ugly one
    while any(cmap.name in s for s in bad_cmaps):
        cmap = plt.cm.get_cmap(np.random.choice(non_reversed_colormaps))
    
    if print_choice:
        print(f'\t Chosen base colormap: {cmap.name}')
    
    if show_plot:
        # Only show plot in main process to avoid worker process GUI issues
        if mp.current_process().name == 'MainProcess':
            fig, ax = plt.subplots(figsize=(12, 2))
            ax.imshow(np.outer(np.ones(100), np.arange(0, 1, 0.0001)), cmap=cmap, origin='lower', extent=[0, 1, 0, 0.1])
            ax.set_yticks([])
            ax.set_title(f'Your randomly-chosen base colormap: {cmap.name}')
            fig.tight_layout()
            plt.show(block=False)
        else:
            print(f'Worker process: chosen colormap {cmap.name} (plot suppressed)')
    
    return cmap


class StrangeAttractor:
    """
    A class for generating Peter de Jong strange attractor images.
    
    Inspired by the PaintPour class structure for consistency and ease of use.
    """
    
    def __init__(self, 
                 image_resolution=[1920, 1080],
                 timesteps=1e7,
                 coefficient_values=None,
                 coefficient_max=5,
                 output_directory='outputs/',
                 show_troubleshooting_plots=False,
                 display_image=True,
                 save_image=True,
                 gauss_smoothing=0.5,
                 colormap_name=None,
                 background_color=None,
                 chunk_size=1e7,
                 **kwargs):
        """
        Initialize the StrangeAttractor object.
        
        coefficients
        ----------
        image_resolution : list, optional
            [width, height] in pixels (default is [1920, 1080])
        timesteps : float, optional  
            Number of points to calculate (default is 1e7)
        coefficient_values : list, optional
            Specific [a, b, c, d] values. If None, will be randomized
        coefficient_max : float, optional
            Maximum absolute value for random coefficients (default is 10)
        output_directory : str, optional
            Directory to save images (default is 'outputs/')
        show_troubleshooting_plots : bool, optional
            Show intermediate plots for troubleshooting (default is False)
        display_image : bool, optional
            Display the final image (default is True)
        save_image : bool, optional
            Save the image to disk (default is True)
        gauss_smoothing : float, optional
            Standard deviation for Gaussian smoothing (default is 0.5, set to 0 for no smoothing)
        colormap_name : str, optional
            Specific colormap name, if None will pick randomly
        background_color : str, optional
            Color name for background (set_under value). If None, uses lowest colormap value
        chunk_size : float, optional
            Size of calculation chunks to manage memory (default is 1e7)
        """
        
        # Store coefficients
        self.image_resolution = image_resolution
        self.timesteps = int(timesteps)
        self.coefficient_values = coefficient_values
        self.coefficient_max = coefficient_max
        self.output_directory = output_directory
        self.show_troubleshooting_plots = show_troubleshooting_plots
        self.display_image = display_image
        self.save_image = save_image
        self.gauss_smoothing = gauss_smoothing
        self.colormap_name = colormap_name
        self.background_color = background_color
        self.chunk_size = int(chunk_size)
        
        # Initialize coefficients
        self._initialize_coefficients()
        
        # Results storage
        self.histogram = None
        self.final_image = None
        self.colormap = None
        self.filename = None
        
    def _initialize_coefficients(self):
        """Initialize the attractor coefficients a, b, c, d."""
        if self.coefficient_values is not None:
            if len(self.coefficient_values) != 4:
                raise ValueError("coefficient_values must contain exactly 4 values [a, b, c, d]")
            self.a, self.b, self.c, self.d = self.coefficient_values
        else:
            # Generate random coefficients
            self.a = np.random.uniform(-self.coefficient_max, self.coefficient_max)
            self.b = np.random.uniform(-self.coefficient_max, self.coefficient_max)  
            self.c = np.random.uniform(-self.coefficient_max, self.coefficient_max)
            self.d = np.random.uniform(-self.coefficient_max, self.coefficient_max)
            
        print(f'Using coefficients: a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f}, d={self.d:.3f}')
    
    def _select_colormap(self):
        """Select and configure the colormap for the image."""
        if self.colormap_name is None:
            # Pick random colormap using the utility function
            self.colormap = pick_random_colormap(print_choice=True, show_plot=self.show_troubleshooting_plots)
        else:
            # Use specified colormap
            self.colormap = cm.get_cmap(self.colormap_name)
            if self.show_troubleshooting_plots:
                print(f'Using specified colormap: {self.colormap.name}')
        
        # Handle special colormap cases (legacy behavior)
        if self.colormap.name == 'ocean':
            if self.background_color is None:
                self.colormap.set_under('navy')  # Keep original behavior
            else:
                self.colormap.set_under(self.background_color)
        elif self.colormap.name == 'BuPu':
            self.colormap = darken_cmap(lambda x: x * 0.75, self.colormap)
            self.colormap.name = 'BuPu_darkened'
            if self.background_color is None:
                self.colormap.set_under('black')  # Keep original behavior
            else:
                self.colormap.set_under(self.background_color)
        else:
            # For all other colormaps, set background color
            if self.background_color is not None:
                self.colormap.set_under(self.background_color)
            else:
                # Use the lowest value in the colormap
                lowest_color = self.colormap(0.0)  # Get color at position 0
                self.colormap.set_under(lowest_color)
        
        return self.colormap
    
    def calculate_attractor(self):
        """Calculate the strange attractor points."""
        print('Starting calculations...')
        
        # Initialize
        x = np.zeros(self.chunk_size)
        y = np.zeros(self.chunk_size)
        
        # Calculate number of chunks
        num_chunks = round(self.timesteps / self.chunk_size)
        
        # Calculate first chunk
        x, y = attractor(x, y, self.chunk_size, self.a, self.b, self.c, self.d)
        
        # Create initial histogram for this first chunk and save the bin edges
        self.histogram, bin_edges_x, bin_edges_y = np.histogram2d(x, y, bins=self.image_resolution)
        print(f'Finished chunk 1 of {num_chunks}')
        
        # Calculate remaining chunks if needed
        if self.timesteps > self.chunk_size:
            for i in range(1, num_chunks):
                # Pick random starting point from last chunk, to ensure this chunk isn't identical to the first
                x[0] = np.random.choice(x)
                y[0] = np.random.choice(y)

                x, y = attractor(x, y, self.chunk_size, self.a, self.b, self.c, self.d)

                # Histogram this chunk and add to histogram using the same bin edges
                chunk_hist, _, _ = np.histogram2d(x, y, bins=[bin_edges_x, bin_edges_y])
                self.histogram += chunk_hist
                
                print(f'Finished chunk {i + 1} of {num_chunks}')
        
        # Log-scale the histogram (from original script)
        self.histogram = np.log10(self.histogram + 0.01)
        
        return self.histogram
    
    def process_image(self):
        """Apply smoothing and prepare final image."""
        if self.histogram is None:
            raise ValueError("Must calculate attractor first")
        
        if self.gauss_smoothing > 0:
            print(f'Smoothing image with sigma={self.gauss_smoothing}...')
            self.final_image = gaussian_filter(self.histogram, self.gauss_smoothing, 0)
        else:
            self.final_image = self.histogram.copy()
        
        # Rotate the histogram array to match scatterplot orientation (from original script)
        self.final_image = np.transpose(np.flip(self.final_image, 1))
        
        return self.final_image
    
    def create_plot(self):
        """Create and display/save the final plot."""
        if self.final_image is None:
            raise ValueError("Must process image first")
        
        # Only create plots in main process to avoid worker process GUI issues
        if mp.current_process().name != 'MainProcess':
            print("Skipping plot creation in worker process")
            return None, None
        
        # Select colormap
        colormap = self._select_colormap()
        
        # Create figure with exact dimensions
        my_dpi = 120
        fig, ax = plt.subplots(figsize=(self.image_resolution[0] / my_dpi, self.image_resolution[1] / my_dpi), 
                              dpi=my_dpi, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])  # Make plot take up entire figure
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Display image with colormap
        if colormap.name == 'ocean':
            ax.imshow(self.final_image, cmap=colormap, aspect='equal', vmin=self.histogram.min() + 0.0001)
        elif colormap.name == 'customdark':
            ax.imshow(self.final_image, cmap=colormap, aspect='equal', vmin=self.histogram.min() + 0.0001)
        else:
            ax.imshow(self.final_image, cmap=colormap, aspect='equal', vmin=self.histogram.min() + 0.0000)
        
        if self.save_image:
            self._save_image(fig, my_dpi)
        
        if self.display_image:
            plt.show()
        else:
            plt.close(fig)
        
        return fig, ax
    
    def _save_image(self, fig, dpi):
        """Save the image to disk."""
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        # Generate filename with colormap name first
        timesteps_str = "{:.0e}".format(self.timesteps)
        params_str = f"{self.a:.3f}-{self.b:.3f}-{self.c:.3f}-{self.d:.3f}"
        self.filename = f"{self.colormap.name}-{timesteps_str}steps-{params_str}.png"
        
        filepath = os.path.join(self.output_directory, self.filename)
        fig.savefig(filepath, format='png', dpi=dpi)
        print(f'Image saved to: {filepath}')
        
    def generate(self):
        """
        Complete generation pipeline: calculate, process, and plot the attractor.
        
        Returns
        -------
        final_image : np.ndarray
            The final processed image array
        """
        self.calculate_attractor()
        
        final_image = self.process_image()
        fig, ax = self.create_plot()
        
        return final_image


def generate_strange_attractor_images(num_images=1, **kwargs):
    """
    Generate multiple strange attractor images with the given coefficients.
    
    Inspired by the generate_paint_pour_images function for consistency.
    
    coefficients
    ----------
    num_images : int, optional
        Number of images to generate (default is 1).
    **kwargs : keyword arguments
        Any arguments accepted by the StrangeAttractor class constructor.

    Returns
    -------
    results : list of tuples
        List of (image, attractor_object) tuples for each generated image.
    """
    results = []
    
    for i in range(num_images):
        plt.close('all')  # Close any existing plots
        start_time = time.time()
        print(f'\n=== Generating image {i + 1} of {num_images} ===')
        
        # Create and generate attractor
        attractor = StrangeAttractor(**kwargs)
        image = attractor.generate()
        
        if image is not None:
            results.append((image, attractor))
        else:
            print(f'Image {i + 1} failed due to bad coefficients, skipping...')

        # Print elapsed time for each image
        elapsed_time = time.time() - start_time
        print(f'Image generation completed in {elapsed_time:.2f} seconds')
    
    print(f'\nSuccessfully generated {len(results)} images!')
    return results
