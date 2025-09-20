"""
Strange Attractor Sandbox

A user-friendly script for generating Peter de Jong strange attractor images.
This script allows you to easily specify coefficients and generate multiple images.

"""

import strange_attractor_utilities as sa_utils
import matplotlib

# Set the matplotlib backend (adjust if needed for your system)
# Common options: 'QtAgg', 'TkAgg', 'Agg' (for headless)
matplotlib.use('QtAgg')

# ######## USER-DEFINED VARIABLES #########

# Image settings
image_resolution = [1920, 1080]  # [width, height] in pixels
# image_resolution = [3840, 2160]   # 4K resolution
# image_resolution = [2560, 1440]   # QHD resolution

# Generation settings
timesteps = 1e8  # Number of points to calculate (1E8 for high quality, 1E6 for quick testing)
num_images = 1   # How many separate images to generate

# Coefficient settings
coefficient_values = None          # Specific [a, b, c, d] values (set to None for random)
coefficient_values = [2.76, -1.15, 0.50, -4.5]  # Example specific values
coefficient_max = 5               # Maximum absolute value for random coefficients

# Output settings
output_directory = 'outputs/'    # Where to save images
save_image = True               # Save images to disk
display_image = True               # Display images on screen

# Visual settings
gauss_smoothing = 0.5          # Standard deviation for smoothing (set to 0 for no smoothing)
# colormap_name = None           # Specific colormap name (None for random selection)
colormap_name = 'inferno'     # Example: use a specific colormap
background_color = None        # Background color (None = use lowest colormap value)
# background_color = 'black'    # Example: black background
# background_color = 'white'    # Example: white background

# Advanced settings
show_troubleshooting_plots = False  # Show intermediate plots for debugging

# ######## GENERATION CODE #########
print(f"\nGenerating images with {timesteps:.0f} timesteps each...")
results = sa_utils.generate_strange_attractor_images(
    num_images=num_images,
    image_resolution=image_resolution,
    timesteps=timesteps,
    coefficient_values=coefficient_values,
    coefficient_max=coefficient_max,
    output_directory=output_directory,
    show_troubleshooting_plots=show_troubleshooting_plots,
    display_image=display_image,
    save_image=save_image,
    gauss_smoothing=gauss_smoothing,
    colormap_name=colormap_name,
    background_color=background_color
)

print("=" * 50)
print(f"Generation complete! Successfully created {len(results)} images.")
