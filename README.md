<img width="1641" height="941" alt="image" src="https://github.com/user-attachments/assets/83e30039-74f5-40d3-8cf2-c0edee55d596" />

## Overview

This project provides tools for generating and visualizing De Jong strange attractors. Users can specify as many or as few parameters as they'd like, and the remainder will be chosen automatically.

The intent of this tool is that users can quickly produce many widely-varying, randomly-generated strange attractors, identify their favorites, and re-render them in high-quality for use as wallpapers on their phone or computer. 

## Quick Start

1. **Basic usage**: Run the sandbox script in your IDE
   - Open `strange_attractor_sandbox.py` in your IDE
   - Adjust the coefficients at the top of the file as desired
   - Run the script (F5 in most IDEs)

2. **Programmatic usage**: Import the utilities in a new script
   ```python
   import strange_attractor_utilities as sa_utils
   
   # Generate a single image with random coefficients
   attractor = sa_utils.StrangeAttractor(
       image_resolution=[1920, 1080],
       timesteps=1e7
   )
   image = attractor.generate()
   
   # Generate a single image with specific coefficients
   attractor = sa_utils.StrangeAttractor(
       image_resolution=[1920, 1080],
       timesteps=1e7,
       coefficient_values=[2.756, -1.145, 0.502, -4.5],
       gauss_smoothing=0.5,
   )
   image = attractor.generate()
   
   # Generate multiple images
   results = sa_utils.generate_strange_attractor_images(
       num_images=5,
       image_resolution=[1920, 1080],
       timesteps=1e7
   )
   ```

## Key Features

### Input Coefficient Control
- **Specific values**: Set `coefficient_values=[a, b, c, d]` for precise control, or leave `coefficient_values=None` (default) for random coefficients

### Image Quality
- **Resolution**: `image_resolution=[width, height]` in pixels
- **Quality**: `timesteps=1e7` (higher = better quality, but slower)
- **Smoothing**: `gauss_smoothing=0.5`to produce a slightly less pixelated image

### Output Options
- **Save images**: `save_image=True` with `output_directory='outputs/'`
- **Display**: `display_image=True` to show images on screen
- **Colormap**: `colormap_name='inferno'` or `None` for random selection
- **Background color**: `background_color='black'` or `None` for lowest colormap value
- **Smart filenames**: Now start with colormap name for easier sorting

### Advanced Features
- **Troubleshooting**: `show_troubleshooting_plots=True` for debugging
- **Memory management**: `chunk_size=1e7` for large calculations
- **Parallel processing**: `use_parallel=True` automatically uses all CPU cores for faster generation
- **Object-based**: All plots use matplotlib object-oriented interface

## Usage Recommendations

This project is designed around a three-phase workflow for creating stunning strange attractor art:

### 1. Exploration Phase
Generate many images with random coefficients to discover interesting attractors:
```python
results = sa_utils.generate_strange_attractor_images(
    num_images=20,
    image_resolution=[800, 600],    # Smaller for speed
    timesteps=1e7,                  # Lower quality for speed
    coefficient_max=5                 # Focused coefficient range
)
```

### 2. Selection Phase
Review the generated images and identify your favorites. Note the coefficient values from the filenames or console output. The filename format is: `{colormap}-{timesteps}steps-{a}-{b}-{c}-{d}.png`

### 3. Recreation Phase
Use the specific coefficients from your favorites to create high-quality final versions:
```python
attractor = sa_utils.StrangeAttractor(
    coefficient_values=[2.756, -1.145, 0.502, -4.5],  # From your favorite
    image_resolution=[3840, 2160],   # 4K resolution
    timesteps=1e8,                   # High quality
    colormap_name='inferno',         # Consistent colormap
    gauss_smoothing=0.6,            # Refined smoothing
    background_color='black'        # Clean black background
)
image = attractor.generate()
```

## Dependencies

- matplotlib
- numpy
- scipy
- numba (for fast calculation)
- time, os (standard library)

## Approximate Performance (with parallel processing)

- Small test (100x100, 1e6 steps): <1 second
- Medium quality (1920x1080, 1e7 steps): ~2 seconds
- High quality (1920x1080, 1e8 steps): ~6 seconds
- Extreme quality (2560x1440, 1e9 steps): ~25 seconds

