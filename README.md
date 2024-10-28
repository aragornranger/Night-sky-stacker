# The night sky stacker project
This is a script for aligning and stacking raw images of stars. 
Current version works on realizing alignment and stacking of comet photos. Alignment data are calculated using a convolution-based algorithm.

Packages needed:
`matplotlib` `rawpy`

# Recent update

Implemented reading and averaging functions for the correction frames.
Implemented image correction processes based on bias, dark, and flat fields.
Optimized the stacking procedure. The images are read and averaged individually to avoid overflow and reduce memory usage. 

# Tutorial

- The images to be stacked are stored under the directory "lights/", preferably in time order.
- Correction images are stored under directories "biases/", "darks/", and "flats/" respectively.
- The script reads the target images and plots the sum of the first and last frames.
- The user chooses the ROI by selecting a rectangle box around the comet's core. The ROI should cover the core movement with an additional margin. Close the window after choosing the ROI.
- Drag the slider at the bottom of the new plot to mask out the comet's core. Close the window after the threshold is set.
- The stacked image is plotted in a new window.

# To be implemented

- Raw image processing.
  - Customized code for Bayer interpolation.
- Improvement on the averaging algorithm for better dynamic range of the results.
- Read config file for setting the processing parameters.
- Aligning algorithm for landscape photos and deep sky photos.
