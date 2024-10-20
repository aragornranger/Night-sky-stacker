# The night sky stacker project
This is a script for aligning and stacking raw images of stars. 
Currently working on realizing alignment and stacking of comet photos. Alignment data are calculated using a convolution based algorithm.

# Tutorial

- The images to be stacked are stored under the directory "lights".
- The script reads the target images and plots the sum of the first and last frames.
- The user chooses the ROI by selecting a rectangle box around the comet's core. The ROI should cover the core movement with an additional margin. Close the window after choosing the ROI.
- Drag the slider at the bottom of the new plot to mask out the comet's core. Close the window after the threshold is set.
- The stacked image is plotted in a new window.

# To be implemented

- Image correction with biases, darks, and flat-fields.
- Optimizing the stacking procedure for optimum memory management.
- Raw image processing.
