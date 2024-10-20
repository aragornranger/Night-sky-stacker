import rawpy
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import RangeSlider
import numpy as np
from os import listdir

def on_select(eclick, erelease): # Callback function for ROI selection
    global roi

    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    roi = [int(y1), int(x1), int(y2), int(x2)]
    print(roi)

def update(val):
    global mask

    maskReversed = (ref < val[0]) | (ref > val[1])
    ax.imshow(maskReversed, cmap = 'hot', alpha = 0.3)
    mask = (ref >= val[0]) & (ref <= val[1])
    mask = mask.astype(int)
    return val



def convolve2d_cust(x,y):
#    h,w = x.shape
#    x = np.pad(x,((h // 2, h // 2),(w // 2, w // 2)), "constant", constant_values = 0)
#    y = np.pad(y,((h // 2, h // 2),(w // 2, w // 2)), "constant", constant_values = 0)

    im1 = np.fft.fft2(x)
    im2 = np.fft.fft2(y)

    return np.real(np.fft.fftshift(np.fft.ifft2(np.multiply(im1, im2))))

global roi 

#
# Raw image reading with Rawypy
#

path = 'lights/'
files = listdir(path)
files.sort()

print(files)

frames = []
shifts = []

for name in files:
    with rawpy.imread(path + name) as raw:
        frames.append(raw.raw_image.copy())

print('read end')


h = len(frames[0])
w = len(frames[0][0])
print(h,w)

#
# Draw ROI interactively on the overlap between the first and the last frame
#

selection = frames[0] + frames[-1] 
fig, ax = plt.subplots()
ax.imshow(selection, cmap = 'gray') 

toggle_selector = RectangleSelector(ax, on_select,
                                        useblit=True,
                                        button=[1],  # Only left click allowed
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)

plt.show()

ref = frames[0][roi[0]:roi[2], roi[1]:roi[3]]
global mask

#
# With the ROI selected, threshold the ref frame to get the comet core
#

fig, ax = plt.subplots()
ax.imshow(ref, cmap = 'gray')
slider_ax = fig.add_axes([0.20, 0, 0.60, 0.03])
slider = RangeSlider(slider_ax, 'Threshold', ref.min(), ref.max())
slider.on_changed(update)

plt.show()

ref = ref * mask

#
# Convolution of the ref frame with each frame to find maximum overlap
#

for i in range(len(frames)):
    temp = frames[i][roi[0]:roi[2], roi[1]:roi[3]] 
    conv = convolve2d_cust(ref, temp)

    convolutionMax = np.argmax(conv)
    row, col = np.unravel_index(convolutionMax, conv.shape)

    shifts.append([row, col])

print(shifts)

for i in range(1, len(shifts)):
    row = shifts[i][0] - shifts[0][0]
    col = shifts[i][1] - shifts[0][1]

    print(row, col)

    if row >= 0:
        top = row
        bottom = h
        padVertical = (0, row)
    else:
        top = 0
        bottom = h + row
        padVertical = (-row, 0)

    if col >= 0:
        left = col
        right = w
        padHorizontal = (0, col)
    else:
        left = 0
        right = w + col
        padHorizontal = (-col, 0)

    temp = frames[i][top:bottom, left:right]
    #print(temp.shape)
    temp = np.pad(temp, (padVertical, padHorizontal), mode='constant', constant_values=0)
    frames[i] = temp

    #print(temp.shape)

stacked = frames[0].astype(float)

for i in range(1, len(frames)):
    stacked += (frames[i] - stacked) / (float(i) + 1.)


plt.imshow(stacked, cmap = 'gray')
plt.show()

