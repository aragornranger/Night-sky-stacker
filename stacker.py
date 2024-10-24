import rawpy
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import RangeSlider
import numpy as np
import tifffile
from os import listdir

def getAverage(fileList): # Calculate averaged image from the list
    for i in range(len(fileList)):
        print(f'Averaging {fileList[i]}...')
        currentFrame = read(fileList[i]).astype(float)
        if i == 0:
            averageFrame = currentFrame
        else:
            averageFrame += (currentFrame - averageFrame) / (float(i) + 1)
    
    return averageFrame

def on_select(eclick, erelease): # Callback function for ROI selection
    global roi

    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    roi = [int(y1), int(x1), int(y2), int(x2)]
    print(roi)


def read(filename):
    with rawpy.imread(filename) as raw:
        return raw.raw_image.copy()
    
def convolve2d_cust(x,y):
#    h,w = x.shape
#    x = np.pad(x,((h // 2, h // 2),(w // 2, w // 2)), "constant", constant_values = 0)
#    y = np.pad(y,((h // 2, h // 2),(w // 2, w // 2)), "constant", constant_values = 0)

    im1 = np.fft.fft2(x)
    im2 = np.fft.fft2(y)

    return np.real(np.fft.fftshift(np.fft.ifft2(np.multiply(im1, im2))))

def getROI(filelist):
    im1 = read(filelist[0])
    im2 = read(filelist[-1])

    toPlot = (im1 + im2) / 2
    fig, ax = plt.subplots()
    ax.imshow(toPlot, cmap = 'gray')

    toggle_selector = RectangleSelector(ax, on_select,
                                        useblit=True,
                                        button=[1],  # Only left click allowed
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)
    
    plt.show()

def getMask(ref):
    global mask

    def update(val):
        global mask

        mask = (ref >= val[0]) & (ref <= val[1])
        mask = mask.astype(int)

        maskPlot.set_data((mask == 0))
        fig.canvas.draw_idle()

    fig, ax = plt.subplots()
    ax.imshow(ref, cmap = 'gray')
    slider_ax = fig.add_axes([0.20, 0, 0.60, 0.03])
    slider = RangeSlider(slider_ax, 'Threshold', ref.min(), ref.max())
    reversedMask = (ref < slider.val[0]) | (ref > slider.val[1])
    mask = (reversedMask == False).astype(int)
    maskPlot = ax.imshow(reversedMask, cmap = 'Reds', alpha = 0.8)
    slider.on_changed(update)

    plt.show()


global roi
global mask 

folder = 'D:\\image_stacker_test_data\\' 
# Change to empty if you want to place the images in the same directory as the script
print(folder)
filesInFolder = listdir(folder)

#
# Read and get averaged dark noise image
#

if 'masterdark.tif' in filesInFolder:
    print('Master dark found, read from file')
    masterDark = tifffile.imread(folder + 'masterdark.tif')
else:
    print('Calculating dark frame...')
    darkPath = folder + 'darks\\'
    darkfiles = listdir(darkPath)
    darkList = [darkPath + file for file in darkfiles]
    masterDark = getAverage(darkList)
    tifffile.imwrite(folder + 'masterdark.tif', masterDark.astype('uint16'))
    del darkPath, darkfiles, darkList



#
# Read and get averaged bias image
#

if 'masterbias.tif' in filesInFolder:
    print('Master bias found, read from file')
    masterBias = tifffile.imread(folder + 'masterbias.tif')
else:
    print('Calculating bias frame...')
    biasPath = folder + 'biases\\'
    biasfiles = listdir(biasPath)
    biasList = [biasPath + file for file in biasfiles]
    masterBias = getAverage(biasList)
    tifffile.imwrite(folder + 'masterbias.tif', masterBias)
    del biasPath, biasfiles, biasList

#
# Read and get averaged flat field image
#

if 'masterflat.tif' in filesInFolder:
    print('Master flat found, read from file')
    masterFlat = tifffile.imread(folder + 'masterflat.tif')
else:
    print('Calculating flat field...')
    flatPath = folder + 'flats\\'
    flatfiles = listdir(flatPath)
    flatList = [flatPath + file for file in flatfiles]
    masterFlat = getAverage(flatList)
    tifffile.imwrite(folder + 'masterflat.tif', masterFlat)
    del flatfiles, flatPath, flatList

flat = masterFlat - masterBias
flat = flat / np.max(flat)

#
# Dir of light field images. Read first image as reference
#

lightPath = folder + 'lights\\'
files = listdir(lightPath)
files.sort()
filelist = [lightPath + file for file in files]   


print(files)

refFrame = read(filelist[0])



h = len(refFrame)
w = len(refFrame[0])
print(h,w)

#
# Draw ROI interactively on the overlap between the first and the last frame
#

getROI(filelist)

print(roi)

ref = refFrame[roi[0]:roi[2], roi[1]:roi[3]]


#
# With the ROI selected, threshold the ref frame to get the comet core
#

getMask(ref)

ref = ref * mask

#
# Convolution of the ref frame with each frame to find maximum overlap
#

for i in range(len(filelist)):
    print(f"Proccessing file {files[i]}")

    print(f"Reading {files[i]}...")
    currentFrame = read(filelist[i])
    currentROI = currentFrame[roi[0]:roi[2], roi[1]:roi[3]]

    print(f'Aligning {files[i]}...')
    conv = convolve2d_cust(ref, currentROI)

    convolutionMax = np.argmax(conv)
    row, col = np.unravel_index(convolutionMax, conv.shape)

    if i == 0:
        shiftsRef = (row, col)

    row = row - shiftsRef[0]
    col = col - shiftsRef[1]

    print(f"Alignment parameter for {files[i]} are Y:{row}, X:{col}")

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

    correctedFrame = currentFrame - masterDark
    correctedFrame = np.divide(correctedFrame, flat)
    temp = correctedFrame[top:bottom, left:right]
    #print(temp.shape)
    temp = np.pad(temp, (padVertical, padHorizontal), mode='constant', constant_values=0)
    
    if i == 0:
        stacked = temp.astype(float)
    else:
        stacked += (temp - stacked) / (float(i) + 1.)

    #print(temp.shape)

tifffile.imwrite(folder + "stacked.tif", stacked.astype('uint16'))
plt.imshow(stacked, cmap = 'gray', vmin = 0, vmax = np.max(stacked))
plt.show()
