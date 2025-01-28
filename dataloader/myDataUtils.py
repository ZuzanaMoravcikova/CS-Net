import numpy as np
import scipy.ndimage.filters as flt
import warnings
import pylab as pl
from time import sleep
import os
import napari
from scipy.ndimage import gaussian_filter, convolve
import cv2
from skimage.filters import threshold_li
from skimage import io
import matplotlib.pyplot as plt

# 	kappa controls conduction as a function of gradient.  If kappa is low
# 	small intensity gradients are able to block conduction and hence diffusion
# 	across step edges.  A large value reduces the influence of intensity
# 	gradients on conduction.
#
# 	gamma controls speed of diffusion (you usually want it at a maximum of
# 	0.25)
#
# 	step is used to scale the gradients in case the spacing between adjacent
# 	pixels differs in the x and y axes
#
# 	Diffusion equation 1 favours high contrast edges over low contrast ones.
# 	Diffusion equation 2 favours wide regions over smaller ones.


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1.0, 1.0), sigma=0, option=2, ploton=False):
    """
    Anisotropic diffusion.

    Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
        img    - input image
        niter  - number of iterations
        kappa  - conduction coefficient 20-100 ?
        gamma  - max value of .25 for stability
        step   - tuple, the distance between adjacent pixels in (y,x)
        option - 1 Perona Malik diffusion equation No 1
                 2 Perona Malik diffusion equation No 2
        ploton - if True, the image will be plotted on every iteration

    Returns:
        imgout - diffused image.

    Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using anisotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.
    """
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    img = img.astype('float32')
    imgout = img.copy()

    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    if ploton:
        fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(20, 5.5), num="Anisotropic diffusion")

        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")
        fig.canvas.draw()

    for ii in range(1, niter):
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        if sigma > 0:
            deltaSf = flt.gaussian_filter(deltaS, sigma)
            deltaEf = flt.gaussian_filter(deltaE, sigma)
        else:
            deltaSf = deltaS
            deltaEf = deltaE

        if option == 1:
            gS = np.exp(-(deltaSf / kappa) ** 2) / step[0]
            gE = np.exp(-(deltaEf / kappa) ** 2) / step[1]
        elif option == 2:
            gS = 1.0 / (1.0 + (deltaSf / kappa) ** 2) / step[0]
            gE = 1.0 / (1.0 + (deltaEf / kappa) ** 2) / step[1]

        E = gE * deltaE
        S = gS * deltaS

        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        imgout += gamma * (NS + EW)

        if ploton:
            ih.set_data(imgout)
            ax2.set_title(f"Iteration {ii}")
            fig.canvas.draw()

    return imgout

# kappa controls conduction as a function of gradient.  If kappa is low
# 	small intensity gradients are able to block conduction and hence diffusion
# 	across step edges.  A large value reduces the influence of intensity
# 	gradients on conduction.
#
# 	gamma controls speed of diffusion (you usually want it at a maximum of
# 	0.25)
#
# 	step is used to scale the gradients in case the spacing between adjacent
# 	pixels differs in the x,y and/or z axes
#
# 	Diffusion equation 1 favours high contrast edges over low contrast ones.
# 	Diffusion equation 2 favours wide regions over smaller ones.

def anisodiff3(stack, niter=1, kappa=50, gamma=0.1, step=(1.0, 1.0, 1.0), option=2, ploton=False):
    """
    3D Anisotropic diffusion.

    Usage:
        stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
        stack  - input stack
        niter  - number of iterations
        kappa  - conduction coefficient 20-100 ?
        gamma  - max value of .25 for stability
        step   - tuple, the distance between adjacent pixels in (z, y, x)
        option - 1 Perona Malik diffusion equation No 1
                 2 Perona Malik diffusion equation No 2
        ploton - if True, the middle z-plane will be plotted on every iteration

    Returns:
        stackout - diffused stack.

    Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using anisotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.
    """
    if stack.ndim == 4:
        warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

    stack = stack.astype('float32')
    stackout = stack.copy()

    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    if ploton:
        import pylab as pl

        showplane = stack.shape[0] // 2

        fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(20, 5.5), num="Anisotropic diffusion")

        ax1.imshow(stack[showplane, ...].squeeze(), interpolation='nearest')
        ih = ax2.imshow(stackout[showplane, ...].squeeze(), interpolation='nearest', animated=True)
        ax1.set_title(f"Original stack (Z = {showplane})")
        ax2.set_title("Iteration 0")
        fig.canvas.draw()

    for ii in range(1, niter):
        deltaD[:-1, :, :] = np.diff(stackout, axis=0)
        deltaS[:, :-1, :] = np.diff(stackout, axis=1)
        deltaE[:, :, :-1] = np.diff(stackout, axis=2)

        if option == 1:
            gD = np.exp(-(deltaD / kappa) ** 2) / step[0]
            gS = np.exp(-(deltaS / kappa) ** 2) / step[1]
            gE = np.exp(-(deltaE / kappa) ** 2) / step[2]
        elif option == 2:
            gD = 1.0 / (1.0 + (deltaD / kappa) ** 2) / step[0]
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2) / step[1]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2) / step[2]

        D = gD * deltaD
        E = gE * deltaE
        S = gS * deltaS

        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        stackout += gamma * (UD + NS + EW)

        if ploton:
            ih.set_data(stackout[showplane, ...].squeeze())
            ax2.set_title(f"Iteration {ii}")
            fig.canvas.draw()

    return stackout

def main():
    root_dir = "C:/muni/DP/TNT_data"
    dirs = os.listdir(root_dir)[0:-1]
    print(dirs[3])
    dir_idx = 0
    cur_dir = os.path.join(root_dir, dirs[dir_idx])
    files = os.listdir(cur_dir)
    file_num = [2, 2, 76, 6]
    img = io.imread(os.path.join(cur_dir, files[file_num[dir_idx]]))
    img = img.astype('float32')
    m = np.mean(img)
    s = np.std(img)
    nimg = (img - m) / s
    plt.imshow(nimg, cmap='gray')
    plt.show()

    params = {
        'niter': 10,
        'kappa': 40,
        'gamma': 0.1,
        'step': (1, 1),
        'sigma': 0,
        'option': 1,
        'ploton': False
    }
    fimg = anisodiff(nimg, niter=params['niter'], kappa=params['kappa'], gamma=params['gamma'], step=params['step'],
                     sigma=params['sigma'], option=params['option'], ploton=params['ploton'])

    viewer = napari.Viewer()
    viewer.add_image(nimg, name='Original', colormap='gray')
    viewer.add_image(fimg, name='Filtered', colormap='gray')
    napari.run()


# run main
#main()
