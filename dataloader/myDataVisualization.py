import napari
import numpy as np
from tifffile import imread
import os
import itk


print(dir(itk))

from myDataUtils import anisodiff3

def stack_images_1(data_dir, t_value="001", c_value="001"):
    images = []
    files = os.listdir(data_dir)
    for file in files:
        if file.split("T")[1][0:3] == t_value and file.split("C")[1][0:3] == c_value:
            images.append(imread(os.path.join(data_dir, file)))

    return np.stack(images, axis=0)


def stack_images_2(data_dir, timepoint="t001", channel="c001"):
    images = []
    files = os.listdir(data_dir)
    for file in files:
        if file.endswith(f"{channel}.tif") and file.split("_")[2] == timepoint:
            images.append(imread(os.path.join(data_dir, file)))

    return np.stack(images, axis=0)


def apply_ced_filter(img_3d,
                     time_step=0.05,
                     num_iterations=10,
                     conductance=1.0):
    """
    Apply Coherence Enhancing Diffusion (CED) filter to a 3D NumPy array.

    Parameters:
    -----------
    img_3d : np.ndarray
        3D NumPy array of shape (D, H, W).
    time_step : float
        Diffusion time step. Must be small enough for stability.
    num_iterations : int
        Number of diffusion iterations to apply.
    conductance : float
        Conductance parameter controlling diffusion strength.

    Returns:
    --------
    out_np : np.ndarray
        The filtered 3D NumPy array.
    """

    # Ensure the array is in float format (ITK typically needs float or double)
    img_3d = img_3d.astype(np.float32)

    # Convert NumPy array to ITK image
    itk_image = itk.GetImageFromArray(img_3d)

    # Create the filter. We let ITK infer the ImageType from itk_image.
    ImageType = type(itk_image)
    ced_filter = itk.GradientAnisotropicDiffusionImageFilter[ImageType, ImageType].New()
    # ced_filter = itk.CurvatureAnisotropicDiffusionImageFilter[ImageType, ImageType].New()
    # ced_filter = itk.VectorCurvatureAnisotropicDiffusionImageFilter[ImageType, ImageType].New()
    ced_filter.SetInput(itk_image)

    # Set the main PDE parameters
    ced_filter.SetTimeStep(time_step)
    ced_filter.SetNumberOfIterations(num_iterations)
    ced_filter.SetConductanceParameter(conductance)

    # You can also try other optional settings:
    # ced_filter.SetEnhancement('cED')  # 'cED', 'eED', 'cED2', 'sED', ...
    # ced_filter.SetNoiseScale(2.0)
    # ced_filter.SetFeatureScale(2.0)
    # ced_filter.SetAlpha(0.5)

    # Run the filter
    ced_filter.Update()

    # Get the output as an ITK image
    out_itk = ced_filter.GetOutput()

    # Convert the ITK image back to a NumPy array
    out_np = itk.GetArrayFromImage(out_itk)

    return out_np


def main():
    root_dir = "C:/muni/DP/TNT_data"
    dirs = os.listdir(root_dir)[0:-1]

    #MitoRoundtripBundling
    # data_dir = os.path.join(root_dir, dirs[3])
    # img_3d = stack_images_2(data_dir, "t002", "c001")

    #mitoRoundtrip2layers
    # data_dir = os.path.join(root_dir, dirs[2])
    # img_3d = stack_images_2(data_dir, "t001", "c001")

    # 180322_Sqh-mCh Tub-GFP 16h_110.tif.files
    data_dir = os.path.join(root_dir, dirs[1])
    img_3d = stack_images_1(data_dir, "001", "001")

    # 161223_ptcG4_x_mito-mCh_5
    # data_dir = os.path.join(root_dir, dirs[0])
    # img_3d = stack_images_1(data_dir, "001", "001")

    # img_diffused = anisodiff3(img_3d, niter=10, kappa=80, gamma=0.25, step=(1, 1, 1), option=1, ploton=False)
    # ced_filter = itk.CoherenceEnhancingDiffusionFilter[itk.Image[itk.F, 3]].New()
    # ced_filter.SetInput(image)

    img_diffused = apply_ced_filter(img_3d, time_step=0.05, num_iterations=10, conductance=1.0)




    viewer = napari.Viewer()
    viewer.add_image(img_3d, name="Original", scale=(1, 1, 1))
    viewer.add_image(img_diffused, name="Diffused", scale=(1, 1, 1))
    napari.run()

if __name__ == "__main__":
    main()