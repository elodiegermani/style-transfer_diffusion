from glob import glob
from nilearn.image import resample_img, resample_to_img
from nilearn import datasets, plotting, masking, image
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os.path as op
import os

def get_imlist(images):
    '''
    Search for the list of images in the repository "images" that are in NiFti file format.
    
    Parameters:
        - images: str, path to the directory containing images or list, list of the paths to images
        
    Return:
        - files: list, list containing all paths to the images
        - inpdir: boolean, True if images is the directory containing the images, False otherwise
    '''
    files = sorted(glob(images))
    inpdir = True

    return files, inpdir

def compute_intersection_mask(data, pipeline):
    '''
    Compute intersection mask of images located in a directory and resample this mask to MNI.

    Parameters
    ----------
    data : list
        List of images images

    Returns
    -------
    mask : Nifti1Image
        Mask image
    '''
    img_list = []
    mask_list = []

    target = datasets.load_mni152_template(resolution=4)#nib.load(fsl.Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')) #datasets.load_mni152_gm_template(4)

    print('Computing mask for pipeline', pipeline)
    data_pipeline = [f for f in data if pipeline in f]

    print('Number of masks:', len(data_pipeline))
    for fpath in data_pipeline:
        img = nib.load(fpath)

        mask_img = image.binarize_img(img)

        resampled_mask = image.resample_to_img(
                    mask_img,
                    target,
                    interpolation='nearest')

        mask_list.append(resampled_mask)
    print('All subjects masks resampled.')

    mask = masking.intersect_masks(mask_list, threshold=1)

    return mask

def preprocessing(data_dir, output_dir):
    '''
    Preprocess all maps that are stored in the 'original' repository of the data_dir. 
    Store these maps in subdirectories of the data_dir corresponding to the preprocessing step applied.


    Parameters:
        - data_dir, str: path to directory where 'original' directory containing all original images is stored
        - output_dir, str: path to directory where 'preprocessed' directory is stored and where all preprocessed images will be stored.

    '''
    print('----PREPROCESSING----')
    # Get image list to preprocess
    img_list, input_dir = get_imlist(op.join(data_dir))
        
    # Create dirs to save images
    if not op.isdir(op.join(output_dir, f'normalized')):
        os.mkdir(op.join(output_dir, f'normalized'))

    standard = datasets.load_mni152_template(resolution=4)
    target_affine = standard.affine.copy()
    target_affine[:3,:3] = np.sign(target_affine[:3,:3]) * 4
    target_shape = (50,59,48)

    if not os.path.exists(op.join(output_dir, f'normalized', 'group_mask.nii.gz')):
        print('Computing group-level mask')
        mask_list = []
        for pipeline in ['fsl-5-0-0', 'fsl-5-0-1', 
        'fsl-8-0-0', 'fsl-8-0-1', 'fsl-5-6-0', 'fsl-5-6-1', 'fsl-8-6-0', 'fsl-8-6-1',
        'fsl-5-24-0', 'fsl-5-24-1', 'fsl-8-24-0', 'fsl-8-24-1',
        'spm-5-0-0', 'spm-5-0-1', 
        'spm-8-0-0', 'spm-8-0-1', 'spm-5-6-0', 'spm-5-6-1', 'spm-8-6-0', 'spm-8-6-1',
        'spm-5-24-0', 'spm-5-24-1', 'spm-8-24-0', 'spm-8-24-1']:
            mask_list.append(compute_intersection_mask(img_list, pipeline))

        mask = masking.intersect_masks(mask_list, threshold=1)
        nib.save(mask, op.join(output_dir, f'normalized', 'group_mask.nii.gz'))
    else:
        print('Mask exists.')
        mask = nib.load(op.join(output_dir, f'normalized', 'group_mask.nii.gz'))
    
    for idx, img in enumerate(img_list):
        print('Image', img)

        nib_img = nib.load(img)
        img_data = nib_img.get_fdata()
        img_data = np.nan_to_num(img_data)
        img_affine = nib_img.affine
        nib_img = nib.Nifti1Image(img_data, img_affine)
        
        print('Original shape of image ', idx+1, ':',  nib_img.shape)

        try:
            print("Resampling image {0} of {1}...".format(idx + 1, len(img_list)))
            
            res_img = resample_to_img(nib_img, standard, interpolation='continuous')

            print('New shape for image', idx, res_img.shape)            
            print("Masking image {0} of {1}...".format(idx + 1, len(img_list)))
            
            mask_data = mask.get_fdata()
            res_img_data = res_img.get_fdata()
            
            res_masked_img_data = res_img_data * mask_data
            
            res_masked_img = nib.Nifti1Image(res_masked_img_data, res_img.affine)
            
            norm_img_data = res_masked_img_data.copy().astype(float)
            norm_img_data = np.nan_to_num(norm_img_data)
            norm_img_data *= 1.0/np.abs(norm_img_data).max()
            norm_img_data = norm_img_data[0:48,0:56,0:48]
            norm_img = nib.Nifti1Image(norm_img_data, res_img.affine)
            
            #nib.save(norm_img, op.join(output_dir, f'resampled-masked', op.basename(img))) # Save original image resampled and normalized
            nib.save(norm_img, op.join(output_dir, f'normalized', op.basename(img)))

            print(f"Image {idx} : DONE.")

        except Exception as e:
            print("Failed!")
            print(e)
            continue