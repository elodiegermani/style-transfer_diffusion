from glob import glob
from nilearn import datasets, plotting, masking, image
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os.path as op
import os

def preprocessing(data_dir, output_dir, resolution):
    '''
    Preprocess all maps that are stored in the 'original' repository of the data_dir. 
    Store these maps in subdirectories of the data_dir corresponding to the preprocessing step applied.

    Parameters:
        - data_dir, str: path to directory where 'original' directory containing all original images is stored
        - output_dir, str: path to directory where 'preprocessed' directory is stored and where all preprocessed images will be stored.

    '''
    # Get image list to preprocess
    img_list = sorted(glob(f'{data_dir}/group-*_*_*_tstat.nii'))
        
    # Create dirs to save images
    if not op.isdir(op.join(output_dir, f'resampled_mni_masked_res_{resolution}')):
        os.mkdir(op.join(output_dir, f'resampled_mni_masked_res_{resolution}'))

    if not op.isdir(op.join(output_dir, f'resampled_mni_masked_normalized_res_{resolution}')):
        os.mkdir(op.join(output_dir, f'resampled_mni_masked_normalized_res_{resolution}'))
    
    target = datasets.load_mni152_template(resolution)

    for idx, fpath in enumerate(img_list):

        print('Image', fpath)

        if not os.path.exists(op.join(output_dir, f'resampled_mni_masked_normalized_res_{resolution}', op.basename(fpath))):

            img = nib.load(fpath)

            print('Original shape of image ', idx+1, ':',  img.shape)

            try:
                print('Computing mask...')
                mask_img = image.binarize_img(img)
                resampled_mask = image.resample_to_img(
                        mask_img,
                        target,
                        interpolation='nearest')

                print('Resampling image...')
                resampled_gm = image.resample_to_img(
                            img,
                            target,
                           interpolation='continuous')

                print('Masking image...')
                masked_resampled_gm_data = resampled_gm.get_fdata() * resampled_mask.get_fdata()

                masked_resampled_gm = nib.Nifti1Image(masked_resampled_gm_data, affine=resampled_gm.affine)
                nib.save(masked_resampled_gm, op.join(output_dir, f'resampled_mni_masked_res_{resolution}', op.basename(fpath)))

                print('New shape before normalization: ', masked_resampled_gm.shape)

                print('Min-Max normalizing...')
                masked_resampled_data = masked_resampled_gm.get_fdata().copy().astype(float)
                masked_resampled_data = np.nan_to_num(masked_resampled_data)
                masked_resampled_data *= 1.0/np.abs(masked_resampled_data).max()
                masked_resampled_norm_data = masked_resampled_data[0:48,0:56,0:48]
                masked_resampled_norm_img = nib.Nifti1Image(masked_resampled_norm_data, masked_resampled_gm.affine)
                print('New shape after normalization:', masked_resampled_norm_img.shape)
                nib.save(masked_resampled_norm_img, op.join(output_dir, f'resampled_mni_masked_normalized_res_{resolution}', op.basename(fpath))) # Save original image resampled, masked and normalized

                print(f"Image {idx} : DONE.")

            except Exception as e:
                
                print("Failed!")
                print(e)
                continue
        else:
            print('Image already preprocessed.')