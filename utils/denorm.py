import nibabel as nib
import numpy as np

def mask_using_original(inim, outim):
    '''
    Compute the mask of the original map and apply it to the reconstructed one. 
    '''
    # Set masking using NaN's
    data_orig = inim.get_fdata()
    data_repro = outim.get_fdata()
    
    if np.any(np.isnan(data_orig)):
        data_nan_orig = data_orig
        data_nan_repro = data_repro
        
        data_nan_repro[np.isnan(data_orig)] = 0
    else:
        data_nan_orig = data_orig
        data_nan_repro = data_repro

        data_nan_repro[data_orig == 0] = 0
        data_nan_orig[data_orig == 0] = 0
        
    # Save as image
    data_img_nan_orig = nib.Nifti1Image(data_nan_orig, inim.affine)
    data_img_nan_repro = nib.Nifti1Image(data_nan_repro, outim.affine)

    return data_img_nan_orig, data_img_nan_repro

def find_norm_factor(src):

    src_data = np.nan_to_num(src).get_fdata().copy()
    norm_factor = 1.0/np.abs(src_data).max()

    return norm_factor

def un_normalize(generated, src, scaling=100):
    src, generated = mask_using_original(src, generated)
    norm_factor = find_norm_factor(src)

    affine = src.affine
    header = src.header

    generated_un_norm = generated.get_fdata() / norm_factor
    generated_un_norm = generated_un_norm * scaling

    generated_un_norm = nib.Nifti1Image(generated_un_norm, affine=affine, header=header)

    return generated_un_norm