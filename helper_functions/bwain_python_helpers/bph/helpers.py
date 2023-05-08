import numpy as np
import torch

from typing import Tuple, List

###############################################################################################################
######################################### IMAGE PROCESSING ####################################################
###############################################################################################################

@torch.jit.script
def phase_correlation_helper(
    im_template,
    im_moving,
    mask_fft=None, 
    compute_maskFFT: bool=False, 
    template_precomputed: bool=False
):
    if im_template.ndim == 2:
        im_template = im_template[None, ...]
    if im_moving.ndim == 2:
        im_moving = im_moving[None, ...]
        return_2D = True
    else:
        return_2D = False
    if compute_maskFFT:
        mask_fft = mask_fft[None, ...]

    dims = (-2, -1)
        
    if compute_maskFFT:
        mask_fft = torch.fft.fftshift(mask_fft/mask_fft.sum(), dim=dims)
        fft_template = torch.conj(torch.fft.fft2(im_template, dim=dims) * mask_fft) if not template_precomputed else im_template
        fft_moving = torch.fft.fft2(im_moving, dim=dims) * mask_fft
    else:
        fft_template = torch.conj(torch.fft.fft2(im_template, dim=dims)) if not template_precomputed else im_template
        fft_moving = torch.fft.fft2(im_moving, dim=dims)

    R = fft_template * fft_moving

    if compute_maskFFT:
        R /= torch.abs(R + 1e-10)
    else:
        R /= torch.abs(R)
    
    cc = torch.fft.fftshift(torch.fft.ifft2(R, dim=dims), dim=dims).real
    
    return cc if not return_2D else cc[0]
def phase_correlation(
    im_template, 
    im_moving,
    mask_fft=None, 
    template_precomputed=False, 
    device='cpu'
):
    """
    Perform phase correlation on two images.
    Uses pytorch for speed
    RH 2022
    
    Args:
        im_template (np.ndarray or torch.Tensor):
            Template image(s).
            If ndim=2, a single image is assumed.
                shape: (height, width)
            if ndim=3, multiple images are assumed, dim=0 is the batch dim.
                shape: (batch, height, width)
                dim 0 should either be length 1 or the same as im_moving.
            If template_precomputed is True, this is assumed to be:
             np.conj(np.fft.fft2(im_template, axis=(1,2)) * mask_fft)
        im_moving (np.ndarray or torch.Tensor):
            Moving image(s).
            If ndim=2, a single image is assumed.
                shape: (height, width)
            if ndim=3, multiple images are assumed, dim=0 is the batch dim.
                shape: (batch, height, width)
                dim 0 should either be length 1 or the same as im_template.
        mask_fft (np.ndarray or torch.Tensor):
            Mask for the FFT.
            Shape: (height, width)
            If None, no mask is used.
        template_precomputed (bool):
            If True, im_template is assumed to be:
             np.conj(np.fft.fft2(im_template, axis=(1,2)) * mask_fft)
        device (str):
            Device to use.
    
    Returns:
        cc (np.ndarray):
            Phase correlation coefficient.
            Middle of image is zero-shift.
    """
    if isinstance(im_template, np.ndarray):
        im_template = torch.from_numpy(im_template).to(device)
        return_numpy = True
    else:
        return_numpy = False
    if isinstance(im_moving, np.ndarray):
        im_moving = torch.from_numpy(im_moving).to(device)
    if isinstance(mask_fft, np.ndarray):
        mask_fft = torch.from_numpy(mask_fft).to(device)

    cc = phase_correlation_helper(
        im_template=im_template,
        im_moving=im_moving,
        mask_fft=mask_fft if mask_fft is not None else torch.as_tensor([1], device=device),
        compute_maskFFT=(mask_fft is not None),
        template_precomputed=template_precomputed,
    )

    if return_numpy:
        cc = cc.cpu().numpy()
    return cc

@torch.jit.script 
def phaseCorrelationImage_to_shift_helper(cc_im):
    cc_im_shape = cc_im.shape
    cc_im = cc_im[None,:] if cc_im.ndim==2 else cc_im
    vals_max, idx = torch.max(cc_im.reshape(cc_im_shape[0], cc_im_shape[1]*cc_im_shape[2]), dim=1)
    shift_x_raw = idx % cc_im_shape[2]
    shift_y_raw = (idx // cc_im_shape[2]) % cc_im_shape[1]
    shifts_y_x = torch.stack(
        (
            (torch.floor(torch.as_tensor(cc_im_shape[1])/2) - shift_y_raw),
            (torch.ceil(torch.as_tensor(cc_im_shape[2])/2) - shift_x_raw)
        ),
        dim=1)
    return shifts_y_x, vals_max
def phaseCorrelationImage_to_shift(cc_im):
    """
    Convert phase correlation image to pixel shift values.
    RH 2022

    Args:
        cc_im (np.ndarray):
            Phase correlation image.
            Middle of image is zero-shift.

    Returns:
        shifts (np.ndarray):
            Pixel shift values (y, x).
    """
    if isinstance(cc_im, np.ndarray) == True:
        cc_im = torch.as_tensor(cc_im)
        return_numpy = True
    else:
        return_numpy = False
    shifts_y_x, cc_max = phaseCorrelationImage_to_shift_helper(cc_im)
    
    if return_numpy:
        return shifts_y_x.cpu().numpy(), cc_max.cpu().numpy()
    else:
        return shifts_y_x, cc_max

def mask_image_border(im, border_inner, border_outer, mask_value=0):
    """
    Mask an image with a border.
    RH 2022

    Args:
        im (np.ndarray):
            Input image.
        border_inner (int):
            Inner border width.
            Number of pixels in the center to mask.
            Value is the edge length of the center square.
        border_outer (int):
            Outer border width.
            Number of pixels in the border to mask.
        mask_value (float):
            Value to mask with.
    
    Returns:
        im_out (np.ndarray):
            Output image.
    """
    ## Find the center of the image
    height, width = im.shape
    center_y = cy = int(np.floor(height/2))
    center_x = cx = int(np.floor(width/2))

    ## make edge_lengths
    center_edge_length = cel = int(np.ceil(border_inner/2))
    outer_edge_length = oel = int(border_outer)

    ## Mask the center
    im[cy-cel:cy+cel, cx-cel:cx+cel] = mask_value
    ## Mask the border
    im[:oel, :] = mask_value
    im[-oel:, :] = mask_value
    im[:, :oel] = mask_value
    im[:, -oel:] = mask_value
    return im

###############################################################################################################
########################################## FEATURIZATION ######################################################
###############################################################################################################

def make_distance_image(im_height=512, im_width=512, idx_center_yx=(256, 256)):
    """
    creates a matrix of cartesian coordinate distances from the center
    RH 2021
    
    Args:
        im_height (int):
            height of image
        im_width (int):
            width of image
        idx_center (tuple):
            center index (y, x) - 0-indexed

    Returns:
        distance_image (np.ndarray): 
            array of distances to the center index

    """

    # create meshgrid
    x, y = np.meshgrid(range(im_width), range(im_height))  # note dim 1:X and dim 2:Y
    return np.sqrt((y - int(idx_center_yx[1])) ** 2 + (x - int(idx_center_yx[0])) ** 2)


###############################################################################################################
############################################# SPECTRAL ########################################################
###############################################################################################################

def butter_bandpass(lowcut, highcut, fs, order=5, plot_pref=True):
    '''
    designs a butterworth bandpass filter.
    Found on a stackoverflow, but can't find it
     anymore.
    RH 2021

        Args:
            lowcut (scalar): 
                frequency (in Hz) of low pass band
            highcut (scalar):  
                frequency (in Hz) of high pass band
            fs (scalar): 
                sample rate (frequency in Hz)
            order (int): 
                order of the butterworth filter
        
        Returns:
            b (ndarray): 
                Numerator polynomial coeffs of the IIR filter
            a (ndarray): 
                Denominator polynomials coeffs of the IIR filter
    '''
    import scipy.signal
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    
    if plot_pref:
        import matplotlib.pyplot as plt
        w, h = scipy.signal.freqz(b, a, worN=2000)
        plt.figure()
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('frequency response (a.u)')
    return b, a


###############################################################################################################
############################################### OTHER #########################################################
###############################################################################################################

def matlab_array_to_np_noCopy(arr):
    return np.array(arr, copy=False)

def matlab_array_to_torch_noCopy(arr, device='cpu', dtype=torch.float32):
    return torch.as_tensor(np.array(arr, copy=False), device=device, dtype=dtype)

def function_call(func_str, args=[], kwargs={}):
    f_list = func_str.split('.')
    
    f_cum_str = ''
    for ii, f in enumerate(f_list[:-1]):
        f_cum_str = f if ii==0 else f_cum_str+'.'+f
        exec(f'import {f_cum_str}')
    func = eval(func_str)

    return func(*args, **kwargs)

def dense_stack_to_sparse_stack(
    stack_in, 
    num_frames_per_slice=60, 
    num_slices=25, 
    num_volumes=10, 
    step_size_um=0.8, 
    frames_to_discard_per_slice=30, 
    sparse_step_size_um=4
):
    """
    Converts a dense stack of images into a sparse stack of images.
    RH 2023

    Args:
        stack_in (np.ndarray):
            Input stack of images.
        num_frames_per_slice (int):
            Number of frames per slice.
            From SI z-stack params.
        num_slices (int):
            Number of slices.
            From SI z-stack params.
        num_volumes (int):
            Number of volumes.
            From SI z-stack params.
        step_size_um (float):
            Step size in microns.
            From SI z-stack params.
        frames_to_discard_per_slice (int):
            Number of frames to discard per slice.
        sparse_step_size_um (float):
            Desired step size in microns for the sparse stack.
    """
    range_slices = num_slices * step_size_um
    range_idx_half = int((range_slices / 2) // sparse_step_size_um)
    step_numIdx = int(sparse_step_size_um // step_size_um)
    idx_center = int(num_slices // 2)
    idx_slices = [idx_center + n for n in np.arange(-range_idx_half*step_numIdx, range_idx_half*step_numIdx + 1, step_numIdx, dtype=np.int64)]
    assert (min(idx_slices) >= 0) and (max(idx_slices) <= num_slices), f"RH ERROR: The range of slice indices expected is greater than the number of slices available: {idx_slices}"
    positions_idx = [idx*step_size_um for idx in idx_slices]
    
    slices_rs = np.reshape(stack_in, (num_frames_per_slice, num_slices, num_volumes, stack_in.shape[1], stack_in.shape[2]), order='F');
    slices_rs = slices_rs[frames_to_discard_per_slice:,:,:,:,:];
    slices_rs = np.mean(slices_rs, axis=(0, 2))

    stack_out = slices_rs[idx_slices]
    return stack_out, positions_idx

###############################################################################################################
############################################ FROM SKIMAGE #####################################################
###############################################################################################################

import functools

import numpy as np
import scipy.fft

# from .._shared.utils import _supported_float_type


## Fixed the indexing because the skimage nerds drink too much coffee. RH.
def get_nd_butterworth_filter(shape, factor, order, high_pass, real,
                               dtype=np.float64, squared_butterworth=True):
    """Create a N-dimensional Butterworth mask for an FFT
    Parameters
    ----------
    shape : tuple of int
        Shape of the n-dimensional FFT and mask.
    factor : float
        Fraction of mask dimensions where the cutoff should be.
    order : float
        Controls the slope in the cutoff region.
    high_pass : bool
        Whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated).
    real : bool
        Whether the FFT is of a real (True) or complex (False) image
    squared_butterworth : bool, optional
        When True, the square of the Butterworth filter is used.
    Returns
    -------
    wfilt : ndarray
        The FFT mask.
    """
    ranges = []
    for i, d in enumerate(shape):
        # start and stop ensures center of mask aligns with center of FFT
        # axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        axis = np.arange(-(d - 1) / 2, (d - 1) / 2 + 0.5) / (d * factor)  ## FIXED, RH 2023
        ranges.append(scipy.fft.ifftshift(axis ** 2))
    # for real image FFT, halve the last axis
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    # q2 = squared Euclidean distance grid
    q2 = functools.reduce(
            np.add, np.meshgrid(*ranges, indexing="ij", sparse=True)
            )
    q2 = q2.astype(dtype)
    q2 = np.power(q2, order)
    wfilt = 1 / (1 + q2)
    if high_pass:
        wfilt *= q2
    if not squared_butterworth:
        np.sqrt(wfilt, out=wfilt)
    return wfilt

