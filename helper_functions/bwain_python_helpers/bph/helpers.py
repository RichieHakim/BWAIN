import numpy as np
import torch

from typing import Tuple

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

# @torch.jit.script
def phaseCorrelationImage_to_shift_helper(cc_im):
    cc_im = cc_im[None,:] if cc_im.ndim==2 else cc_im
    height, width = torch.as_tensor(cc_im.shape[-2:])
    vals_max, idx = torch.max(cc_im.reshape(cc_im.shape[0], cc_im.shape[1]*cc_im.shape[2]), dim=1)
    _, shift_y_raw, shift_x_raw = unravel_index(idx, cc_im.shape)
    shifts_y_x = torch.stack(((torch.floor(height/2) - shift_y_raw) , (torch.ceil(width/2) - shift_x_raw)), dim=1)
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
    cc_im = torch.as_tensor(cc_im)
    shifts_y_x, cc_max = phaseCorrelationImage_to_shift_helper(cc_im)
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
########################################## TORCH HELPERS ######################################################
###############################################################################################################

# def unravel_index(index, shape):
#     out = []
#     for dim in shape[::-1]:
#         out.append(index % dim)
#         index = index // dim
#     return tuple(out[::-1])

# ## version compatible with torch.jit.script
# @torch.jit.script
# def unravel_index(index: torch.Tensor, shape: torch.Tensor):
#     num_dims = shape.shape[0]
#     out = torch.zeros(num_dims, dtype=torch.int64)
#     for i, dim in enumerate(shape[::-1]):
#         out[i] = index % dim
#         index = index // dim
#     return tuple(out[::-1])

# def unravel_index(indices: np.ndarray, shape: tuple) -> tuple:
#     indices_arr = torch.from_numpy(indices)
#     shape = list(shape)
#     if np.any([s.ndim != 0 for s in shape]):
#         raise ValueError("unravel_index: shape should be a scalar or 1D sequence.")
#     out_indices = [0] * len(shape)
#     for i, s in reversed(list(enumerate(shape))):
#         indices_arr, out_indices[i] = torch.divmod(indices_arr, s)
#     oob_pos = indices_arr > 0
#     oob_neg = indices_arr < -1
#     out_list = []
#     for s, i in zip(shape, out_indices):
#         out_list.append(torch.where(oob_pos, s - 1, torch.where(oob_neg, 0, i)))
#     return tuple(out_list)

@torch.jit.script
def unravel_index(indices: torch.Tensor, shape: tuple) -> Tuple[torch.Tensor, ...]:
  indices_arr = indices
  shape = list(shape)
  out_indices = [0] * len(shape)
  for i, s in reversed(list(enumerate(shape))):
    indices_arr, out_indices[i] = torch.divmod(indices_arr, s)
  oob_pos = indices_arr > 0
  oob_neg = indices_arr < -1
  return tuple(torch.where(oob_pos, s - 1, torch.where(oob_neg, 0, i))
               for s, i in zip(shape, out_indices))