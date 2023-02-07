import torch
import numpy as np
import matplotlib.pyplot as plt

from . import helpers

def find_translation_shifts(im1, im2, mask_fft=None, device='cpu', dtype=torch.float16):
    """
    Convenience function that combines `phase_correlation`
     and `phaseCorrelationImage_to_shift`
    
    Useful for matlab calls.
    """
    if mask_fft == 'None':
        mask_fft = None

    im1_t = torch.as_tensor(im1).type(dtype).to(device)
    im2_t = torch.as_tensor(im2).type(dtype).to(device)
    cc = helpers.phase_correlation(
        im1_t, 
        im2_t, 
        mask_fft=None,
        template_precomputed=False, 
        device=device
    )
    y_x, cc_max = helpers.phaseCorrelationImage_to_shift(cc)
    return y_x.cpu().numpy(), cc_max.cpu().numpy()

class Shifter_rigid:
    def __init__(
        self,
        frame_shape=(512,512),
        device='cpu',
        dtype=torch.float32,
    ):
        self.frame_shape = frame_shape
        self._device = device
        self._dtype = dtype

    def make_mask(
        self,
        bandpass_spatialFs_bounds=(1/128, 1/3),
        order_butter=5,
        mask=None,
        plot_pref=False,
    ):
        """
        Make a Fourier domain mask for the phase correlation.

        Args:
            bandpass_spatialFs_bounds (tuple): 
                (lowcut, highcut) in spatial frequency
                A butterworth filter is used to make the mask.
            order_butter (int):
                Order of the butterworth filter.
            mask (np.ndarray):
                If not None, use this mask instead of making one.
            plot_pref (bool):
                If True, plot the absolute value of the mask.
        """
        if mask is not None:
            self.mask = torch.as_tensor(mask, device=self._device, dtype=self._dtype)
        else:
            import scipy.signal
            fs = 1

            b, a = helpers.butter_bandpass(
                lowcut=bandpass_spatialFs_bounds[0],
                highcut=bandpass_spatialFs_bounds[1],
                fs=fs,
                order=order_butter,
                plot_pref=False
            )

            w, filt_freq_y = scipy.signal.freqz(b, a, worN=2000)
            filt_freq_x = (fs * 0.5 / np.pi) * w

            im_dist = helpers.make_distance_image(
                im_height=self.frame_shape[0],
                im_width=self.frame_shape[1],
                idx_center_yx=((self.frame_shape[0]-1)/2, (self.frame_shape[1]-1)/2),
            )

            interp = scipy.interpolate.interp1d(
                filt_freq_x, 
                filt_freq_y, 
                bounds_error=False, 
                kind='linear', 
                fill_value=0
            )
            kernel = interp(1/(im_dist+1e-12))
            kernel = torch.as_tensor(kernel, device=self._device, dtype=torch.complex64)

            if plot_pref:
                plt.figure()
                plt.imshow(torch.abs(kernel.cpu()).numpy())