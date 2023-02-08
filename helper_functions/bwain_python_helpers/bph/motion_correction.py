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
        dtype_image=torch.float32,
        dtype_fourier=torch.complex64,
    ):
        self.frame_shape = frame_shape
        self._device = device
        self._dtype_image = dtype_image
        self._dtype_fourier = dtype_fourier

        self.mask = None
        self.mask_fftshift = None

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
            self.mask = torch.as_tensor(mask, device=self._device, dtype=self._dtype_fourier)
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
            kernel = torch.as_tensor(kernel, device=self._device, dtype=self._dtype_fourier)

            if plot_pref:
                plt.figure()
                plt.imshow(torch.abs(kernel.cpu()).numpy())

            self.mask = kernel / kernel.sum()
            self.mask_fftshift = torch.fft.fftshift(self.mask)

    def preprocess_template_images(
        self,
        ims_template,
        template_precomputed=False,
    ):
        """
        Computes the FFT and masks the template images.

        Args:
            template_images (np.ndarray):
                The template images to be masked.
                Shape: (n_templates, im_height, im_width)
            template_precomputed (bool):
                If True, assume the template images are already preprocessed
                 (i.e. FFT'd and masked).
        """
        if template_precomputed:
            self.ims_template = ims_template
        else:
            if len(ims_template.shape) == 2:
                ims_template = ims_template[None, ...]
            self.ims_template = torch.as_tensor(
                ims_template, 
                device=self._device, 
                dtype=self._dtype_image
            )
            self.ims_template_fft_mask = torch.fft.fft2(self.ims_template).type(self._dtype_fourier)
            self.ims_template_fft_mask *= self.mask

    @torch.jit.export
    def phase_correlation_helper(
        self,
        im_moving,
        fft_template,
    ):
        dims = (-2, -1)
            
        if self.mask_fftshift is not None:
            fft_moving = torch.fft.fft2(im_moving, dim=dims).type(self._dtype_fourier) * self.mask_fftshift
        else:
            fft_moving = torch.fft.fft2(im_moving, dim=dims).type(self._dtype_fourier)
        R = fft_template * torch.conj(fft_moving)
        if self.mask_fftshift is not None:
            R /= torch.abs(R + 1e-10)
        else:
            R /= torch.abs(R)
        cc = torch.fft.fftshift(torch.fft.ifft2(R, dim=dims), dim=dims).real
        return cc

    def phaseCorrelationImage_to_shift_helper(cc_im):
        cc_im = cc_im[None,:] if cc_im.ndim==2 else cc_im
        height, width = torch.as_tensor(cc_im.shape[-2:])
        vals_max, idx = torch.max(cc_im.reshape(cc_im.shape[0], cc_im.shape[1]*cc_im.shape[2]), dim=1)
        _, shift_y_raw, shift_x_raw = unravel_index(idx, cc_im.shape)
        shifts_y_x = torch.stack(((torch.floor(height/2) - shift_y_raw) , (torch.ceil(width/2) - shift_x_raw)), dim=1)
        return shifts_y_x, vals_max


    def find_translation_shifts(
        self,
        ims_moving,
        idx_template=0,
    ):
        """
        Find the translation shifts between the template image and the moving images.

        Args:
            ims_moving (np.ndarray):
                The moving images.
                Shape: (n_moving, im_height, im_width)
            idx_template (int):
                The index of the template image to use.
        """
        if len(ims_moving.shape) == 2:
            ims_moving = ims_moving[None, ...]
        assert ims_moving.ndim == 3, "Moving images must be 3D. Shape: (n_ims, im_height, im_width)"
        
        ims_moving = torch.as_tensor(
            ims_moving, 
            device=self._device, 
            dtype=self._dtype_image
        )

        cc = self.phase_correlation_helper(
            im_moving=ims_moving,
            fft_template=self.ims_template[idx_template],
            template_precomputed=True, 
        )
        y_x, cc_max = helpers.phaseCorrelationImage_to_shift(cc)
        return y_x.cpu().numpy(), cc_max.cpu().numpy()
