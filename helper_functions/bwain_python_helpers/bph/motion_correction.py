from typing import Tuple, Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt

from . import helpers

def find_translation_shifts(im1, im2, mask_fft=None, device='cpu', dtype=torch.float32):
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


class Shifter_rigid():
    """
    A class for performing rigid motion correction using
     phase correlation.
    Runs on PyTorch and allows for GPU acceleration.
    
    Workflow:
        1. Initialize the class
        2. Make mask using self.make_mask method:
            shifter.make_mask(
                bandpass_spatialFs_bounds=(1/10000, 1/2),
                order_butter=5,
                plot_pref=False
            )
        3. Preprocess the template images:
            shifter.preprocess_template_images(
                ims_template=np.stack([im1, im2], axis=0),
                template_precomputed=False,
            )
        4. Use the self.find_translation_shifts method 
         (or just call the class' __call__ method) to 
         calculate rigid offsets on new 'moving' images:
            shifts_y_x, cc_max = shifter.find_translation_shifts(im3_rep, idx_template=[0,1]*20)

    RH 2023
    """
    def __init__(
        self,
        device='cpu',
        dtype_image=torch.float32,
        dtype_fft=torch.complex64,
        verbose=True,
    ):
        # super().__init__()
        self._device = device
        self._dtype_image = torch.__dict__[dtype_image] if isinstance(dtype_image, str) else dtype_image
        self._dtype_fft   = torch.__dict__[dtype_fft]   if isinstance(dtype_fft, str)   else dtype_fft
        self._verbose = verbose if verbose==True or verbose=='True' else False

        self.mask = None
        self.mask_fftshift = None

    def make_mask(
        self,
        frame_shape_y_x=(512,512),
        bandpass_spatialFs_bounds=[1/128, 1/3],
        order_butter=5,
        mask=None,
        plot_pref=False,
    ):
        """
        Make a Fourier domain mask for the phase correlation.

        Args:
            frame_shape_y_x (Tuple[int]):
                Shape of the images that will be passed through
                 this class.
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
        bandpass_spatialFs_bounds = list(bandpass_spatialFs_bounds)
        bandpass_spatialFs_bounds[0] = max(bandpass_spatialFs_bounds[0], 1e-9)

        if (isinstance(mask, (np.ndarray, torch.Tensor))) or ((mask != 'None') and (mask is not None)):
            self.mask = torch.as_tensor(mask, device=self._device, dtype=self._dtype_fft)
            self.mask = mask / mask.sum()
            self.mask_fftshift = torch.fft.fftshift(self.mask)
            print(f'User provided mask of shape: {self.mask.shape} was normalized to sum=1, fftshift-ed, and converted to a torch.Tensor')
        else:
            wfilt_h = helpers.get_nd_butterworth_filter(
                shape=frame_shape_y_x, 
                factor=bandpass_spatialFs_bounds[0], 
                order=order_butter, 
                high_pass=True, 
                real=False,
            )
            wfilt_l = helpers.get_nd_butterworth_filter(
                shape=frame_shape_y_x, 
                factor=bandpass_spatialFs_bounds[1], 
                order=order_butter, 
                high_pass=False, 
                real=False,
            )

            kernel = torch.as_tensor(
                wfilt_h * wfilt_l,
                dtype=self._dtype_fft,
                device=self._device
            )

            self.mask = kernel / kernel.sum()
            # self.mask_fftshift = torch.fft.fftshift(self.mask)
            self.mask_fftshift = self.mask
            self.mask_fftshift = self.mask_fftshift.contiguous()

            if plot_pref and plot_pref!='False':
                plt.figure()
                plt.imshow(
                    torch.abs(kernel.cpu()).numpy(), 
                    # clim=[0,1],
                )
            if self._verbose:
                print(f'Created Fourier domain mask. self.mask_fftshift.shape: {self.mask_fftshift.shape}. Images input to find_translation_shifts will now be masked in the FFT domain.')

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
                 (i.e. mask * fftshift(fft2(ims_template))).
        """
        if not isinstance(ims_template, (torch.Tensor, np.ndarray)):
            ims_template = np.array(ims_template, copy=False)

        self.ims_template = torch.as_tensor(
            ims_template, 
            device=self._device, 
            dtype=self._dtype_image
        )
        if len(ims_template.shape) == 2:
            self.ims_template = self.ims_template[None, ...]

        if template_precomputed:
            self.ims_template_fft_mask = ims_template
            print(f'Using provided ims_template as self.ims_template_fft_mask. Calculated self.ims_template as fftshift(ifft2(ims_template))')
        else:
            self.ims_template_fft = torch.fft.fft2(self.ims_template, dim=(-2,-1)).type(self._dtype_fft)
            self.ims_template_fft_mask = self.ims_template_fft * self.mask_fftshift[None,:,:] if self.mask_fftshift is not None else self.ims_template_fft
            self.ims_template_fft_mask = self.ims_template_fft_mask.contiguous()
            if self._verbose:
                print(f'Created ims_template_fft_mask. self.ims_template_fft_mask.shape: {self.ims_template_fft_mask.shape}. Images input to find_translation_shifts will now be masked in the FFT domain.')

        self.ims_template_fft_mask_ifft = torch.fft.ifft2(self.ims_template_fft_mask)

    def find_translation_shifts(
        self,
        ims_moving,
        idx_template=[0],
    ):
        """
        Find the translation shifts between the template image and the moving images.

        Args:
            ims_moving (np.ndarray):
                The moving images.
                Shape: (n_ims, im_height, im_width)
            idx_template (int or List[int]):
                The index or indices of the template image to use.
                Should either be a single int or a list of int of
                 length ims_moving.shape[0] (n_ims).
        """
        idx_template = [idx_template] if isinstance(idx_template, int) else idx_template
        assert isinstance(idx_template, (list, np.ndarray, torch.Tensor))
        assert all([isinstance(idx, int) for idx in idx_template])
        
        if not isinstance(ims_moving, (torch.Tensor, np.ndarray)):
            ims_moving = np.array(ims_moving, copy=False)

        ims_moving = torch.as_tensor(
            ims_moving, 
            dtype=self._dtype_image
        ).to(self._device)

        if len(ims_moving.shape) == 2:
            ims_moving = ims_moving[None, ...]
        assert ims_moving.ndim == 3, "Moving images must be 3D. Shape: (n_ims, im_height, im_width)"
        
        cc = _phase_correlation_helper(
            im_moving=ims_moving,
            fft_template=self.ims_template_fft_mask[idx_template],
            mask_fftshift=self.mask_fftshift,
            use_mask=self.mask_fftshift is not None,
        )
        y_x, cc_max = helpers.phaseCorrelationImage_to_shift_helper(cc)

        return y_x.cpu(), cc_max.cpu(), cc.cpu()
        
    def __call__(
        self,
        ims_moving,
        idx_template=[0],
    ):
        """
        Calls find_translation_shifts
        """
        return self.find_translation_shifts(
            ims_moving=ims_moving, 
            idx_template=idx_template
        )

    def getattribute(
        self,
        attr
    ):
        return self.__getattribute__(attr)

@torch.jit.script
def _phase_correlation_helper(
    im_moving: torch.Tensor,
    fft_template: torch.Tensor,
    mask_fftshift: torch.Tensor=None,
    use_mask: bool=True,
):
    """
    Performs phase correlation over a batch of images.
    RH 2023
    Args:
        im_moving (torch.Tensor):
            Shape: (height, width) OR (n_im, height, width)
            Images to shift to be aligned with the template image(s).
        fft_template (torch.Tensor):
            Shape: (height, width) OR (n_im, height, width)
            Template images after fft and mask multiplication.
            These fft images will be directly multiplied against the
            conj(mask * fft2(im_moving))
        mask_fftshift (bool):
            FFT mask to multiply against the fft(im_moving) to perform
            frequency filtering.
            Note that any mask that has been pre-applied (or not) to
            fft_template should be used here as well.
        use_mask (bool):
            Whether or not to do frequency filtering at all.
    """
    im_moving    = im_moving[None,:,:]    if im_moving.ndim == 2    else im_moving
    fft_template = fft_template[None,:,:] if fft_template.ndim == 2 else fft_template

    assert (im_moving.shape[0] == fft_template.shape[0]) or (fft_template.shape[0] == 1) or (im_moving.shape[0] == 1), f"\
ERROR: Either of the following must be True: im_moving.shape[0] ({im_moving.shape[0]}) == fft_template.shape[0] ({fft_template.shape[0]}),  \
OR  fft_template.shape[0] == 1  OR  im_moving.shape[0] == 1"
    dims = (-2,-1)
    
    fft_moving = torch.fft.fft2(im_moving, dim=dims)
    if use_mask:
        fft_moving *= mask_fftshift
    R = fft_template * torch.conj(fft_moving)
    R /= torch.abs(R + 1e-12)
    cc = torch.abs(torch.fft.fftshift(torch.fft.ifft2(R, dim=dims), dim=dims))
    return cc
