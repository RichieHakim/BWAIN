%% Online Automatic Z-Correction
% Akshay Jaggi and Richard Hakim, 2022
%
% Code to automatically and actively align a stream of 2p microscopy images
% by comparing them to a reference z-stack. 
% This code will interface with scanimage to intake recorded images, 
% determine the z-correction necessary, and then send that command
% to the fast-z controller in scanimage. 
% Since this code will run online, it will be called on a certain interval.
% To keep the computer from slowing down, we'll run this only during
% ITI periods or during rewards. 

% Input: preprocessed image, reference image, reference fft, reference
% image diffs
% Does: 
% 1. Find the phase correlation of the image with each of the reference
%    images in the z-stack
% 2. Given the z-stack height between the reference images, return the
%    distance between the central reference image and the reference image 
%    that the current image is most similar to. 
% 3. Send distance in um to the z-stack controller
% Output: distance from most similar reference image to the central
%         reference image

function [delta, MC_corr, xShifts, yShifts] = zCorrection(image, reference, reference_fft, reference_diffs, maskPref, borderOuter, borderInner)
    n_slices = size(reference, 1);
    cxx = NaN(n_slices, size(image,1));
    cyy = NaN(n_slices, size(image,2));
    for z_frame = 1:n_slices
        if ~isempty(reference_fft)
            refIm_tmp = [];
        else
            refIm_tmp = squeeze(reference(z_frame,:,:));
        end
        
        [yShift_tmp , xShift_tmp, cyy_tmp, cxx_tmp] = motionCorrection_ROI(image, ...
            refIm_tmp, squeeze(reference_fft(z_frame,:,:)), ...
            maskPref, borderOuter, borderInner);
        
        cxx(z_frame,:) = cxx_tmp;
        cyy(z_frame,:) = cyy_tmp';
%         figure; imagesc(squeeze(abs(reference_fft(z_frame,:,:))))
    end
    [MC_corr, xShifts] = max(cxx, [], 2);
    [null, yShifts] = max(cyy, [], 2);
    [maxVal, maxArg] = max(MC_corr);
    delta = (ceil(n_slices/2)-maxArg) * reference_diffs; 
end