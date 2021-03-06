function [xShift , yShift , cxx , cyy] = motionCorrection_ROI(movingIm , refIm, refIm_conjFFT_padded)
% tic
crop_factor = 3;

if ~isa(movingIm,'single')
    movingIm = single(movingIm);
end

if ~exist('refIm_conjFFT_padded')
    if isa(refIm,'single')
        refIm = single(refIm);
    end
    
    
    refIm_conjFFT = conj(fft2(refIm));
    % refIm_conjFFT = gpuArray(conj(fft2(refIm)));
    
    refIm_conjFFT_shift = fftshift(refIm_conjFFT);
    refIm_conjFFT_padded = zeros(size(refIm_conjFFT_shift,1) , size(refIm_conjFFT_shift,2));
        
    refIm_conjFFT_padded(round(size(refIm_conjFFT_shift,1)/2 - size(refIm_conjFFT_shift,1)/crop_factor : ...
        size(refIm_conjFFT_shift,1)/2 + size(refIm_conjFFT_shift,1)/crop_factor)...
        , round(size(refIm_conjFFT_shift,2)/2 - size(refIm_conjFFT_shift,2)/crop_factor : ...
        size(refIm_conjFFT_shift,2)/2 + size(refIm_conjFFT_shift,2)/crop_factor)) = 1;
    
    refIm_conjFFT_padded = bsxfun(@times, logical(refIm_conjFFT_padded), refIm_conjFFT_shift);
end

movingIm_FFT = fft2(movingIm);
% movingIm_FFT = gpuArray(fft2(movingIm));
% refIm_conjFFT_padded = gpuArray(refIm_conjFFT_padded);

movingIm_FFT_shift = fftshift(movingIm_FFT);
movingIm_FFT_padded = zeros(size(movingIm_FFT_shift,1) , size(movingIm_FFT_shift,2));
movingIm_FFT_padded(round(size(movingIm_FFT_shift,1)/2 - size(movingIm_FFT_shift,1)/crop_factor : ...
    size(movingIm_FFT_shift,1)/2 + size(movingIm_FFT_shift,1)/crop_factor)...
    , round(size(movingIm_FFT_shift,2)/2 - size(movingIm_FFT_shift,2)/crop_factor : ...
    size(movingIm_FFT_shift,2)/2 + size(movingIm_FFT_shift,2)/crop_factor)) = 1;
% figure; imagesc(abs(movingIm_FFT_padded))
movingIm_FFT_padded = bsxfun(@times, movingIm_FFT_padded, movingIm_FFT_shift);
% figure; imagesc(log(abs(movingIm_FFT_padded)))

% size(refIm_conjFFT)
% size(movingIm_FFT)
% refIm_conjFFT_padded
% figure; imagesc(abs(movingIm_FFT_padded))
% figure; imagesc(abs(refIm_conjFFT_padded))
% spectralCorr = bsxfun(@times,refIm_conjFFT,movingIm_FFT);
spectralCorr = bsxfun(@times,refIm_conjFFT_padded, movingIm_FFT_padded);
phaseCorr = spectralCorr./(abs(spectralCorr)+0.01); % phase correlation, add eps to avoid division by zero
Corr = ifft2(phaseCorr,'symmetric'); % cc is a 3D array. ifft2 takes the 2-D fourier transform for each slice
% Corr = ifft2(phaseCorr,'nonsymmetric'); % cc is a 3D array. ifft2 takes the 2-D fourier transform for each slice

% figure; imagesc(abs(phaseCorr))
cxx = max(Corr,[],2);
cyy = max(Corr,[],1);

[~,xIdx] = max(cxx);
% xShift = xIdx - size(refIm,2)/2;
yShift = xIdx;
if yShift > size(movingIm,1)/2
    yShift = yShift-size(movingIm,1);
end
% xShift = gather(xIdx - size(refIm,2)/2);
[~,yIdx] = max(cyy);
% yShift = yIdx - size(refIm,1)/2;
xShift = yIdx;
if xShift > size(movingIm,1)/2
    xShift = xShift-size(movingIm,2);
end
% yShift = gather(yIdx - size(refIm,1)/2);

% figure; plot(cxx)
% cxx
% toc

% xShift = gather(xShift);
% yShift = gather(yShift);
% cxx = gather(cxx);
% cyy = gather(cyy);

xShift = xShift -1; % this is because a peak at index 1 means zero shift
yShift = yShift -1;

end