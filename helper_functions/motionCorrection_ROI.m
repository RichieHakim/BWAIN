function [xShift , yShift , cxx , cyy] = motionCorrection_ROI(movingIm , refIm, refIm_conjFFT_padded, maskPref, borderOuter, borderInner)
if ~exist('maskPref')
    maskPref = 0;
end

% tic
crop_factor = 3;

if ~isa(movingIm,'single')
    movingIm = single(movingIm);
end

if ~exist('refIm_conjFFT_padded')
    run_fft_ref = 1;
elseif isempty(refIm_conjFFT_padded)
    run_fft_ref = 1;
else
    run_fft_ref = 0;
end

if run_fft_ref
    if isa(refIm,'single')
        refIm = single(refIm);
    end
    
    
    refIm_conjFFT = conj(fft2(refIm));
    % refIm_conjFFT = gpuArray(conj(fft2(refIm)));
    
    refIm_conjFFT_shift = fftshift(gather(refIm_conjFFT));
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

movingIm_FFT_shift = fftshift(gather(movingIm_FFT));
% movingIm_FFT_shift = movingIm_FFT;

% movingIm_FFT_padded = zeros(size(movingIm_FFT_shift,1) , size(movingIm_FFT_shift,2));
% movingIm_FFT_padded(round(size(movingIm_FFT_shift,1)/2 - size(movingIm_FFT_shift,1)/crop_factor : ...
%     size(movingIm_FFT_shift,1)/2 + size(movingIm_FFT_shift,1)/crop_factor)...
%     , round(size(movingIm_FFT_shift,2)/2 - size(movingIm_FFT_shift,2)/crop_factor : ...
%     size(movingIm_FFT_shift,2)/2 + size(movingIm_FFT_shift,2)/crop_factor)) = 1;
% % figure; imagesc(abs(movingIm_FFT_padded))
% movingIm_FFT_padded = bsxfun(@times, movingIm_FFT_padded, movingIm_FFT_shift);

if maskPref
    movingIm_FFT_toUse = maskImage(movingIm_FFT_shift, borderOuter, borderInner);
%     movingIm_FFT_toUse = maskImage(movingIm_FFT_shift, borderInner, borderOuter);
else
    movingIm_FFT_toUse = movingIm_FFT_shift;
end

% size(refIm_conjFFT_padded)
% size(movingIm_FFT_toUse)
% size(refIm_conjFFT)
% size(movingIm_FFT)
% refIm_conjFFT_padded
% figure; imagesc(abs(movingIm_FFT_padded))
% figure; imagesc(abs(refIm_conjFFT_padded))
% spectralCorr = bsxfun(@times,refIm_conjFFT,movingIm_FFT);
% spectralCorr = bsxfun(@times, refIm_conjFFT_padded, movingIm_FFT_toUse);
spectralCorr = refIm_conjFFT_padded * movingIm_FFT_toUse;
% spectralCorr = refIm_conjFFT_padded * gpuArray(movingIm_FFT_toUse);
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