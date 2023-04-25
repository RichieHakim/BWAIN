function userFunction_Alignment_3D_1(source, event, varargin)
%% Variable stuff
global registrationImage refIm_crop_conjFFT_shift img_MC_moving ...
    img_MC_moving_rolling loadedCheck_registrationImage counter_frameNum ...
    indRange_y_crop indRange_x_crop

directory = 'D:\RH_local\data\BMI_cage_g8Test\mouse_g8t\20230314\analysis_data';

currentImage = source.hSI.hDisplay.lastFrame{1};

maskPref = 1;
borderOuter = 20;
borderInner = 10;

% loadedCheck_registrationImage = 0

if ~exist('loadedCheck_registrationImage') | isempty(loadedCheck_registrationImage) | loadedCheck_registrationImage ~= 1
% if 1
%     tmp = load([directory , '\stack.mat']);
% %     tmp = load([directory , '\stack_warped.mat']);
% %     file_fieldName = fieldnames(tmp);
% %     registrationImage = eval(['stack.' , file_fieldName{1}]);
%     registrationImage = eval(['tmp.stack.' , 'stack_avg']);

%     type_stack = 'stack';
%     type_stack = 'stack_warped';
    type_stack = 'stack_sparse';
    tmp = load([directory , '\', type_stack, '.mat']);
    disp(['LOADED zstack from', directory , '\', type_stack, '.mat'])
    registrationImage = eval(['tmp.', type_stack, '.stack_avg']);
    
    clear refIm_crop_conjFFT_shift_centerIdx
    for ii = 1:size(registrationImage,1)
        [refIm_crop_conjFFT_shift(ii,:,:), indRange_y_crop, indRange_x_crop] = make_fft_for_MC(squeeze(registrationImage(ii,:,:)));
    end
    loadedCheck_registrationImage = 1;
    
end
%% == USER SETTINGS ==
% ROI vars
frameRate = 30;
duration_plotting = 15 * frameRate; % ADJUSTABLE: change number value (in seconds)

numFramesToAvgForMotionCorr = 60;

n_slices = size(registrationImage, 1);

%% == Session Starting & counting ==

% == Start Session ==
if ~exist('counter_frameNum')
    counter_frameNum =0; 
end
if isempty(counter_frameNum)
    disp('hi')
    counter_frameNum = 1;
end

% Gyu 20221026
if isnan(counter_frameNum)
    disp('counter is NaN')
    counter_frameNum = 1;
end

if counter_frameNum == 1
    disp('frameNum = 1')
end

counter_frameNum = counter_frameNum + 1;
% disp(counter_frameNum)

%% == MOTION CORRECTION ==

if ~isa(img_MC_moving_rolling, 'double') || isempty(img_MC_moving_rolling)
    %     img_MC_moving_rolling = img_MC_moving;
    img_MC_moving_rolling = zeros([size(registrationImage,2), size(registrationImage,3) , numFramesToAvgForMotionCorr]);
end
if sum(size(img_MC_moving_rolling) == [size(registrationImage,2), size(registrationImage,3) , numFramesToAvgForMotionCorr]) ~= 3
    disp('changing size of img_MC_moving_rolling')
    img_MC_moving_rolling = zeros([size(registrationImage,2), size(registrationImage,3) , numFramesToAvgForMotionCorr]);
end

img_MC_moving = currentImage;

% img_MC_moving_rolling(:,:,end+1) = img_MC_moving;
if counter_frameNum >= 0
    img_MC_moving_rolling(:,:,mod(counter_frameNum , numFramesToAvgForMotionCorr)+1) = img_MC_moving;
end
img_MC_moving_rollingAvg = single(mean(img_MC_moving_rolling,3));
% size(img_MC_moving_rolling)

% figure; imagesc(img_MC_moving_rollingAvg)
% figure; imagesc(img_MC_moving_rollingAvg - double(currentImage))

% [xShift , yShift, cxx, cyy] = motionCorrection_singleFOV(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop , baselineStuff.MC.meanImForMC_crop_conjFFT_shift);
% [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg(1:100,1:100) , baselineStuff.MC.meanImForMC_crop(1:100,1:100) , baselineStuff.MC.meanImForMC_crop_conjFFT_shift(1:100,1:100));
% [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , registrationImage );
% [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop);

% threshold = prctile(img_MC_moving_rollingAvg, 95);
% % image_toUse = img_MC_moving_rollingAvg;
% image_toUse = img_MC_moving_rollingAvg .* (img_MC_moving_rollingAvg < threshold) + (img_MC_moving_rollingAvg > threshold).*threshold;
% figure; imagesc(image_toUse)

xShift = NaN(n_slices,1);

yShift = NaN(n_slices,1);
cxx = NaN(n_slices, indRange_x_crop(2) - indRange_x_crop(1) +1);
cyy = NaN(n_slices, indRange_y_crop(2) - indRange_y_crop(1) +1);
for z_frame = 1:n_slices
%     [yShift_tmp , xShift_tmp, cyy_tmp, cxx_tmp] = motionCorrection_ROI(img_MC_moving_rollingAvg , squeeze(registrationImage(z_frame,:,:)), [], maskPref, borderOuter, borderInner );
    [yShift_tmp , xShift_tmp, cyy_tmp, cxx_tmp] = motionCorrection_ROI(...
        img_MC_moving_rollingAvg(indRange_y_crop(1) : indRange_y_crop(2) , indRange_x_crop(1) : indRange_x_crop(2)),...
        [],...
        squeeze(refIm_crop_conjFFT_shift(z_frame,:,:)),...
        maskPref, borderOuter, borderInner...
        );
    
    xShift(z_frame,:) = xShift_tmp;
    yShift(z_frame,:) = yShift_tmp;
    cxx(z_frame,:) = cxx_tmp;
    cyy(z_frame,:) = cyy_tmp;
end

MC_corr = max(cxx, [], 2);

% xShift = 0;
% yShift = 0;
% MC_corr = 0;

% img_ROI_corrected{ii} = currentImage((baselineStuff.idxBounds_ROI{ii}(1,2):baselineStuff.idxBounds_ROI{ii}(2,2)) +round(yShift(ii)) ,...
%     (baselineStuff.idxBounds_ROI{ii}(1,1):baselineStuff.idxBounds_ROI{ii}(2,1)) +round(xShift(ii))); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]

% if abs(xShift) >10
%     xShift = 0;
% end
% if abs(yShift) >10
%     yShift = 0;
% end

%% Plotting

plotUpdatedOutput2([yShift(3)', xShift(3)'], duration_plotting, frameRate, 'Motion Correction Shifts', 10, 3)
% plotUpdatedOutput3([yShift'], duration_plotting, frameRate, 'Motion Correction Shifts', 10, 3)

if counter_frameNum > 15
    %     if counter_frameNum > 1
    plotUpdatedOutput4(MC_corr,...
        duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 3)
    legend()
end

end

%%
function [refIm_crop_conjFFT_shift, idxRange_y_crop, idxRange_x_crop] = make_fft_for_MC(refIm)

    refIm = single(refIm);
    % crop_factor = 5;
    crop_size = 256; % MAKE A POWER OF 2! eg 32,64,128,256,512


    length_x = size(refIm,2);
    length_y = size(refIm,1);
    middle_x = size(refIm,2)/2;
    middle_y = size(refIm,1)/2;

    % indRange_y_crop = [round(middle_y - length_y/crop_factor) , round(middle_y + length_y/crop_factor) ];
    % indRange_x_crop = [round(middle_x - length_y/crop_factor) , round(middle_x + length_y/crop_factor) ];

    idxRange_y_crop = [round(middle_y - (crop_size/2-1)) , round(middle_y + (crop_size/2)) ];
    idxRange_x_crop = [round(middle_x - (crop_size/2-1)) , round(middle_x + (crop_size/2)) ];
    
    refIm_crop = refIm(idxRange_y_crop(1) : idxRange_y_crop(2) , idxRange_x_crop(1) : idxRange_x_crop(2)) ;
   
    refIm_crop_conjFFT = conj(fft2(refIm_crop));
    refIm_crop_conjFFT_shift = fftshift(refIm_crop_conjFFT);

    % size(refIm_crop_conjFFT_shift,1);
    % if mod(size(refIm_crop_conjFFT_shift,1) , 2) == 0
    %     disp('RH WARNING: y length of refIm_crop_conjFFT_shift is even. Something is very wrong')
    % end
    % if mod(size(refIm_crop_conjFFT_shift,2) , 2) == 0
    %     disp('RH WARNING: x length of refIm_crop_conjFFT_shift is even. Something is very wrong')
    % end

%     refIm_crop_conjFFT_shift_centerIdx = ceil(size(refIm_crop_conjFFT_shift)/2);
    
%     size(refIm_crop_conjFFT_shift)

end