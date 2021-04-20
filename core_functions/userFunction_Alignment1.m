function outputVoltageToTeensy = userFunction_Alignment1(source, event, varargin)
%% Variable stuff
global registrationImage img_MC_moving img_MC_moving_rolling loadedCheck_registrationImage counter_frameNum

directory = 'D:\RH_local\data\scanimage data\round 5 experiments\refIms';

currentImage = source.hSI.hDisplay.lastFrame{1};

% loadedCheck_registrationImage = 0

if ~exist('loadedCheck_registrationImage') | isempty(loadedCheck_registrationImage) | loadedCheck_registrationImage ~= 1
    tmp = load([directory , '\refIm_2_6__day0.mat']);
    file_fieldName = fieldnames(tmp);
    registrationImage = eval(['tmp.' , file_fieldName{1}]);
    loadedCheck_registrationImage=1;
end
%% == USER SETTINGS ==
% ROI vars
frameRate = 30;
duration_plotting = 15 * frameRate; % ADJUSTABLE: change number value (in seconds)

numFramesToAvgForMotionCorr = 5;

%% == Session Starting & counting ==

% == Start Session ==
if ~exist('counter_frameNum')
    counter_frameNum =0; 
end
if isempty(counter_frameNum)
    disp('hi')
    counter_frameNum = 1;
end
if counter_frameNum == 1
    disp('frameNum = 1')
end
% ======== COMMENT THIS IN/OUT TO START SESSION =======
% startSession
% =====================================================

counter_frameNum = counter_frameNum + 1;

%% == MOTION CORRECTION ==

if ~isa(img_MC_moving_rolling, 'uint16') || isempty(img_MC_moving_rolling)
    %     img_MC_moving_rolling = img_MC_moving;
    img_MC_moving_rolling = zeros([size(registrationImage) , numFramesToAvgForMotionCorr]);
end

img_MC_moving = currentImage;

% img_MC_moving_rolling(:,:,end+1) = img_MC_moving;
if counter_frameNum >= 0
    img_MC_moving_rolling(:,:,mod(counter_frameNum , numFramesToAvgForMotionCorr)+1) = img_MC_moving;
end
img_MC_moving_rollingAvg = single(mean(img_MC_moving_rolling,3));
% size(img_MC_moving_rolling)

% [xShift , yShift, cxx, cyy] = motionCorrection_singleFOV(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop , baselineStuff.MC.meanImForMC_crop_conjFFT_shift);
% [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg(1:100,1:100) , baselineStuff.MC.meanImForMC_crop(1:100,1:100) , baselineStuff.MC.meanImForMC_crop_conjFFT_shift(1:100,1:100));
[xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , registrationImage );
% [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop);
MC_corr = max(cxx);

% xShift = 0;
% yShift = 0;
% MC_corr = 0;

% img_ROI_corrected{ii} = currentImage((baselineStuff.idxBounds_ROI{ii}(1,2):baselineStuff.idxBounds_ROI{ii}(2,2)) +round(yShift(ii)) ,...
%     (baselineStuff.idxBounds_ROI{ii}(1,1):baselineStuff.idxBounds_ROI{ii}(2,1)) +round(xShift(ii))); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]

if abs(xShift) >80
    xShift = 0;
end
if abs(yShift) >80
    yShift = 0;
end

%% Plotting

plotUpdatedOutput3([xShift' yShift'], duration_plotting, frameRate, 'Motion Correction Shifts', 10, 1)

if counter_frameNum > 15
    %     if counter_frameNum > 1
    plotUpdatedOutput4(MC_corr,...
        duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 1)
end

end