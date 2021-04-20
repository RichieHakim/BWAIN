function outputVoltageToTeensy = userFunction_BMIv7_Pablo(source, event, varargin)
%tic
%% Variable stuff
global counter_frameNum counter_CS_threshold counter_trialDuration counter_timeout counter_rewardToneHold...
    counter_rewardDelivery counter_ITI_successful...
    CE_buildingUpStats CE_experimentRunning CE_waitForBaseline CE_trial CE_timeout CE_rewardToneHold...
    CE_rewardDelivery CE_ITI_successful...
    ET_waitForBaseline ET_trialStart ET_timeout ET_rewardToneHold ET_rewardDelivery ET_ITI_successful...
    frequencyOverride...
    NumOfRewardsAcquired soundVolume...
    vals img_MC_moving img_MC_moving_rolling img_ROI_corrected...
    logger loggerNames...
    cursor cursor_pos

persistent baselineStuff %movie_all
 
% persistent movie_all

currentImage = source.hSI.hDisplay.lastFrame{1};

directory   = 'D:\RH_local\data\scanimage data\round 4 experiments\mouse 8.6\20201203\';

% dir_movies  = directory;                                                             % simulation. load in one tif, and repeat that movie for the duration of frames. set current image as a frame
% if isempty(movie_all)                                                           %load in all tifs only at the first call, and retain movie all between calls (movie all defined as persistent). currentImage will be the image at frame counter
%     movie_all   = [];
%     movie_names = dir([directory,'*baseline_*']);
%     for i = 1:min(length(movie_names),1)
%         disp(['importing tif #',num2str(i)])
%         fileName    = movie_names(i).name;
%         movie_curr  = bigread5([directory, '\', fileName]);
%         movie_all   = cat(3,movie_all,movie_curr);
%     end
% end
% 
% movie_duration  = size(movie_all,3);
% if isempty(counter_frameNum)
%     currentImage    = movie_all(:,:,1);
% else
%     currentImage    = movie_all(:,:, mod(counter_frameNum , movie_duration-1)+1);
% end

if ~isstruct(baselineStuff) 
    load([directory , '\baselineStuff.mat']);
    figure; imagesc(baselineStuff.MC.meanIm)             %show the reference Image we are basing off of
end


%% == USER SETTINGS ==
% ROI vars
frameRate           = 30;
duration_session    = frameRate * 45 * 60; % ADJUSTABLE: change number value (in seconds/minutes)
win_smooth          = 5; % smoothing window (in frames)
F_baseline_prctile  = 30; % percentile to define as F_baseline of cells

numCells            = max(baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,1));


numFramesToAvgForMotionCorr = 1;

threshold_value     =   0.05;
threshold_quiescence=   0.01;

show_MC_ref_images  = 1;

prep_mode_on        = 0;

if prep_mode_on
    numFramesToAvgForMotionCorr = 20;
    show_MC_ref_images          = 0;
    threshold_value             = 1000;
end

reward_duration     = 100; % in ms
reward_delay    	= 200; % in ms
LED_duration        = 1; % in s
LED_ramp_duration   = 0.1; % in s

% % range_cursorSound = [-threshold_value , threshold_value *1.5];
% range_cursorSound = [-0.1 threshold_value];
% range_freqOutput = [2000 18000]; % this is set in the teensy code (Ofer made it)
% directory = [baselineStuff.directory, '\expParams.mat'];
% directory = ['F:\RH_Local', '\expParams.mat'];
%% Trial Structure settings & Rolling Stats settings
duration_trial          = 30;
duration_timeout        = 5;
duration_threshold      = 0.5;          % must hold target for 0.5 seconds             
duration_rewardTone     = 1.5;
duration_ITI_success    = 1;            % 1 second elapses between succeeding and trying again
duration_rewardDelivery = 1;

duration_rollingStats = round(frameRate * 30);

% baselinePeriodSTD = 1./scale_factors;
% threshold_quiescence = baselinePeriodSTD;
%threshold_quiescence = baselineStuff.base_data;

%% == Session Starting & counting ==
    thresh     = baselineStuff.thresh;
    gain       = baselineStuff.gain;
    alpha      = baselineStuff.alpha;
    target_pos = baselineStuff.target_pos;
    start_pos  = baselineStuff.start_pos;
    npos       = baselineStuff.npos;
% == Start Session ==
if isempty(counter_frameNum)
    disp('hi')
    startSession
end
if counter_frameNum == 1
    disp('frameNum = 1')
end
% ======== COMMENT THIS IN/OUT TO START SESSION =======
%  startSession
% =====================================================

counter_frameNum = counter_frameNum + 1;

endSession

% %% SAVING
% saveParams(directory)

% saveParams('F:\RH_Local\Rich data\scanimage data\20191110\mouse 10.13B\expParams.mat')

%% == MOTION CORRECTION ==
% %     img_MC_moving_rolling = [];

% if numel(baselineStuff.idxBounds_ROI) ~= numCells
%     error('NUMBER OF ROIs IS DIFFERENT THAN NUMBER OF DESIRED CELLS.  RH20191215')
% end
% 
% xShift = zeros(numCells,1); yShift = zeros(numCells,1);
% MC_corr = zeros(numCells,1);

% img_MC_moving = []; % COMMENT THESE TWO LINES IN TO UPDATE REFERENCE IMAGE
% img_MC_moving_rolling = [];

if isempty(img_MC_moving)
    img_MC_moving = baselineStuff.MC.meanImForMC_crop;
end

if ~isa(img_MC_moving_rolling, 'int32')
    img_MC_moving_rolling = img_MC_moving;
end

img_MC_moving = currentImage(baselineStuff.MC.indRange_y_crop(1):baselineStuff.MC.indRange_y_crop(2)  ,...
    baselineStuff.MC.indRange_x_crop(1):baselineStuff.MC.indRange_x_crop(2)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]

if size(img_MC_moving_rolling,3) > numFramesToAvgForMotionCorr-1
    img_MC_moving_rolling(:,:,1:end-(numFramesToAvgForMotionCorr-1)) = [];
end

img_MC_moving_rolling(:,:,end+1) = img_MC_moving;
img_MC_moving_rollingAvg = mean(img_MC_moving_rolling,3);
% size(img_MC_moving_rolling)

% figure; imagesc(img_MC_moving_rollingAvg)
% figure; imagesc(abs(baselineStuff.MC.meanImForMC_crop_conjFFT_shift))


% [xShift , yShift, cxx, cyy] = motionCorrection_singleFOV(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop , baselineStuff.MC.meanImForMC_crop_conjFFT_shift);
% [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg(1:100,1:100) , baselineStuff.MC.meanImForMC_crop(1:100,1:100) , baselineStuff.MC.meanImForMC_crop_conjFFT_shift(1:100,1:100));
[xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop , baselineStuff.MC.meanImForMC_crop_conjFFT_shift);
% [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop);
MC_corr = max(cxx);

% xShift = 0;
% yShift = 0;
% MC_corr = 0;

% img_ROI_corrected{ii} = currentImage((baselineStuff.idxBounds_ROI{ii}(1,2):baselineStuff.idxBounds_ROI{ii}(2,2)) +round(yShift(ii)) ,...
%     (baselineStuff.idxBounds_ROI{ii}(1,1):baselineStuff.idxBounds_ROI{ii}(2,1)) +round(xShift(ii))); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]

if abs(xShift) >40
    xShift = 0;
end
if abs(yShift) >40
    yShift = 0;
end

% xShift
% yShift

% xShift = zeros(numCells,1);
% yShift = zeros(numCells,1);

% xShift = ones(numCells,1) .* xShift(1);
% yShift = ones(numCells,1) .* yShift(1);


%% == Find Product of Current Image and Decoder Template ==

image_footprints_MC_weighted = zeros(size(currentImage,1) , size(currentImage,2));

% image_footprints_MC_weighted(sub2ind([size(currentImage,1),size(currentImage,2)] , baselineStuff.spatial_footprints_tall(:,3) + yShift , baselineStuff.spatial_footprints_tall(:,2) + xShift)) = ...
% image_footprints_MC_weighted(sub2ind([size(currentImage,1),size(currentImage,2)] , baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,3) +xShift , baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,2) +yShift)) = ...
%     bsxfun( @times , baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,4) , baselineStuff.ROIs.cellWeightings_tall_warped);

% idx_pablo = (baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,3) +xShift < 512) &...
%             (baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,3) +xShift > 0) & ...
%             (baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,2) +yShift < 1024) & ...
%             (baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,2) +yShift > 0);

x_idx     =  baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,2) +xShift;
y_idx     =  baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,3) +yShift;

idx_pablo = (x_idx < 1024) &...
            (x_idx > 0) & ...
            (y_idx < 512) & ...
            (y_idx > 0);

% val_idx   =  bsxfun( @times , baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,4) , baselineStuff.ROIs.cellWeightings_tall_warped)    ;    
xShift
yShift
image_footprints_MC_weighted(sub2ind([size(currentImage,1),size(currentImage,2)] , y_idx(idx_pablo) , x_idx(idx_pablo))) = ...
    baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(idx_pablo,4);


% image_footprints_MC_weighted(sub2ind([size(currentImage,1),size(currentImage,2)] , baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,3) +xShift , baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,2) +yShift)) = ...
%     bsxfun( @times , baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,4) , baselineStuff.ROIs.cellWeightings_tall_warped);

vals = sum(sum(bsxfun( @times , image_footprints_MC_weighted , single(currentImage) )));


%% == ROLLING STATS ==
if CE_experimentRunning
    if isempty(cursor)
       cursor = start_pos/npos;
       cursor_pos = start_pos;
    end
    %gain=.1;  % GAIN OVERRIDE. MAKE SURE TO ACTUALLY SET GAIN IN REAL EXPERIMENT
   
    runningVals     = [logger.decoder.rawVals(max(counter_frameNum - duration_rollingStats,1) : counter_frameNum);vals];            %PABLO ADD FOR RICH
    vals_smooth     = nanmean(runningVals(end-min(length(runningVals)-1,win_smooth):end),1);           % smooth the cursor on a rolling window
    base_data       = prctile(runningVals,F_baseline_prctile);                   % find baseline of running window
    cursor_vel      = gain*((vals_smooth-base_data)/base_data);                 % find the cursor velocity as the activity deviation from baseline, in units of baseline, times the gain
    relax           = alpha*(cursor_pos - start_pos)/npos;                      % find dimensionless relaxation term pulling cursor back to starting position
    cursor          = (cursor + cursor_vel) - relax;                            % update the cursor, in dimensionless space (units of baseline), with the velocity minus a relaxation term for how far from starting position. 
    cursor          = max(min(cursor,1),0);                                     % cursor should be unitless between 0 and 1. transformed to LEDs (*npos)
    cursor_pos      = max(min(cursor*npos,npos),1);                             % update cursor position by adding velocity. leds must be between 1 and npos
    
    CS_quiescence   = abs(cursor_vel) < threshold_quiescence;                   % logical. is the cursor at quienscence (close to baseline, aka velocity close to 0)
    CS_threshold    = abs(cursor_pos - target_pos) < thresh;                    % logical. is the cursor within 2 places of the target?

    cursor_voltage = double((cursor_pos/npos)*5);
    target_voltage = double((target_pos/npos)*5);
    
    %%  ===== TRIAL STRUCTURE =====
    % CE = current epoch
    % ET = epoch transition signal
    % CS = current state
    
    % START BUILDING UP STATS
    if counter_frameNum == 1
        CE_buildingUpStats  = 1;
        %     soundVolume = 0;
        %     setSoundVolumeTeensy(soundVolume);
    end
    % BUILDING UP STATS
    if CE_buildingUpStats
    end
    % END BUILDING UP STATS
    if CE_buildingUpStats && counter_frameNum > duration_rollingStats
        CE_buildingUpStats  = 0;
        ET_waitForBaseline  = 1;
    end
    
    % START WAIT FOR BASELINE
    if ET_waitForBaseline
        ET_waitForBaseline  = 0;
        CE_waitForBaseline  = 1;
        soundVolume         = 0;
        %     setSoundVolumeTeensy(soundVolume);
    end
    % WAIT FOR BASELINE
    if CE_waitForBaseline
        % nothing yet
    end
    % END WAIT FOR BASELINE (QUIESCENCE ACHIEVED)
    if CE_waitForBaseline && CS_quiescence
        CE_waitForBaseline  = 0;
        ET_trialStart       = 1;
    end
    
    % START TRIAL
    if ET_trialStart
        ET_trialStart           = 0;
        CE_trial                = 1;
        counter_trialDuration   = 0;
        counter_CS_threshold    = 0;
        frequencyOverride       = 0;
        source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(target_voltage);
  %     setSoundVolumeTeensy(soundVolume);
    end
    % COUNT TRIAL DURATION & COUNT THRESHOLD DURATIONS
    if CE_trial
        
        counter_trialDuration   = counter_trialDuration + 1;
        
        if CS_threshold
            counter_CS_threshold = counter_CS_threshold + 1;
        else
            counter_CS_threshold = 0;
        end
    end
    % END TRIAL: THRESHOLD REACHED
    if CE_trial && counter_CS_threshold >= round(frameRate * duration_threshold)
        counter_trialDuration = NaN;
        CE_trial = 0;
        ET_rewardDelivery = 1;    
  %     ET_rewardToneHold = 1;
    end
    
    % START DELIVER REWARD
    if ET_rewardDelivery
        ET_rewardDelivery       = 0;
        CE_rewardDelivery       = 1;
        counter_rewardDelivery  = 0;
        giveReward3(source, 1, 0, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
        NumOfRewardsAcquired    = NumOfRewardsAcquired + 1;
        
        %     giveReward3(source, 1, 0, 500, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
        %     soundVolume = 0;
        %     setSoundVolumeTeensy(soundVolume);
        %     giveReward2(1, 1, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once     
        %         save([directory , '\logger.mat'], 'logger')
        %         saveParams(directory)
        %         disp(['Logger & Params Saved: frameCounter = ' num2str(counter_frameNum)]);
    end
    % COUNT DELIVER REWARD
    if CE_rewardDelivery
        counter_rewardDelivery  = counter_rewardDelivery + 1;
        cursor                  = start_pos/npos;  %if receiving a reward, keep the cursor_pos at the start pos
        cursor_pos              = start_pos;
    end
    % END DELIVER REWARD
    if CE_rewardDelivery && counter_rewardDelivery >= round(frameRate * duration_rewardDelivery)
        CE_rewardDelivery       = 0;
        ET_ITI_successful       = 1;
        cursor                  = start_pos/npos;  %if receiving a reward, keep the cursor_pos at the start pos
        cursor_pos              = start_pos;
    end
    
    % START INTER-TRIAL-INTERVAL (POST-REWARD)
    if ET_ITI_successful
        ET_ITI_successful       = 0;
        CE_ITI_successful       = 1;
        counter_ITI_successful  = 0;
        cursor                  = start_pos/npos;  %if receiving a reward, keep the cursor_pos at the start pos
        cursor_pos              = start_pos;
  %     soundVolume =           0;
  %     setSoundVolumeTeensy(soundVolume);
    end
    % COUNT INTER-TRIAL-INTERVAL (POST-REWARD)
    if CE_ITI_successful
        counter_ITI_successful  = counter_ITI_successful + 1;
        cursor                  = start_pos/npos;  %if receiving a reward, keep the cursor_pos at the start pos
        cursor_pos              = start_pos;
    end
    % END INTER-TRIAL-INTERVAL (POST-REWARD)
    if CE_ITI_successful && counter_ITI_successful >= round(frameRate * duration_ITI_success)
        counter_ITI_successful  = NaN;
        CE_ITI_successful       = 0;
        ET_waitForBaseline      = 1;
        cursor                  = start_pos/npos;  %if receiving a reward, keep the cursor_pos at the start pos
        cursor_pos              = start_pos;
    end
    
end
%% Plotting

if CE_experimentRunning
     plotLEDs(cursor_vel,cursor_pos,target_pos,npos,thresh,500);
     %show_weights(currentImage(1:3:end,1:3:end),image_footprints_MC_weighted(1:3:end,1:3:end),counter_frameNum);

end

% if mod(counter_frameNum,30*60*5) == 0
if counter_frameNum == round(duration_session * 0.9)...
        || counter_frameNum == round(duration_session * 0.95)...
        || counter_frameNum == round(duration_session * 0.98)...
        || counter_frameNum == round(duration_session * 0.99)
    save([directory , '\logger.mat'], 'logger')
    disp(['Logger Saved: frameCounter = ' num2str(counter_frameNum)]);
end



 %% Teensy Output calculations
 % if mod(counter_frameNum , 2)==0
%     cursor_voltage=1;
%     disp('hey')
% else
%     cursor_voltage=0;
%     disp('ho')
% end

source.hSI.task_FrequencyOutputVoltage.writeAnalogData(cursor_voltage);
source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(target_voltage);



%% DATA LOGGING

if ~isnan(counter_frameNum)
    
    logger.trial(counter_frameNum,1) = counter_frameNum;
    logger.trial(counter_frameNum,2) = CS_quiescence;
    logger.trial(counter_frameNum,3) = ET_trialStart;
    logger.trial(counter_frameNum,4) = CE_trial;
    logger.trial(counter_frameNum,5) = soundVolume;
    logger.trial(counter_frameNum,6) = counter_trialDuration;
    logger.trial(counter_frameNum,7) = CS_threshold;
    logger.trial(counter_frameNum,8) = ET_rewardToneHold; % reward signals
    logger.trial(counter_frameNum,9) = CE_rewardToneHold;
    logger.trial(counter_frameNum,10) = counter_rewardToneHold;
    logger.trial(counter_frameNum,11) = frequencyOverride;
    logger.trial(counter_frameNum,12) = ET_rewardDelivery;
    logger.trial(counter_frameNum,13) = CE_rewardDelivery;
    logger.trial(counter_frameNum,14) = counter_rewardDelivery;
    logger.trial(counter_frameNum,15) = ET_ITI_successful;
    logger.trial(counter_frameNum,16) = CE_ITI_successful;
    logger.trial(counter_frameNum,17) = counter_ITI_successful;
    logger.trial(counter_frameNum,18) = ET_waitForBaseline;
    logger.trial(counter_frameNum,19) = CE_waitForBaseline;
    logger.trial(counter_frameNum,20) = ET_timeout;
    logger.trial(counter_frameNum,21) = CE_timeout;
    logger.trial(counter_frameNum,22) = counter_timeout;
    logger.trial(counter_frameNum,23) = CE_waitForBaseline;
    logger.trial(counter_frameNum,24) = CE_buildingUpStats;
    logger.trial(counter_frameNum,25) = CE_experimentRunning;
    logger.trial(counter_frameNum,26) = now;

    logger.decoder.outputs(counter_frameNum,1) = cursor;
    logger.decoder.outputs(counter_frameNum,2) = cursor_voltage; %freqToOutput;
    logger.decoder.outputs(counter_frameNum,3) = target_voltage; %outputVoltageToTeensy;
   
    logger.decoder.rawVals(counter_frameNum,:) = vals;
    logger.decoder.cursor(counter_frameNum,1) = cursor;
    logger.decoder.cursorpos(counter_frameNum,1) = cursor_pos;
    logger.decoder.targetpos(counter_frameNum,1) = target_pos;
    
    %logger.decoder.dFoF(counter_frameNum,:) = dFoF;
    
    logger.motionCorrection.xShift(counter_frameNum,:) = xShift;
    logger.motionCorrection.yShift(counter_frameNum,:) = yShift;
    logger.motionCorrection.MC_correlation(counter_frameNum,:) = MC_corr;
end

%% End Session
if counter_frameNum >= duration_session
    endSession
end

%% FUNCTIONS
    function startSession
        disp('CHANGE DIRECTORY')
        % INITIALIZE VARIABLES
        CE_waitForBaseline      = 0;
        CS_quiescence           = 0;
        ET_trialStart           = 0;
        CE_trial                = 0;
        soundVolume             = 0;
        counter_trialDuration   = 0;
        CS_threshold            = 0;
        ET_rewardToneHold       = 0; % reward signals
        CE_rewardToneHold       = 0;
        counter_rewardToneHold  = 0;
        frequencyOverride       = 0;
        ET_rewardDelivery       = 0;
        CE_rewardDelivery       = 0;
        counter_rewardDelivery  = 0;
        ET_ITI_successful       = 0;
        CE_ITI_successful       = 0;
        counter_ITI_successful  = 0;
        ET_waitForBaseline      = 0;
        CE_waitForBaseline      = 0;
        ET_timeout              = 0;
        CE_timeout              = 0;
        counter_timeout         = 0;
        
        counter_frameNum        = 0;
        CE_buildingUpStats      = 1;
        CE_experimentRunning    = 1;
        cursor                  = start_pos/npos;
        cursor_pos              = start_pos;
        %dFoF = 0;
        
        NumOfRewardsAcquired    = 0;
        
        %         clear logger
        logger.trial = NaN(duration_session,27);
        logger.decoder.outputs = NaN(duration_session,3);
        logger.decoder.rawVals = NaN(duration_session,1);
        logger.decoder.dFoF = NaN(duration_session,1);
        %logger.decoder.cursor(counter_frameNum,1) = NaN(duration_session,1);
        %logger.decoder.cursorpos(counter_frameNum,1) = NaN(duration_session,1);
        %logger.decoder.targetpos(counter_frameNum,1) = NaN(duration_session,1);
        logger.motionCorrection.xShift = (NaN(duration_session,1));
        logger.motionCorrection.yShift = (NaN(duration_session,1));
        logger.motionCorrection.MC_correlation = (NaN(duration_session,1));
        
        loggerNames.trial{1} = 'counter_frameNum';
        loggerNames.trial{2} = 'CS_quiescence';
        loggerNames.trial{3} = 'ET_trialStart';
        loggerNames.trial{4} = 'CE_trial';
        loggerNames.trial{5} = 'soundVolume';
        loggerNames.trial{6} = 'counter_trialDuration';
        loggerNames.trial{7} = 'CS_threshold';
        loggerNames.trial{8} = 'ET_rewardToneHold'; % reward signals
        loggerNames.trial{9} = 'CE_rewardToneHold';
        loggerNames.trial{10} = 'counter_rewardToneHold';
        loggerNames.trial{11} = 'frequencyOverride';
        loggerNames.trial{12} = 'ET_rewardDelivery';
        loggerNames.trial{13} = 'CE_rewardDelivery';
        loggerNames.trial{14} = 'counter_rewardDelivery';
        loggerNames.trial{15} = 'ET_ITI_successful';
        loggerNames.trial{16} = 'CE_ITI_successful';
        loggerNames.trial{17} = 'counter_ITI_successful';
        loggerNames.trial{18} = 'ET_waitForBaseline';
        loggerNames.trial{19} = 'CE_waitForBaseline';
        loggerNames.trial{20} = 'ET_timeout';
        loggerNames.trial{21} = 'CE_timeout';
        loggerNames.trial{22} = 'counter_timeout';
        loggerNames.trial{23} = 'CE_waitForBaseline';
        loggerNames.trial{24} = 'CE_buildingUpStats';
        loggerNames.trial{25} = 'CE_experimentRunning';
        loggerNames.trial{26} = 'time_now';
        
        loggerNames.decoder.outputs{1} = 'cursor';
        loggerNames.decoder.outputs{2} = 'freqToOutput';
        loggerNames.decoder.outputs{3} = 'outputVoltageToTeensy';
        
        
        saveParams(directory)
    end

    function endSession
        counter_frameNum = NaN;
        CE_experimentRunning = 0;
        
        save([directory , '\logger.mat'], 'logger')
        saveParams(directory)
        disp('SESSION OVER')
        
        CE_waitForBaseline      = 0;
        CS_quiescence           = 0;
        ET_trialStart           = 0;
        CE_trial                = 0;
        soundVolume             = 0;
        counter_trialDuration   = 0;
        CS_threshold            = 0;
        ET_rewardToneHold       = 0; % reward signals
        CE_rewardToneHold       = 0;
        counter_rewardToneHold  = 0;
        frequencyOverride       = 0;
        ET_rewardDelivery       = 0;
        CE_rewardDelivery       = 0;
        counter_rewardDelivery  = 0;
        ET_ITI_successful       = 0;
        CE_ITI_successful       = 0;
        counter_ITI_successful  = 0;
        ET_waitForBaseline      = 0;
        CE_waitForBaseline      = 0;
        ET_timeout              = 0;
        CE_timeout              = 0;
        counter_timeout         = 0;
        
        
        CE_buildingUpStats      = 0;
        
        cursor                  = 0;
        %         counter_frameNum = 0;
        %         CE_experimentRunning = 0;
        %dFoF = 0;
        
        %         setSoundVolumeTeensy(0);
        source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(0);
        
    end

    function saveParams(directory)
        expParams.frameRate             = frameRate;
        expParams.duration_session      = duration_session;
        expParams.duration_trial        = duration_trial;
        expParams.win_smooth            = win_smooth;
        expParams.F_baseline_prctile    = F_baseline_prctile;
        expParams.duration_threshold    = duration_threshold;
        expParams.threshold_value       = threshold_value;
        expParams.duration_timeout      = duration_timeout;
        expParams.numCells              = numCells;
        expParams.directory             = directory;
        expParams.duration_rollingStats = duration_rollingStats;
        expParams.threshold_quiescence  = threshold_quiescence;
        expParams.duration_rewardTone   = duration_rewardTone;
        expParams.duration_ITI_success  = duration_ITI_success;
        expParams.duration_rewardDelivery = duration_rewardDelivery;
        expParams.reward_duration       = reward_duration; % in ms
        expParams.reward_delay          = reward_delay;
        expParams.LED_duration          = LED_duration;
        expParams.LED_ramp_duration     = LED_ramp_duration;
        expParams.numFramesToAvgForMotionCorr = numFramesToAvgForMotionCorr;
        
        expParams.loggerNames = loggerNames;
        
        expParams.baselineStuff = baselineStuff;
        
        save([directory , '\expParams.mat'], 'expParams')
        %         save([directory , '\motionCorrectionRefImages.mat'], 'motionCorrectionRefImages')
    end
%toc
end