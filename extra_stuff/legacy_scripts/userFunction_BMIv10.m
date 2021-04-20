function outputVoltageToTeensy = userFunction_BMIv10(source, event, varargin)
%% Variable stuff
tic
global counter_frameNum counter_CS_threshold counter_trialDuration counter_timeout counter_rewardToneHold...
    counter_rewardDelivery counter_ITI_successful...
    CE_buildingUpStats CE_experimentRunning CE_waitForBaseline CE_trial CE_timeout CE_rewardToneHold...
    CE_rewardDelivery CE_ITI_successful...
    ET_waitForBaseline ET_trialStart ET_timeout ET_rewardToneHold ET_rewardDelivery ET_ITI_successful...
    frequencyOverride...
    NumOfRewardsAcquired soundVolume...
    vals img_MC_moving img_MC_moving_rolling img_ROI_corrected...
    logger loggerNames logger_valsROIs...
    runningVals counter_runningVals...
    rolling_var_and_mean_obj vals_new vals_old F_mean F_var

persistent baselineStuff

currentImage = source.hSI.hDisplay.lastFrame{1};

% Should be TODAY's directory
directory = 'D:\RH_local\data\scanimage data\round 4 experiments\mouse 11.5\20210125';


if ~isstruct(baselineStuff)
    path_baselineStuff = [directory , '\baseline\baselineStuff.mat'];
    load(path_baselineStuff);
    disp(['LOADED baselineStuff from:  ' , path_baselineStuff])
end
%     load([directory , '\baselineStuff.mat']);

%% == USER SETTINGS ==
% ROI vars  
frameRate                   = 30;
duration_plotting           = 30 * frameRate; % ADJUSTABLE: change number value (in seconds). Duration of x axis in plots
duration_session            = frameRate * 60 * 60; % ADJUSTABLE: change number value (in seconds/minutes)
win_smooth                  = 10; % smoothing window (in frames)
% F_baseline_prctile          = 30; % percentile to define as F_baseline of cells
show_MC_ref_images          = 0;

numFramesToAvgForMotionCorr = 10;

threshold_value             = 0.08;

% numCells = max(baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,1));
numCells = baselineStuff.ROIs.num_cells;

prep_mode_on = 0;

if prep_mode_on
    numFramesToAvgForMotionCorr =  30;
    show_MC_ref_images = 1;
    threshold_value =   inf;
end

% range_cursorSound = [-threshold_value , threshold_value *1.5];
range_cursorSound = [-threshold_value threshold_value];
range_freqOutput = [2000 18000]; % this is set in the teensy code (Ofer made it)

reward_duration = 50; % in ms
reward_delay = 0; % in ms
% reward_delay = 5; % in ms
LED_duration = 0.2; % in s
LED_ramp_duration = 0.1; % in s

% directory = [baselineStuff.directory, '\expParams.mat'];
% directory = ['F:\RH_Local', '\expParams.mat'];
%% Trial Structure settings & Rolling Stats settings
duration_trial          = 30;
duration_timeout        = 5;
duration_threshold      = 0.066;
duration_rewardTone     = 1.5;
duration_ITI_success    = 1;
duration_rewardDelivery = 0.20;

duration_rollingStats       = round(frameRate * 60 * 10);
subSampleFactor_runningVals = 1;
numSamples_rollingStats = round(duration_rollingStats/subSampleFactor_runningVals);
duration_buildingUpStats    = round(frameRate * 60 * 1);

% baselinePeriodSTD = 1./scale_factors;
% threshold_quiescence = baselinePeriodSTD;
threshold_quiescence    = 0;

%% == Session Starting & counting ==

% == Start Session ==
if isempty(counter_frameNum)
    disp('hi. NEW SESSION STARTED')
    startSession
end

% ======== COMMENT THIS IN/OUT TO START SESSION =======
% startSession
% =====================================================

counter_frameNum = counter_frameNum + 1;

if counter_frameNum == 1
    disp('frameNum = 1')
end
% endSession

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

if ~isa(img_MC_moving_rolling, 'uint16')
    %     img_MC_moving_rolling = img_MC_moving;
    img_MC_moving_rolling = zeros([size(baselineStuff.MC.meanImForMC_crop) , numFramesToAvgForMotionCorr]);
end

img_MC_moving = currentImage(baselineStuff.MC.indRange_y_crop(1):baselineStuff.MC.indRange_y_crop(2)  ,...
    baselineStuff.MC.indRange_x_crop(1):baselineStuff.MC.indRange_x_crop(2)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]

% img_MC_moving_rolling(:,:,end+1) = img_MC_moving;
if counter_frameNum >= 0
    img_MC_moving_rolling(:,:,mod(counter_frameNum , numFramesToAvgForMotionCorr)+1) = img_MC_moving;
end
img_MC_moving_rollingAvg = mean(img_MC_moving_rolling,3);
% size(img_MC_moving_rolling)

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

if abs(xShift) >20
    xShift = 0;
end
if abs(yShift) >20
    yShift = 0;
end

%% == EXTRACT DECODER Product of Current Image and Decoder Template ==

% New extractor
% tic
x_idx     =  baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,2) +xShift;
y_idx     =  baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,3) +yShift;

idx_safe = single((x_idx < 1024) &...
    (x_idx > 0) & ...
    (y_idx < 512) & ...
    (y_idx > 0));
% idx_safe_nan = idx_safe;
% idx_safe_nan(idx_safe_nan==0) = NaN;

% x_idx(isnan(x_idx)) = 1;
% y_idx(isnan(y_idx)) = 1;

x_idx(~idx_safe) = 1;
y_idx(~idx_safe) = 1;

tall_currentImage = single(currentImage(sub2ind(size(currentImage), y_idx , x_idx)));
% toc
% TA_CF_lam = tall_currentImage .* baselineStuff.ROIs.spatial_footprints_tall_warped(:,4);
TA_CF_lam = tall_currentImage .* baselineStuff.ROIs.spatial_footprints_tall_warped(:,4) .* idx_safe;
TA_CF_lam_reshape = reshape(TA_CF_lam , baselineStuff.ROIs.cell_size_max , numCells);
vals_neurons = nansum( TA_CF_lam_reshape , 1 );
% toc
%% == ROLLING STATS ==
if counter_frameNum >=0
    logger_valsROIs(counter_frameNum,:) = vals_neurons;
end
% disp(num2str(size(baselineStuff.cellNumsToUse)))
if CE_experimentRunning
    %     vals_smooth = single(nanmean(logger_valsROIs( max(1 , counter_frameNum-win_smooth) : counter_frameNum,:),1));
    vals_smooth = vals_neurons;
    
    if mod(counter_frameNum-1 , subSampleFactor_runningVals) == 0
        next_idx = mod(counter_runningVals-1 , numSamples_rollingStats)+1;
        vals_old = runningVals(next_idx , :);
        runningVals(next_idx,:) = vals_smooth;
        vals_new = runningVals(next_idx , :);
        counter_runningVals = counter_runningVals+1;
        [rolling_var_and_mean_obj , F_mean , F_var] = rolling_var_and_mean_obj.step(counter_frameNum , vals_new , vals_old);
    end
    if counter_frameNum == 1
        F_mean = vals_smooth;
        F_var = ones(size(F_mean));
    end
    
    F_std = sqrt(F_var);
    %     F_std = nanstd(runningVals , [] , 1);
    %     F_std = ones(size(vals_smooth));
    F_std(F_std < 0.01) = inf;
    %     F_mean = nanmean(runningVals , 1);
    F_zscore = single((vals_smooth-F_mean)./F_std);
    F_zscore(isnan(F_zscore)) = 0;
    cursor_raw = F_zscore * baselineStuff.ROIs.cellWeightings';
    logger.decoder(counter_frameNum,2) = cursor_raw;
    
    if counter_frameNum >= win_smooth
        cursor = mean(logger.decoder(counter_frameNum-(win_smooth-1):counter_frameNum,2));
    else
        cursor = cursor_raw;
    end
    
    CS_quiescence = algorithm_quiescence(cursor, threshold_quiescence);
    CS_threshold = algorithm_thresholdState(cursor, threshold_value);
    
    %%  ===== TRIAL STRUCTURE =====
    % CE = current epoch
    % ET = epoch transition signal
    % CS = current state
    
    % START BUILDING UP STATS
    if counter_frameNum == 1
        CE_buildingUpStats = 1;
        %     soundVolume = 0;
        %     setSoundVolumeTeensy(soundVolume);
    end
    % BUILDING UP STATS
    if CE_buildingUpStats
    end
    % END BUILDING UP STATS
    if CE_buildingUpStats && counter_frameNum > duration_buildingUpStats
        CE_buildingUpStats = 0;
        ET_waitForBaseline = 1;
    end
    
    % START WAIT FOR BASELINE
    if ET_waitForBaseline
        ET_waitForBaseline = 0;
        CE_waitForBaseline = 1;
        soundVolume = 0;
        %     setSoundVolumeTeensy(soundVolume);
    end
    % WAIT FOR BASELINE
    if CE_waitForBaseline
        % nothing yet
    end
    % END WAIT FOR BASELINE (QUIESCENCE ACHIEVED)
    if CE_waitForBaseline && CS_quiescence
        CE_waitForBaseline = 0;
        ET_trialStart = 1;
    end
    
    % START TRIAL
    if ET_trialStart
        ET_trialStart = 0;
        CE_trial = 1;
        counter_trialDuration = 0;
        counter_CS_threshold = 0;
        soundVolume = 3.3;
        %     setSoundVolumeTeensy(soundVolume);
        source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(soundVolume);
        
        frequencyOverride = 0;
    end
    % COUNT TRIAL DURATION & COUNT THRESHOLD DURATIONS
    if CE_trial
        
        counter_trialDuration = counter_trialDuration + 1;
        
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
        %     ET_rewardToneHold = 1;
        ET_rewardDelivery = 1;
    end
    
    % START DELIVER REWARD
    if ET_rewardDelivery
        ET_rewardDelivery = 0;
        CE_rewardDelivery = 1;
        counter_rewardDelivery = 0;
        %     soundVolume = 0;
        %     setSoundVolumeTeensy(soundVolume);
        %     giveReward2(1, 1, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
        giveReward3(source, 1, 0, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
        %         giveReward3(source, 1, 0, 500, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
        NumOfRewardsAcquired = NumOfRewardsAcquired + 1;
        
        %         save([directory , '\logger.mat'], 'logger')
        %         saveParams(directory)
        %         disp(['Logger & Params Saved: frameCounter = ' num2str(counter_frameNum)]);
    end
    % COUNT DELIVER REWARD
    if CE_rewardDelivery
        counter_rewardDelivery = counter_rewardDelivery + 1;
    end
    % END DELIVER REWARD
    if CE_rewardDelivery && counter_rewardDelivery >= round(frameRate * duration_rewardDelivery)
        CE_rewardDelivery = 0;
        ET_ITI_successful = 1;
    end
    
    % START INTER-TRIAL-INTERVAL (POST-REWARD)
    if ET_ITI_successful
        ET_ITI_successful = 0;
        CE_ITI_successful = 1;
        counter_ITI_successful = 0;
        soundVolume = 0;
        source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(soundVolume);
    end
    % COUNT INTER-TRIAL-INTERVAL (POST-REWARD)
    if CE_ITI_successful
        counter_ITI_successful = counter_ITI_successful + 1;
    end
    % END INTER-TRIAL-INTERVAL (POST-REWARD)
    if CE_ITI_successful && counter_ITI_successful >= round(frameRate * duration_ITI_success)
        counter_ITI_successful = NaN;
        CE_ITI_successful = 0;
        ET_waitForBaseline = 1;
    end
    
end
%% Teensy Output calculations

if CE_experimentRunning
    if frequencyOverride
        freqToOutput = cursorToFrequency(threshold_value, range_cursorSound, range_freqOutput);
    else
        freqToOutput = cursorToFrequency(cursor, range_cursorSound, range_freqOutput);
    end
end

% freqToOutput = 6500* (mod(counter_frameNum,2)+0.5);

if exist('freqToOutput')
    outputVoltageToTeensy = teensyFrequencyToVoltageTransform(freqToOutput , range_freqOutput);
    source.hSI.task_FrequencyOutputVoltage.writeAnalogData(outputVoltageToTeensy);
end
% source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(soundVolume);

% save([directory , '\logger.mat'], 'logger')
% saveParams(directory)
% disp(['Logger & Params Saved: frameCounter = ' num2str(counter_frameNum)]);
%% Plotting

if counter_frameNum>1
    % %     plotUpdatedOutput([cursor , dFoF.*scale_factors], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 20)
    
    %     plotUpdatedOutput([cursor , dFoF], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 59)
    
    plotUpdatedOutput2([CE_waitForBaseline+0.1, CE_trial+0.2,...
        CE_rewardDelivery+0.3, CE_timeout+0.4, CE_buildingUpStats+0.5,]...
        , duration_plotting, frameRate, 'Rewards', 10, 52, ['Num of Rewards:  ' , num2str(NumOfRewardsAcquired)])
    
    plotUpdatedOutput3([xShift' yShift'], duration_plotting, frameRate, 'Motion Correction Shifts', 10, 51)
    
    if counter_frameNum > 15
        %     if counter_frameNum > 1
        plotUpdatedOutput4([nanmean(logger.motionCorrection(counter_frameNum-15:counter_frameNum,3),1)], duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 62)
        %         plotUpdatedOutput4([nanmean(logger.motionCorrection.MC_correlation(counter_frameNum-15:counter_frameNum,:),1)], duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 12)
    end
    
    %     plotUpdatedOutput4(dFoF(1:10) , duration_plotting , frameRate , 'dFoF' , 20, 50)
    plotUpdatedOutput6(cursor , duration_plotting , frameRate , 'cursor' , 10,53)
    %     plotUpdatedOutput7(logger.motionCorrection(counter_frameNum,3) , duration_plotting , frameRate , 'Motion Correction Correlation' , 10,54)
    
    
    if mod(counter_frameNum,300) == 0 && counter_frameNum > 300
        plotUpdatedOutput5([nanmean(logger.motionCorrection(counter_frameNum-300:counter_frameNum-1,3),1)],...
            duration_session, frameRate, 'Motion Correction Correlation All', 10, 1)
    end
    
    if show_MC_ref_images
        if mod(counter_frameNum,1) == 0
            %         if mod(counter_frameNum,11) == 0
            plotUpdatedMotionCorrectionImage_singleRegion(baselineStuff.MC.meanImForMC_crop , img_MC_moving_rollingAvg , img_MC_moving_rollingAvg, 'Motion Correction')
        end
    end
    
    
end
%% DATA LOGGING

if ~isnan(counter_frameNum)
    %     logger_valsROIs(counter_frameNum,:) = vals_neurons; %already done above
    
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
    logger.trial(counter_frameNum,26) = NumOfRewardsAcquired;
    
    logger.timers(counter_frameNum,1) = now;
    logger.timers(counter_frameNum,2) = toc;
    
    logger.decoder(counter_frameNum,1) = cursor;
    %     logger.decoder(counter_frameNum,2) = cursor_raw;
    logger.decoder(counter_frameNum,3) = freqToOutput;
    logger.decoder(counter_frameNum,4) = outputVoltageToTeensy;
    
    logger.motionCorrection(counter_frameNum,1) = xShift;
    logger.motionCorrection(counter_frameNum,2) = yShift;
    logger.motionCorrection(counter_frameNum,3) = MC_corr;
    
    %     toc
end

%% End Session
if  counter_frameNum == round(duration_session * 0.99)
    endSession
end

%% FUNCTIONS
    function startSession
        % INITIALIZE VARIABLES
        CE_waitForBaseline = 0;
        CS_quiescence = 0;
        ET_trialStart = 0;
        CE_trial = 0;
        soundVolume = 0;
        counter_trialDuration = 0;
        CS_threshold = 0;
        ET_rewardToneHold = 0; % reward signals
        CE_rewardToneHold = 0;
        counter_rewardToneHold = 0;
        frequencyOverride = 0;
        ET_rewardDelivery = 0;
        CE_rewardDelivery = 0;
        counter_rewardDelivery = 0;
        ET_ITI_successful = 0;
        CE_ITI_successful = 0;
        counter_ITI_successful = 0;
        ET_waitForBaseline = 0;
        CE_waitForBaseline = 0;
        ET_timeout = 0;
        CE_timeout = 0;
        counter_timeout = 0;
        
        counter_frameNum = 0;
        CE_buildingUpStats = 1;
        CE_experimentRunning = 1;
        cursor = 0;
        cursor_raw = 0;
        
        NumOfRewardsAcquired = 0;
        
        %         clear logger
        logger.trial = NaN(duration_session,25);
        logger.timers = NaN(duration_session,2);
        logger.decoder = NaN(duration_session,4);
        logger.motionCorrection = (NaN(duration_session,3));
        
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
        loggerNames.trial{26} = 'NumOfRewardsAcquired';
        
        loggerNames.timers{1} = 'time_now';
        loggerNames.timers{2} = 'tic_toc';
        
        loggerNames.decoder{1} = 'cursor';
        loggerNames.decoder{2} = 'freqToOutput';
        loggerNames.decoder{2} = 'outputVoltageToTeensy';
        
        loggerNames.motionCorrection{1} = 'xShift';
        loggerNames.motionCorrection{2} = 'yShift';
        loggerNames.motionCorrection{3} = 'MC_correlation';
        
        logger_valsROIs = nan(duration_session , numCells);
        runningVals = nan(numSamples_rollingStats , baselineStuff.ROIs.num_cells);
        counter_runningVals = 1;
        
        rolling_var_and_mean_obj = rolling_var_and_mean();
        rolling_var_and_mean_obj = rolling_var_and_mean_obj.set_key_properties(size(runningVals) , duration_rollingStats);
        
        saveParams(directory)
    end

    function endSession
        disp('SESSION OVER')
        counter_frameNum = NaN;
        CE_experimentRunning = 0;
        
        save([directory , '\logger.mat'], 'logger')
        save([directory , '\logger_valsROIs.mat'], 'logger_valsROIs')
        saveParams(directory)
        disp('=== Loggers and expParams saved ===')
        
        CE_waitForBaseline = 0;
        CS_quiescence = 0;
        ET_trialStart = 0;
        CE_trial = 0;
        soundVolume = 0;
        counter_trialDuration = 0;
        CS_threshold = 0;
        ET_rewardToneHold = 0; % reward signals
        CE_rewardToneHold = 0;
        counter_rewardToneHold = 0;
        frequencyOverride = 0;
        ET_rewardDelivery = 0;
        CE_rewardDelivery = 0;
        counter_rewardDelivery = 0;
        ET_ITI_successful = 0;
        CE_ITI_successful = 0;
        counter_ITI_successful = 0;
        ET_waitForBaseline = 0;
        CE_waitForBaseline = 0;
        ET_timeout = 0;
        CE_timeout = 0;
        counter_timeout = 0;
        
        %         counter_frameNum = 0;
        CE_buildingUpStats = 0;
        %         CE_experimentRunning = 0;
        cursor = 0;
        
        %         setSoundVolumeTeensy(0);
        source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(0);
        
    end

    function saveParams(directory)
        expParams.frameRate = frameRate;
        expParams.duration_session = duration_session;
        expParams.duration_trial = duration_trial;
        expParams.win_smooth = win_smooth;
        %         expParams.F_baseline_prctile = F_baseline_prctile;
        expParams.ROI_weightings = baselineStuff.ROIs.cellWeightings;
        expParams.duration_threshold = duration_threshold;
        expParams.threshold_value = threshold_value;
        expParams.range_cursorSound = range_cursorSound;
        expParams.range_freqOutput = range_freqOutput;
        expParams.duration_timeout = duration_timeout;
        expParams.numCells = numCells;
        expParams.directory = directory;
        expParams.duration_rollingStats = duration_rollingStats;
        expParams.duration_rollingStats = duration_buildingUpStats;
        expParams.subSampleFactor_runningVals = subSampleFactor_runningVals;
        expParams.threshold_quiescence = threshold_quiescence;
        expParams.duration_rewardTone = duration_rewardTone;
        expParams.duration_ITI_success = duration_ITI_success;
        expParams.duration_rewardDelivery = duration_rewardDelivery;
        expParams.reward_duration = reward_duration; % in ms
        expParams.reward_delay = reward_delay;
        expParams.LED_duration = LED_duration;
        expParams.LED_ramp_duration = LED_ramp_duration;
        expParams.numFramesToAvgForMotionCorr = numFramesToAvgForMotionCorr;
        
        expParams.loggerNames = loggerNames;
        
        expParams.baselineStuff = baselineStuff;
        
        save([directory , '\expParams.mat'], 'expParams')
        %         save([directory , '\motionCorrectionRefImages.mat'], 'motionCorrectionRefImages')
    end

end