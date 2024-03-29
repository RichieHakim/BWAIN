function [logger_output , logger_valsROIs_output  , NumOfRewardsAcquired] =...
    BMIv11_simulation(vals_neurons, counter_frameNum , baselineStuff , trialStuff,  last_frameNum , threshold_value , threshold_quiescence, duration_quiescenceHold, num_frames_total)
%% Variable stuff
tic
global counter_trialIdx counter_CS_threshold counter_timeout counter_rewardToneHold...
    counter_rewardDelivery counter_ITI_withZ...
    CE_buildingUpStats CE_experimentRunning CE_waitForBaseline CE_trial CE_timeout CE_rewardToneHold...
    CE_rewardDelivery CE_ITI_withZ...
    ET_waitForBaseline ET_trialStart ET_timeout ET_rewardToneHold ET_rewardDelivery ET_ITI_withZ...
    frequencyOverride...
    NumOfRewardsAcquired NumOfTimeouts trialNum soundVolume...
    img_MC_moving img_MC_moving_rolling...
    logger loggerNames logger_valsROIs...
    runningVals running_cursor_raw counter_runningVals counter_runningCursor...
    rolling_var_obj_cells rolling_var_obj_cursor loadedCheck_registrationImage...
    registrationImage referenceDiffs refIm_crop_conjFFT_shift refIm_crop indRange_y_Crop indRange_x_Crop...
    img_MC_moving_rolling_z refIm_crop_conjFFT_shift_masked counter_last_z_correction...
    counter_quiescenceHold...

% persistent baselineStuff

% currentImage = source.hSI.hDisplay.lastFrame{1};
% hash_image = simple_image_hash(currentImage);

% Should be TODAY's directory
% directory = 'D:\RH_local\data\scanimage data\round 4 experiments\mouse 8.6\20201223_rich\sesh_2';


% if ~isstruct(baselineStuff)
%     load([directory , '\baseline\baselineStuff.mat']);
% end
% if ~isstruct(trialStuff)
%     load([directory , '\trialStuff.mat']);
% end

%% == USER SETTINGS ==
% ROI vars
frameRate                   = 30;
duration_plotting           = 30 * frameRate; % ADJUSTABLE: change number value (in seconds). Duration of x axis in plots
duration_session            = num_frames_total; % ADJUSTABLE: change number value (in seconds/minutes)
win_smooth                  = 4; % smoothing window (in frames)
% F_baseline_prctile          = 30; % percentile to define as F_baseline of cells
% show_MC_ref_images          = 0;


% numFramesToAvgForMotionCorr = 10;

% threshold_value             = 30000;

% numCells = max(baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,1));
numCells = baselineStuff.ROIs.num_cells;

% range_cursor = [-threshold_value , threshold_value *1.5];
range_cursor = [-threshold_value threshold_value];
range_freqOutput = [1000 18000]; % this is set in the teensy code (only here for logging purposes)
voltage_at_threshold = 3.1; % this will be the maximum output voltage ([0:voltage_at_threshold])
interval_z_correction       = 20*frameRate;

% reward_duration = 50; % in ms
reward_delay = 0; % in ms
% reward_delay = 5; % in ms
LED_duration = 0.2; % in s
LED_ramp_duration = 0.1; % in s

% directory = [baselineStuff.directory, '\expParams.mat'];
% directory = ['F:\RH_Local', '\expParams.mat'];
%% Trial Structure settings & Rolling Stats settings
% below in unit seconds
duration_trial          = 20;
duration_timeout        = 4;
duration_threshold      = 0.066;
% duration_rewardTone     = 1.5; % currently unused
duration_ITI    = 3;
duration_rewardDelivery = 1.00;

duration_rollingStats       = round(frameRate * 60 * 15);
subSampleFactor_runningVals = 1;
numSamples_rollingStats = round(duration_rollingStats/subSampleFactor_runningVals);
duration_buildingUpStats    = round(frameRate * 60 * 1);

% threshold_quiescence    = 0;
% duration_quiescenceHold = 0.5; % in seconds

%% == Session Starting & counting ==

% == Start Session ==
if counter_frameNum == 1
    disp('hi. NEW SESSION STARTED')
    startSession
    disp(size(logger.timeSeries))
end
% ======== COMMENT THIS IN/OUT TO START SESSION =======
% startSession
% =====================================================

% % counter_frameNum = counter_frameNum + 1;
% counter_frameNum = source.hSI.hStackManager.framesDone;

% endSession

% %% SAVING
% saveParams(directory)

% saveParams('F:\RH_Local\Rich data\scanimage data\20191110\mouse 10.13B\expParams.mat')

%% == MOTION CORRECTION ==
% % %     img_MC_moving_rolling = [];
% 
% % if numel(baselineStuff.idxBounds_ROI) ~= numCells
% %     error('NUMBER OF ROIs IS DIFFERENT THAN NUMBER OF DESIRED CELLS.  RH20191215')
% % end
% %
% % xShift = zeros(numCells,1); yShift = zeros(numCells,1);
% % MC_corr = zeros(numCells,1);
% 
% % img_MC_moving = []; % COMMENT THESE TWO LINES IN TO UPDATE REFERENCE IMAGE
% % img_MC_moving_rolling = [];
% 
% if isempty(img_MC_moving)
%     img_MC_moving = baselineStuff.MC.meanImForMC_crop;
% end
% 
% if ~isa(img_MC_moving_rolling, 'uint16')
%     %     img_MC_moving_rolling = img_MC_moving;
%     img_MC_moving_rolling = zeros([size(baselineStuff.MC.meanImForMC_crop) , numFramesToAvgForMotionCorr]);
% end
% 
% img_MC_moving = currentImage(baselineStuff.MC.indRange_y_crop(1):baselineStuff.MC.indRange_y_crop(2)  ,...
%     baselineStuff.MC.indRange_x_crop(1):baselineStuff.MC.indRange_x_crop(2)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
% 
% % img_MC_moving_rolling(:,:,end+1) = img_MC_moving;
% if counter_frameNum >= 0
%     img_MC_moving_rolling(:,:,mod(counter_frameNum , numFramesToAvgForMotionCorr)+1) = img_MC_moving;
% end
% img_MC_moving_rollingAvg = single(mean(img_MC_moving_rolling,3));
% % size(img_MC_moving_rolling)
% 
% % [xShift , yShift, cxx, cyy] = motionCorrection_singleFOV(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop , baselineStuff.MC.meanImForMC_crop_conjFFT_shift);
% % [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg(1:100,1:100) , baselineStuff.MC.meanImForMC_crop(1:100,1:100) , baselineStuff.MC.meanImForMC_crop_conjFFT_shift(1:100,1:100));
% [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop , baselineStuff.MC.meanImForMC_crop_conjFFT_shift);
% % [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop);
% MC_corr = max(cxx);
% 
% % xShift = 0;
% % yShift = 0;
% % MC_corr = 0;
% 
% % img_ROI_corrected{ii} = currentImage((baselineStuff.idxBounds_ROI{ii}(1,2):baselineStuff.idxBounds_ROI{ii}(2,2)) +round(yShift(ii)) ,...
% %     (baselineStuff.idxBounds_ROI{ii}(1,1):baselineStuff.idxBounds_ROI{ii}(2,1)) +round(xShift(ii))); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
% 
% if abs(xShift) >80
%     xShift = 0;
% end
% if abs(yShift) >80
%     yShift = 0;
% end
% 
% % xShift = zeros(numCells,1);
% % yShift = zeros(numCells,1);
% 
% % xShift = ones(numCells,1) .* xShift(1);
% % yShift = ones(numCells,1) .* yShift(1);
% 

%% == EXTRACT DECODER Product of Current Image and Decoder Template ==

% % New extractor
% % tic
% x_idx     =  baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,2) +xShift;
% y_idx     =  baselineStuff.ROIs.spatial_footprints_tall_warped_weighted(:,3) +yShift;
% 
% idx_safe = single((x_idx < 1024) &...
%     (x_idx > 0) & ...
%     (y_idx < 512) & ...
%     (y_idx > 0));
% % idx_safe_nan = idx_safe;
% % idx_safe_nan(idx_safe_nan==0) = NaN;
% 
% % x_idx(isnan(x_idx)) = 1;
% % y_idx(isnan(y_idx)) = 1;
% 
% x_idx(~idx_safe) = 1;
% y_idx(~idx_safe) = 1;
% 
% tall_currentImage = single(currentImage(sub2ind(size(currentImage), y_idx , x_idx)));
% % toc
% % TA_CF_lam = tall_currentImage .* baselineStuff.ROIs.spatial_footprints_tall_warped(:,4);
% TA_CF_lam = tall_currentImage .* baselineStuff.ROIs.spatial_footprints_tall_warped(:,4) .* idx_safe;
% TA_CF_lam_reshape = reshape(TA_CF_lam , baselineStuff.ROIs.cell_size_max , numCells);
% vals_neurons = nansum( TA_CF_lam_reshape , 1 );
% % toc
%% == ROLLING STATS ==
cursor_brain_raw = NaN;
fakeFeedback_inUse = NaN;
if counter_frameNum >=0
    logger_valsROIs(counter_frameNum,:) = vals_neurons;
end

if CE_experimentRunning
    if mod(counter_frameNum-1 , subSampleFactor_runningVals) == 0
        next_idx = mod(counter_runningVals-1 , numSamples_rollingStats)+1;
        vals_old = runningVals(next_idx , :);
        runningVals(next_idx,:) = vals_neurons;
        % 20230126 Now, counter_frameNum follows ScanImage Frames Done
%         [rolling_var_obj_cells , F_mean , F_var] = rolling_var_obj_cells.step(counter_frameNum , runningVals(next_idx,:) , vals_old);
        [rolling_var_obj_cells , F_mean , F_var] = rolling_var_obj_cells.step(runningVals(next_idx,:) , vals_old);
        counter_runningVals = counter_runningVals+1;
    end
    if counter_frameNum == 1
        F_mean = vals_neurons;
        F_var = ones(size(F_mean));
    end
    
    F_std = sqrt(F_var);
    %     F_std = nanstd(runningVals , [] , 1);
    %     F_std = ones(size(vals_smooth));
    F_std(F_std < 0.01) = inf;
    %     F_mean = nanmean(runningVals , 1);
    F_zscore = single((vals_neurons-F_mean)./F_std);
    F_zscore(isnan(F_zscore)) = 0;
    cursor_brain_raw = F_zscore * baselineStuff.ROIs.cellWeightings';
    logger.decoder(counter_frameNum,2) = cursor_brain_raw;
    
    next_idx = mod(counter_runningCursor-1 , duration_rollingStats)+1;
    vals_old = running_cursor_raw(next_idx);
    running_cursor_raw(next_idx) = cursor_brain_raw;
%     [rolling_var_obj_cursor , cursor_mean , cursor_var] = rolling_var_obj_cursor.step(counter_frameNum , running_cursor_raw(next_idx) , vals_old);
    [rolling_var_obj_cursor , cursor_mean , cursor_var] = rolling_var_obj_cursor.step(running_cursor_raw(next_idx) , vals_old);
    counter_runningCursor = counter_runningCursor+1;
    
    if counter_frameNum >= win_smooth
%         cursor_brain = mean(logger.decoder(counter_frameNum-(win_smooth-1):counter_frameNum,2));
        cursor_brain = mean((logger.decoder(counter_frameNum-(win_smooth-1):counter_frameNum,2)-cursor_mean)./sqrt(cursor_var));
    else
        cursor_brain = cursor_brain_raw;
    end
    
    %% Trial prep
    trialType_cursorOn = trialStuff.condTrialBool(trialNum,1);
    trialType_feedbackLinked = trialStuff.condTrialBool(trialNum,2);
    trialType_rewardOn = trialStuff.condTrialBool(trialNum,3);
    
    if CE_trial && (~trialType_feedbackLinked) && (~isnan(counter_trialIdx))
        cursor_output = trialStuff.fakeFeedback.fakeCursors(trialNum, counter_trialIdx+1);
        fakeFeedback_inUse = 1;
    else
        cursor_output = cursor_brain;
        fakeFeedback_inUse = 0;
    end
    
    CS_quiescence = algorithm_quiescence(cursor_brain, threshold_quiescence);
    CS_threshold = algorithm_thresholdState(cursor_output, threshold_value);
    
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
        counter_quiescenceHold = 0;
    end
    % WAIT FOR BASELINE
    if CE_waitForBaseline
        if CS_quiescence == 1
            counter_quiescenceHold = counter_quiescenceHold + 1;
        else
            counter_quiescenceHold = 0;
        end
    end
    % END WAIT FOR BASELINE (QUIESCENCE ACHIEVED)
%     if CE_waitForBaseline && CS_quiescence
%         CE_waitForBaseline = 0;
%         ET_trialStart = 1;
%     end

    % 2022/10/10 Let them HOLD the quiescence
    if CE_waitForBaseline && (counter_quiescenceHold > duration_quiescenceHold*frameRate)
        CE_waitForBaseline = 0;
        ET_trialStart = 1;
    end 
    
    % START TRIAL
    if ET_trialStart
        ET_trialStart = 0;
        CE_trial = 1;
        counter_trialIdx = 0;
        counter_CS_threshold = 0;
        
        updateLoggerTrials_START
        if ~trialType_feedbackLinked
            cursor_output = trialStuff.fakeFeedback.fakeCursors(trialNum, counter_trialIdx+1);
            fakeFeedback_inUse = 1;
        end
        if trialType_cursorOn
            soundVolume = 1;
        else
            soundVolume = 0;
        end
        %     setSoundVolumeTeensy(soundVolume);
%         source.hSI.task_cursorAmplitude.writeDigitalData(soundVolume);
%         source.hSI.task_goalAmplitude.writeDigitalData(1);
%         source.hSI.task_cursorGoalPos.writeAnalogData(3.2);
        
        frequencyOverride = 0;
    end
    % COUNT TRIAL DURATION & COUNT THRESHOLD DURATIONS
    if CE_trial
        counter_trialIdx = counter_trialIdx + 1;
        
        if CS_threshold
            counter_CS_threshold = counter_CS_threshold + 1;
        else
            counter_CS_threshold = 0;
        end
    end
    
    % END TRIAL: FAILURE
    if CE_trial && counter_trialIdx >= round(frameRate * duration_trial)
        CE_trial = 0;
        ET_timeout = 1;
        counter_trialIdx = NaN;
        fakeFeedback_inUse = 0;
        updateLoggerTrials_END(0)
        trialNum = trialNum+1;
    end
    % START TIMEOUT
    if ET_timeout
        ET_timeout = 0;
        CE_timeout = 1;
        counter_timeout = 0;
        soundVolume = 0;
        NumOfTimeouts = NumOfTimeouts + 1;
        %     setSoundVolumeTeensy(soundVolume);
%         source.hSI.task_cursorAmplitude.writeDigitalData(soundVolume);
%         source.hSI.task_goalAmplitude.writeDigitalData(0);
    end
    % COUNT TIMEOUT DURATION
    if CE_timeout
        counter_timeout = counter_timeout + 1;
    end
    % END TIMEOUT
    if CE_timeout && counter_timeout >= round(frameRate * duration_timeout)
        CE_timeout = 0;
        ET_ITI_withZ = 1;
    end
    
    % END TRIAL: THRESHOLD REACHED
    if CE_trial && counter_CS_threshold >= round(frameRate * duration_threshold)
        updateLoggerTrials_END(1)
        CE_trial = 0;
        %     ET_rewardToneHold = 1;
        ET_rewardDelivery = 1;
        trialNum = trialNum+1;
        counter_trialIdx = NaN;
        fakeFeedback_inUse = 0;
    end
    
    % START DELIVER REWARD
    if ET_rewardDelivery
        ET_rewardDelivery = 0;
        CE_rewardDelivery = 1;
        counter_rewardDelivery = 0;
        frequencyOverride = 1;
        %     soundVolume = 0;
        %     setSoundVolumeTeensy(soundVolume);
        if trialType_rewardOn
%         %     giveReward2(1, 1, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
%             giveReward3(source, 1, 0, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
%         %         giveReward3(source, 1, 0, 500, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
        end
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
        ET_ITI_withZ = 1;
        frequencyOverride = 0;
    end
    
    delta_moved = 0; % place holder to potentially be overwritten by 'moveFastZ' function below
    % START INTER-TRIAL-INTERVAL (POST-REWARD): WITH Z-CORRECTION
    if ET_ITI_withZ
        ET_ITI_withZ = 0;
        CE_ITI_withZ = 1;
        counter_ITI_withZ = 0;
        soundVolume = 0;
%         source.hSI.task_cursorAmplitude.writeDigitalData(soundVolume);
%         source.hSI.task_goalAmplitude.writeDigitalData(0);
        
        if counter_frameNum > (counter_last_z_correction + interval_z_correction)

            
%             if delta ~=0
%                 clampedDelta = sign(delta) * min(abs(delta), max_z_delta);
%                 disp(['moving fast Z by one step: ', num2str(clampedDelta)]) %num2str(delta)])
% %                 currentPosition = moveFastZ(source, [], clampedDelta, [], [20,380]);
%                 delta_moved = clampedDelta;
%                 counter_last_z_correction = counter_frameNum;
%             end
        end
        
    end
    % COUNT INTER-TRIAL-INTERVAL (POST-REWARD)
    if CE_ITI_withZ
        counter_ITI_withZ = counter_ITI_withZ + 1;
    end
    % END INTER-TRIAL-INTERVAL
    if CE_ITI_withZ && counter_ITI_withZ >= round(frameRate * duration_ITI)
        counter_ITI_withZ = NaN;
        CE_ITI_withZ = 0;
        ET_waitForBaseline = 1;
    end
    
end
%% Teensy Output calculations

if CE_experimentRunning
    if frequencyOverride
        voltage_cursorCurrentPos = convert_cursor_to_voltage(threshold_value , range_cursor , voltage_at_threshold);
    else
        voltage_cursorCurrentPos = convert_cursor_to_voltage(cursor_output , range_cursor, voltage_at_threshold);
    end
%     voltage_cursorCurrentPos = (mod(counter_frameNum,2)+0.5);
%     source.hSI.task_cursorCurrentPos.writeAnalogData(voltage_cursorCurrentPos);

    freqToOutput = convert_voltage_to_frequency(voltage_cursorCurrentPos , 3.3 , range_freqOutput); % for logging purposes only. function should mimic (exactly) the voltage to frequency transformation on teensy
end

% save([directory , '\logger.mat'], 'logger')
% saveParams(directory)
% disp(['Logger & Params Saved: frameCounter = ' num2str(counter_frameNum)]);
%% Plotting

% if CE_experimentRunning
%     % %     plotUpdatedOutput([cursor , dFoF.*scale_factors], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 20)
%
%     %     plotUpdatedOutput([cursor , dFoF], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 59)
%
%     plotUpdatedOutput2([CE_waitForBaseline*.9, CE_trial*.8, CE_rewardToneHold*.7,...
%         CE_rewardDelivery*.6, CE_ITI_successful*.5, CE_timeout*.4,...
%         CE_buildingUpStats*.3, CE_experimentRunning*.2]...
%         , duration_plotting, frameRate, 'Rewards', 10, 52, ['Num of Rewards:  ' , num2str(NumOfRewardsAcquired)])
%
%     plotUpdatedOutput3([xShift' yShift'], duration_plotting, frameRate, 'Motion Correction Shifts', 10, 51)
%
%     %     if counter_frameNum > 15
%     %         %     if counter_frameNum > 1
%     %         plotUpdatedOutput4([nanmean(logger.motionCorrection.MC_correlation(counter_frameNum-15:counter_frameNum,:),1)], duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 62)
%     %         %         plotUpdatedOutput4([nanmean(logger.motionCorrection.MC_correlation(counter_frameNum-15:counter_frameNum,:),1)], duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 12)
%     %     end
%
%     %     plotUpdatedOutput4(dFoF(1:10) , duration_plotting , frameRate , 'dFoF' , 20, 50)
%     plotUpdatedOutput6(cursor , duration_plotting , frameRate , 'cursor' , 10,53)
%     %     plotUpdatedOutput7(logger.motionCorrection.MC_correlation(counter_frameNum-1) , duration_plotting , frameRate , 'Motion Correction Correlation' , 10,54)
%
%
%     %     if mod(counter_frameNum,300) == 0 && counter_frameNum > 300
%     %         plotUpdatedOutput5([nanmean(logger.motionCorrection.MC_correlation(counter_frameNum-300:counter_frameNum,:),1)],...
%     %             duration_session, frameRate, 'Motion Correction Correlation All', 10, 50)
%     %     end
%
%     %     if show_MC_ref_images
%     %             if mod(counter_frameNum,1) == 0
%     % %         if mod(counter_frameNum,11) == 0
%     %             plotUpdatedMotionCorrectionImage_singleRegion(baselineStuff.MC.meanImForMC_crop , img_MC_moving_rollingAvg , img_MC_moving_rollingAvg, 'Motion Correction')
%     %         end
%     %     end
%
%
% end

% % if mod(counter_frameNum,30*60*5) == 0
% if counter_frameNum == round(duration_session * 0.9)...
%         || counter_frameNum == round(duration_session * 0.95)...
%         || counter_frameNum == round(duration_session * 0.98)...
%         || counter_frameNum == round(duration_session * 0.99)
%     save([directory , '\logger.mat'], 'logger')
%     disp(['Logger Saved: frameCounter = ' num2str(counter_frameNum)]);
% end

% counter_frameNum

%% DATA LOGGING

if ~isnan(counter_frameNum)
    %     logger_valsROIs(counter_frameNum,:) = vals_neurons; %already done above
    
    logger.timeSeries(counter_frameNum,1) = counter_frameNum;
    logger.timeSeries(counter_frameNum,2) = CS_quiescence;
    logger.timeSeries(counter_frameNum,3) = ET_trialStart;
    logger.timeSeries(counter_frameNum,4) = CE_trial;
    logger.timeSeries(counter_frameNum,5) = soundVolume;
    logger.timeSeries(counter_frameNum,6) = counter_trialIdx;
    logger.timeSeries(counter_frameNum,7) = CS_threshold;
    logger.timeSeries(counter_frameNum,8) = ET_rewardToneHold; % reward signals
    logger.timeSeries(counter_frameNum,9) = CE_rewardToneHold;
    logger.timeSeries(counter_frameNum,10) = counter_rewardToneHold;
    logger.timeSeries(counter_frameNum,11) = frequencyOverride;
    logger.timeSeries(counter_frameNum,12) = ET_rewardDelivery;
    logger.timeSeries(counter_frameNum,13) = CE_rewardDelivery;
    logger.timeSeries(counter_frameNum,14) = counter_rewardDelivery;
    logger.timeSeries(counter_frameNum,15) = ET_ITI_withZ;
    logger.timeSeries(counter_frameNum,16) = CE_ITI_withZ;
    logger.timeSeries(counter_frameNum,17) = counter_ITI_withZ;
    logger.timeSeries(counter_frameNum,18) = ET_waitForBaseline;
    logger.timeSeries(counter_frameNum,19) = CE_waitForBaseline;
    logger.timeSeries(counter_frameNum,20) = ET_timeout;
    logger.timeSeries(counter_frameNum,21) = CE_timeout;
    logger.timeSeries(counter_frameNum,22) = counter_timeout;
    logger.timeSeries(counter_frameNum,23) = CE_buildingUpStats;
    logger.timeSeries(counter_frameNum,24) = CE_experimentRunning;
    logger.timeSeries(counter_frameNum,25) = NumOfRewardsAcquired;
    logger.timeSeries(counter_frameNum,26) = NumOfTimeouts;
%     logger.timeSeries(counter_frameNum,27) = hash_image;
    logger.timeSeries(counter_frameNum,28) = trialNum;
    logger.timeSeries(counter_frameNum,29) = fakeFeedback_inUse;
    logger.timeSeries(counter_frameNum,30) = trialType_cursorOn;
    logger.timeSeries(counter_frameNum,31) = trialType_feedbackLinked;
    logger.timeSeries(counter_frameNum,32) = trialType_rewardOn;
    
    
    logger.timers(counter_frameNum,1) = now;
    logger.timers(counter_frameNum,2) = toc;
    
    logger.decoder(counter_frameNum,1) = cursor_brain;
    logger.decoder(counter_frameNum,2) = cursor_brain_raw;   % this is computed above
    logger.decoder(counter_frameNum,3) = cursor_output;
    logger.decoder(counter_frameNum,4) = freqToOutput; % note that this is just approximate, since calculation is done on teensy
    logger.decoder(counter_frameNum,5) = voltage_cursorCurrentPos;
    
%     logger.motionCorrection(counter_frameNum,1) = xShift;
%     logger.motionCorrection(counter_frameNum,2) = yShift;
%     logger.motionCorrection(counter_frameNum,3) = MC_corr;
end

% %% End Session
% if counter_frameNum >= duration_session
%     endSession
% end
% toc
%% FUNCTIONS
    function updateLoggerTrials_START % calls at beginning of a trial
        logger.trials(trialNum,1) = trialNum;
        logger.trials(trialNum,2) = now;
        logger.trials(trialNum,3) = counter_frameNum;
        logger.trials(trialNum,4) = trialType_cursorOn;
        logger.trials(trialNum,5) = trialType_feedbackLinked;
        logger.trials(trialNum,6) = trialType_rewardOn;
    end
    function updateLoggerTrials_END(success_outcome) % calls at end of a trial
        logger.trials(trialNum,7) = trialNum;
        logger.trials(trialNum,8) = now;
        logger.trials(trialNum,9) = counter_frameNum;
        logger.trials(trialNum,10) = success_outcome;
    end
    function startSession
        % INITIALIZE VARIABLES
        CE_waitForBaseline = 0;
        CS_quiescence = 0;
        ET_trialStart = 0;
        CE_trial = 0;
        soundVolume = 0;
        counter_trialIdx = 0;
        CS_threshold = 0;
        ET_rewardToneHold = 0; % reward signals
        CE_rewardToneHold = 0;
        counter_rewardToneHold = 0;
        frequencyOverride = 0;
        ET_rewardDelivery = 0;
        CE_rewardDelivery = 0;
        counter_rewardDelivery = 0;
        ET_ITI_withZ = 0;
        CE_ITI_withZ = 0;
        counter_ITI_withZ = 0;
        ET_waitForBaseline = 0;
        ET_timeout = 0;
        CE_timeout = 0;
        counter_timeout = 0;
        
        %         counter_frameNum = 0;
        CE_buildingUpStats = 1;
        CE_experimentRunning = 1;
        cursor_brain = 0;
        cursor_brain_raw = 0;
        cursor_output = 0;
        
        NumOfRewardsAcquired = 0;
        NumOfTimeouts = 0;
        trialNum = 1;
        
        loggerNames.timeSeries{1} = 'counter_frameNum';
        loggerNames.timeSeries{2} = 'CS_quiescence';
        loggerNames.timeSeries{3} = 'ET_trialStart';
        loggerNames.timeSeries{4} = 'CE_trial';
        loggerNames.timeSeries{5} = 'soundVolume';
        loggerNames.timeSeries{6} = 'counter_trialIdx';
        loggerNames.timeSeries{7} = 'CS_threshold';
        loggerNames.timeSeries{8} = 'ET_rewardToneHold'; % reward signals
        loggerNames.timeSeries{9} = 'CE_rewardToneHold';
        loggerNames.timeSeries{10} = 'counter_rewardToneHold';
        loggerNames.timeSeries{11} = 'frequencyOverride';
        loggerNames.timeSeries{12} = 'ET_rewardDelivery';
        loggerNames.timeSeries{13} = 'CE_rewardDelivery';
        loggerNames.timeSeries{14} = 'counter_rewardDelivery';
        loggerNames.timeSeries{15} = 'ET_ITI_withZ';
        loggerNames.timeSeries{16} = 'CE_ITI_withZ';
        loggerNames.timeSeries{17} = 'counter_ITI_withZ';
        loggerNames.timeSeries{18} = 'ET_waitForBaseline';
        loggerNames.timeSeries{19} = 'CE_waitForBaseline';
        loggerNames.timeSeries{20} = 'ET_timeout';
        loggerNames.timeSeries{21} = 'CE_timeout';
        loggerNames.timeSeries{22} = 'counter_timeout';
        loggerNames.timeSeries{23} = 'CE_buildingUpStats';
        loggerNames.timeSeries{24} = 'CE_experimentRunning';
        loggerNames.timeSeries{25} = 'NumOfRewardsAcquired';
        loggerNames.timeSeries{26} = 'NumOfTimeouts';
        loggerNames.timeSeries{27} = 'image_hash';
        loggerNames.timeSeries{28} = 'trialNum';
        loggerNames.timeSeries{29} = 'fakeFeedback_inUse';
        loggerNames.timeSeries{30} = 'trialType_cursorOn';
        loggerNames.timeSeries{31} = 'trialType_feedbackLinked';
        loggerNames.timeSeries{32} = 'trialType_rewardOn';
        
        loggerNames.timers{1} = 'time_now';
        loggerNames.timers{2} = 'tic_toc';
        
        loggerNames.decoder{1} = 'cursor_brain';
        loggerNames.decoder{2} = 'cursor_brain_raw';
        loggerNames.decoder{3} = 'cursor_output';
        loggerNames.decoder{4} = 'freqToOutput';
        loggerNames.decoder{5} = 'voltage_cursorCurrentPos';
        
        loggerNames.motionCorrection{1} = 'xShift';
        loggerNames.motionCorrection{2} = 'yShift';
        loggerNames.motionCorrection{3} = 'MC_correlation';
        
        loggerNames.trials{1} = 'trialNum_trialStart';
        loggerNames.trials{2} = 'time_now_trialStart';
        loggerNames.trials{3} = 'counter_frameNum_trialStart';
        loggerNames.trials{4} = 'trialType_cursorOn';
        loggerNames.trials{5} = 'trialType_feedbackLinked';
        loggerNames.trials{6} = 'trialType_rewardOn';
        loggerNames.trials{7} = 'trialNum_trialEnd';
        loggerNames.trials{8} = 'time_now_trialEnd';
        loggerNames.trials{9} = 'counter_frameNum_trialEnd';
        loggerNames.trials{10} = 'success_outcome';
        
        %         clear logger
        logger.timeSeries = NaN(duration_session, length(loggerNames.timeSeries));
        logger.timers = NaN(duration_session, length(loggerNames.timers));
        logger.decoder = NaN(duration_session, length(loggerNames.decoder));
        logger.motionCorrection = NaN(duration_session,  length(loggerNames.motionCorrection));
        logger.trials = NaN(size(trialStuff.condTrials,1),  length(loggerNames.trials));

        logger_valsROIs = nan(duration_session , numCells);
        runningVals = nan(numSamples_rollingStats , baselineStuff.ROIs.num_cells);
        running_cursor_raw = nan(numSamples_rollingStats , 1);
        
        rolling_var_obj_cells = rolling_var_and_mean();
        rolling_var_obj_cells = rolling_var_obj_cells.set_key_properties(size(runningVals) , duration_rollingStats);
        rolling_var_obj_cursor = rolling_var_and_mean();
        rolling_var_obj_cursor = rolling_var_obj_cursor.set_key_properties([1,1] , duration_rollingStats);
        
        counter_runningVals = 1;
        counter_runningCursor = 1;
        
        %         saveParams(directory)
    end

%     function endSession
%         counter_frameNum = NaN;
%         CE_experimentRunning = 0;
%
%         save([directory , '\logger.mat'], 'logger')
%         saveParams(directory)
%         disp('SESSION OVER')
%
%         CE_waitForBaseline = 0;
%         CS_quiescence = 0;
%         ET_trialStart = 0;
%         CE_trial = 0;
%         soundVolume = 0;
%         counter_trialIdx = 0;
%         CS_threshold = 0;
%         ET_rewardToneHold = 0; % reward signals
%         CE_rewardToneHold = 0;
%         counter_rewardToneHold = 0;
%         frequencyOverride = 0;
%         ET_rewardDelivery = 0;
%         CE_rewardDelivery = 0;
%         counter_rewardDelivery = 0;
%         ET_ITI_successful = 0;
%         CE_ITI_successful = 0;
%         counter_ITI_successful = 0;
%         ET_waitForBaseline = 0;
%         CE_waitForBaseline = 0;
%         ET_timeout = 0;
%         CE_timeout = 0;
%         counter_timeout = 0;
%
%         %         counter_frameNum = 0;
%         CE_buildingUpStats = 0;
%         %         CE_experimentRunning = 0;
%         cursor = 0;
% %         dFoF = 0;
%
%         %         setSoundVolumeTeensy(0);
% %         source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(0);
%
%     end

%     function saveParams(directory)
%         expParams.frameRate = frameRate;
%         expParams.duration_session = duration_session;
%         expParams.duration_trial = duration_trial;
%         expParams.win_smooth = win_smooth;
%         expParams.F_baseline_prctile = F_baseline_prctile;
% %         expParams.scale_factors = scale_factors;
% %         expParams.ensemble_assignments = ensemble_assignments;
%         expParams.duration_threshold = duration_threshold;
%         expParams.threshold_value = threshold_value;
%         expParams.range_cursor = range_cursor;
%         expParams.range_freqOutput = range_freqOutput;
%         expParams.duration_timeout = duration_timeout;
%         expParams.numCells = numCells;
%         expParams.directory = directory;
%         expParams.duration_rollingStats = duration_rollingStats;
%         expParams.subSampleFactor_runningVals = subSampleFactor_runningVals;
%         expParams.threshold_quiescence = threshold_quiescence;
%         expParams.duration_rewardTone = duration_rewardTone;
%         expParams.duration_ITI_success = duration_ITI_success;
%         expParams.duration_rewardDelivery = duration_rewardDelivery;
%         expParams.reward_duration = reward_duration; % in ms
%         expParams.reward_delay = reward_delay;
%         expParams.LED_duration = LED_duration;
%         expParams.LED_ramp_duration = LED_ramp_duration;
%         expParams.numFramesToAvgForMotionCorr = numFramesToAvgForMotionCorr;
%
%         expParams.loggerNames = loggerNames;
%
%         expParams.baselineStuff = baselineStuff;
%
%         save([directory , '\expParams.mat'], 'expParams')
%         %         save([directory , '\motionCorrectionRefImages.mat'], 'motionCorrectionRefImages')
%     end

if counter_frameNum == last_frameNum
    logger_output = logger;
    logger_valsROIs_output = logger_valsROIs;
else
    logger_output = [];
    logger_valsROIs_output = [];
end
% toc
end