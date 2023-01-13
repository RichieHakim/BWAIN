function voltage_cursorCurrentPos =...
    userFunction_BMIv11_withZ_baseline(source, event, varargin)
%% Variable stuff
tic
global counter_frameNum counter_trialIdx counter_CS_threshold counter_timeout counter_rewardToneHold...
    counter_rewardDelivery counter_ITI_successful...
    CE_buildingUpStats CE_experimentRunning CE_waitForBaseline CE_trial CE_timeout CE_rewardToneHold...
    CE_rewardDelivery CE_ITI_successful...
    ET_waitForBaseline ET_trialStart ET_timeout ET_rewardToneHold ET_rewardDelivery ET_ITI_successful...
    frequencyOverride...
    NumOfRewardsAcquired NumOfTimeouts trialNum soundVolume...
    img_MC_moving img_MC_moving_rolling...
    logger loggerNames logger_valsROIs...
    runningVals running_cursor_raw counter_runningVals counter_runningCursor...
    rolling_var_obj_cells rolling_var_obj_cursor loadedCheck_registrationImage...
    registrationImage referenceDiffs refIm_crop_conjFFT_shift refIm_crop indRange_y_Crop indRange_x_Crop...
    img_MC_moving_rolling_z refIm_crop_conjFFT_shift_masked counter_last_z_correction...
    

%% == USER SETTINGS ==
% ROI vars
frameRate                   = 30;
duration_session            = frameRate * 60 * 60; % ADJUSTABLE: change number value (in seconds/minutes)

threshold_value             = 0.0;

% range_cursor = [-threshold_value , threshold_value *1.5];
range_cursor = [-threshold_value threshold_value];
range_freqOutput = [1000 18000]; % this is set in the teensy code (only here for logging purposes)
voltage_at_threshold = 3.1; % this will be the maximum output voltage ([0:voltage_at_threshold])

reward_duration = 52; % in ms
reward_delay = 0; % in ms

%% Trial Structure settings & Rolling Stats settings
% below in unit seconds
duration_trial          = 20;
duration_timeout        = 4;
duration_threshold      = 0.066;
duration_rewardTone     = 1.5; % currently unused
duration_ITI_success    = 0.5;
duration_rewardDelivery = 0.20;

duration_plotting = 30*10;

% duration_rollingStats       = round(frameRate * 60 * 15);
% subSampleFactor_runningVals = 1;
% numSamples_rollingStats = round(duration_rollingStats/subSampleFactor_runningVals);
duration_buildingUpStats    = 2;
duration_rollingStats = duration_buildingUpStats;

threshold_quiescence    = 1;

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


if CE_experimentRunning
    %% Trial prep

%     if CE_trial && (~trialType_feedbackLinked) && (~isnan(counter_trialIdx))
%         cursor_output = trialStuff.fakeFeedback.fakeCursors(trialNum, counter_trialIdx+1);
%         fakeFeedback_inUse = 1;
%     else
%         disp('trialType must always be feedbackLinked==0');
%         fakeFeedback_inUse = 0;
%     end


    cursor_output = rand(1);
    
    CS_quiescence = algorithm_quiescence(cursor_output, threshold_quiescence);
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
        counter_trialIdx = 0;
        counter_CS_threshold = 0;
        
        soundVolume = 1;
        soundVolume = 0;
        %     setSoundVolumeTeensy(soundVolume);
        source.hSI.task_cursorAmplitude.writeDigitalData(soundVolume);
        source.hSI.task_goalAmplitude.writeDigitalData(1);
        source.hSI.task_cursorGoalPos.writeAnalogData(3.2);
        
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
        trialNum = trialNum+1;
        counter_trialIdx = NaN;
        fakeFeedback_inUse = 0;
        updateLoggerTrials_END(0)
    end
    % START TIMEOUT
    if ET_timeout
        ET_timeout = 0;
        CE_timeout = 1;
        counter_timeout = 0;
        soundVolume = 0;
        NumOfTimeouts = NumOfTimeouts + 1;
        %     setSoundVolumeTeensy(soundVolume);
        source.hSI.task_cursorAmplitude.writeDigitalData(soundVolume);
        source.hSI.task_goalAmplitude.writeDigitalData(0);
    end
    % COUNT TIMEOUT DURATION
    if CE_timeout
        counter_timeout = counter_timeout + 1;
    end
    % END TIMEOUT
    if CE_timeout && counter_timeout >= round(frameRate * duration_timeout)
        CE_timeout = 0;
        ET_waitForBaseline = 1;
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
        %     giveReward2(1, 1, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
        giveReward3(source, 1, 0, reward_duration, reward_delay, 0, 1, 0); % in ms. This is in the START section so that it only delivers once
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
        frequencyOverride = 0;
    end
    
            
    % START INTER-TRIAL-INTERVAL (POST-REWARD): WITH Z-CORRECTION
    if ET_ITI_successful
        ET_ITI_successful = 0;
        CE_ITI_successful = 1;
        counter_ITI_successful = 0;
        soundVolume = 0;
        source.hSI.task_cursorAmplitude.writeDigitalData(soundVolume);
        source.hSI.task_goalAmplitude.writeDigitalData(0);
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
        voltage_cursorCurrentPos = convert_cursor_to_voltage(threshold_value , range_cursor , voltage_at_threshold);
    else
        voltage_cursorCurrentPos = convert_cursor_to_voltage(cursor_output , range_cursor, voltage_at_threshold);
    end
%     voltage_cursorCurrentPos
%     voltage_cursorCurrentPos = (mod(counter_frameNum,2)+0.5);
    source.hSI.task_cursorCurrentPos.writeAnalogData(double(voltage_cursorCurrentPos));

    freqToOutput = convert_voltage_to_frequency(voltage_cursorCurrentPos , 3.3 , range_freqOutput); % for logging purposes only. function should mimic (exactly) the voltage to frequency transformation on teensy
end

% save([directory , '\logger.mat'], 'logger')
% saveParams(directory)
% disp(['Logger & Params Saved: frameCounter = ' num2str(counter_frameNum)]);
%% Plotting

if counter_frameNum>1
    %     plotUpdatedOutput([cursor , dFoF.*scale_factors], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 20)
    
%         plotUpdatedOutput([cursor , dFoF], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 59)
    
    plotUpdatedOutput2([CE_waitForBaseline*0.1, CE_trial*0.2,...
        CE_rewardDelivery*0.3, CE_timeout*0.4, CE_buildingUpStats*0.5]...
        , duration_plotting, frameRate, 'Rewards', 10, 52, ['# Rewards: ' , num2str(NumOfRewardsAcquired) , ' ; # Timeouts: ' , num2str(NumOfTimeouts)])
    
    

    %     plotUpdatedOutput4(dFoF(1:10) , duration_plotting , frameRate , 'dFoF' , 20, 50)
    plotUpdatedOutput6(cursor_output , duration_plotting , frameRate , 'cursor' , 10,53)
    %     plotUpdatedOutput7(logger.motionCorrection(counter_frameNum,3) , duration_plotting , frameRate , 'Motion Correction Correlation' , 10,54)
    

%     plotUpdatedFrame(squeeze(mean(img_MC_moving_rolling_z,3)), 'test')
    
%     if show_MC_ref_images
%         if mod(counter_frameNum,1) == 0
%             %         if mod(counter_frameNum,11) == 0
%             plotUpdatedMotionCorrectionImage_singleRegion(baselineStuff.MC.meanImForMC_crop , img_MC_moving_rollingAvg , img_MC_moving_rollingAvg, 'Motion Correction')
%         end
%     end
    
    
end


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
        ET_ITI_successful = 0;
        CE_ITI_successful = 0;
        counter_ITI_successful = 0;
        ET_waitForBaseline = 0;
        CE_waitForBaseline = 0;
        ET_timeout = 0;
        CE_timeout = 0;
        counter_timeout = 0;
        counter_last_z_correction = 0;
        
        counter_frameNum = 0;
        CE_buildingUpStats = 1;
        CE_experimentRunning = 1;
%         cursor_brain = 0;
%         cursor_brain_raw = 0;
        cursor_output = 0;
        
        NumOfRewardsAcquired = 0;
        NumOfTimeouts = 0;
        trialNum = 1;
        
        counter_runningVals = 1;
        counter_runningCursor = 1;

%         saveParams(directory)
    end



end