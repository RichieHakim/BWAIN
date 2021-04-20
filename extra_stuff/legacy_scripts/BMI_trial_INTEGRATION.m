function teensyOutput = BMI_trial_INTEGRATION(vals,varargin)
%% Variable stuff
global baselineStuff ...
    counter_frameNum counter_CS_threshold counter_trialDuration counter_timeout counter_rewardToneHold...
    counter_rewardDelivery counter_ITI_successful...
    CE_buildingUpStats CE_experimentRunning CE_waitForBaseline CE_trial CE_timeout CE_rewardToneHold...
    CE_rewardDelivery CE_ITI_successful...
    ET_waitForBaseline ET_trialStart ET_timeout ET_rewardToneHold ET_rewardDelivery ET_ITI_successful...
    frequencyOverride...
    logger NumOfRewardsAcquired soundVolume
%% == USER SETTINGS ==
% ROI vars
frameRate = 30;
duration_plotting = 30 * frameRate; % ADJUSTABLE: change number value (in seconds)
duration_session = frameRate * 60 * 60; % ADJUSTABLE: change number value (in seconds/minutes)
win_smooth = 3; % smoothing window (in frames)
F_baseline_prctile = 20; % percentile to define as F_baseline of cells


% scale_factors = [3.33148349811192,6.68105656051084,11.2059977732239,14.6726435284673];
scale_factors = [4.92239470661177,7.37297845353029,9.18637299772757,7.56462289434561];
% scale_factors = baselineStuff.scale_factors;
ensemble_assignments = [1 1 2 2];
numCells = numel(ensemble_assignments);

threshold_value =   999999999;

% range_cursorSound = [-threshold_value , threshold_value];
range_cursorSound = [-2 10];
range_freqOutput = [2000 16000]; % this is set in the teensy code (Ofer made it)

reward_duration = 150; % in ms
reward_delay = 200; % in ms
LED_duration = 1; % in s
LED_ramp_duration = 0.1; % in s

% directory = [baselineStuff.directory, '\expParams.mat'];
% directory = ['F:\RH_Local', '\expParams.mat'];
directory = 'F:\RH_Local\Rich data\scanimage data\mouse 10.30\test';
%% Trial Structure settings & Rolling Stats settings
duration_trial = 30;
duration_timeout = 5;
duration_threshold = 0.066;
duration_rewardTone = 1.5;
duration_ITI_success = 1;
duration_rewardDelivery = 1;

duration_rollingStats = round(frameRate * 30);

baselinePeriodSTD = 1./scale_factors;
threshold_quiescence = baselinePeriodSTD;

%% == Session Starting & counting ==

% == Start Session ==
if isempty(counter_frameNum)
    disp('hi')
    startSession
end
% ======== COMMENT THIS IN/OUT TO START SESSION =======
% startSession
% =====================================================

counter_frameNum = counter_frameNum + 1;

% %% SAVING
% saveParams(directory)

% saveParams('F:\RH_Local\Rich data\scanimage data\20191110\mouse 10.13B\expParams.mat')

%% == ROLLING STATS ==
% CE_buildingUpStats
if CE_experimentRunning
    if ~CE_buildingUpStats
        runningVals = logger([(counter_frameNum - duration_rollingStats):counter_frameNum],29:29+numCells-1);
    end
    if (CE_buildingUpStats == 1) && (counter_frameNum > 1)
        runningVals = [nan(duration_rollingStats, numCells) ; logger(1:counter_frameNum, 29:29+numCells-1)];
    else if counter_frameNum == 1
            runningVals = [nan(duration_rollingStats, numCells) ; vals'];
        end
    end

    vals_smooth = nanmean(runningVals(end-win_smooth : end,:),1);
    F_baseline = prctile(runningVals, F_baseline_prctile);
    dF = vals_smooth - F_baseline;
    dFoF = dF ./ F_baseline;
    cursor = algorithm_decoder(dFoF, scale_factors, ensemble_assignments);
    CS_quiescence = algorithm_quiescence(dFoF, threshold_quiescence);   
    CS_threshold = algorithm_thresholdState(cursor, threshold_value);
end
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
if CE_buildingUpStats && counter_frameNum > duration_rollingStats
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
    setSoundVolumeTeensy(soundVolume);
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
    giveReward2(1, 0, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
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
%     soundVolume = 0;
%     setSoundVolumeTeensy(soundVolume);
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
%% Plotting
if CE_experimentRunning
    % plotUpdatedOutput([cursor , dFoF.*scale_factors], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 20)

        plotUpdatedOutput([cursor , dFoF], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 120)

        plotUpdatedOutput2([CE_waitForBaseline*.9, CE_trial*.8, CE_rewardToneHold*.7,...
        CE_rewardDelivery*.6, CE_ITI_successful*.5, CE_timeout*.4,...
        CE_buildingUpStats*.3, CE_experimentRunning*.2]...
        , duration_plotting, frameRate, 'Rewards', 10, 120, ['Num of Rewards:  ' , num2str(NumOfRewardsAcquired)])
    %     plotUpdatedOutput3([vals_zscore], length_History, frameRate, 'zscore , E1 , E2', 10, 20)
end
%% Teensy Output calculations
range_teensyInputVoltage = [0 3.3]; % using a teensy 3.5 currently

if CE_experimentRunning
    if frequencyOverride
        freqToOutput = cursorToFrequency(threshold_value, range_cursorSound, range_freqOutput);
    else
        freqToOutput = cursorToFrequency(cursor, range_cursorSound, range_freqOutput);
    end
    teensyOutput = teensyFrequencyToVoltageTransform(freqToOutput, range_freqOutput, range_teensyInputVoltage);
else
    teensyOutput = 0;
end

% teensyOutput = mod(counter_frameNum,10)/10;


% HARD CODED CONSTRAINTS ON OUTPUT VOLTAGE FOR TEENSY 3.5
if teensyOutput > 3.3
    teensyOutput = 3.3;
    %     warning('CURSOR IS TRYING TO GO ABOVE 3.3V')
end
if teensyOutput < 0
    teensyOutput = 0;
    %     warning('CURSOR IS TRYING TO GO BELOW 0V')
end

%% DATA LOGGING
if ~isnan(counter_frameNum)
    logger(counter_frameNum,1) = CE_waitForBaseline;
    logger(counter_frameNum,2) = CS_quiescence;
    logger(counter_frameNum,3) = ET_trialStart;
    logger(counter_frameNum,4) = CE_trial;
    logger(counter_frameNum,5) = soundVolume;
    logger(counter_frameNum,6) = counter_trialDuration;
    logger(counter_frameNum,7) = CS_threshold;
    logger(counter_frameNum,8) = ET_rewardToneHold; % reward signals
    logger(counter_frameNum,9) = CE_rewardToneHold;
    logger(counter_frameNum,10) = counter_rewardToneHold;
    logger(counter_frameNum,11) = frequencyOverride;
    logger(counter_frameNum,12) = freqToOutput;
    logger(counter_frameNum,13) = teensyOutput;
    
    logger(counter_frameNum,14) = ET_rewardDelivery;
    logger(counter_frameNum,15) = CE_rewardDelivery;
    logger(counter_frameNum,16) = counter_rewardDelivery;
    logger(counter_frameNum,17) = ET_ITI_successful;
    logger(counter_frameNum,18) = CE_ITI_successful;
    logger(counter_frameNum,19) = counter_ITI_successful;
    logger(counter_frameNum,20) = ET_waitForBaseline;
    logger(counter_frameNum,21) = CE_waitForBaseline;
    logger(counter_frameNum,22) = ET_timeout;
    logger(counter_frameNum,23) = CE_timeout;
    logger(counter_frameNum,24) = counter_timeout;
    
    logger(counter_frameNum,25) = counter_frameNum;
    logger(counter_frameNum,26) = CE_buildingUpStats;
    logger(counter_frameNum,27) = CE_experimentRunning;
    logger(counter_frameNum,28) = cursor;
    logger(counter_frameNum,29:29+numCells-1) = vals;
    logger(counter_frameNum,(29+numCells) : (29+numCells+numCells-1) ) = dFoF;
end

%% End Session
if counter_frameNum >= duration_session
    endSession
end

%% FUNCTIONS
    function startSession
        logger = NaN(duration_session,(29+numCells+numCells-1));
        saveParams(directory)
        
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
        dFoF = 0;
        
        NumOfRewardsAcquired = 0;
        
    end
    function endSession
        counter_frameNum = NaN;
        
        CE_experimentRunning = 0;
        save([directory , '\logger.mat'], 'logger')
        saveParams(directory)
        disp('SESSION OVER')
        
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
        dFoF = 0;
        
        setSoundVolumeTeensy(0);
    end
    function saveParams(directory)
        expParams.frameRate = frameRate;
        expParams.duration_session = duration_session;
        expParams.duration_trial = duration_trial;
        expParams.win_smooth = win_smooth;
        expParams.F_baseline_prctile = F_baseline_prctile;
        expParams.scale_factors = scale_factors;
        expParams.ensemble_assignments = ensemble_assignments;
        expParams.duration_threshold = duration_threshold;
        expParams.threshold_value = threshold_value;
        expParams.range_cursorSound = range_cursorSound;
        expParams.range_freqOutput = range_freqOutput;
        expParams.duration_timeout = duration_timeout;
        expParams.numCells = numCells;
        expParams.directory = directory;
        expParams.duration_rollingStats = duration_rollingStats;
        expParams.threshold_quiescence = threshold_quiescence;
        expParams.duration_rewardTone = duration_rewardTone;
        expParams.duration_ITI_success = duration_ITI_success;
        expParams.duration_rewardDelivery = duration_rewardDelivery;
        expParams.reward_duration = reward_duration; % in ms
        expParams.reward_delay = reward_delay;
        expParams.LED_duration = LED_duration;
        expParams.LED_ramp_duration = LED_ramp_duration;
        
        save([directory , '\expParams.mat'], 'expParams')
    end
end