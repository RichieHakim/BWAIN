function currentLoggerLine = BMI_trial_simulation(vals,startSession_pref, scale_factors, threshold_value, duration_session, duration_trial, varargin)
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
win_smooth = 3; % smoothing window (in frames)
F_baseline_prctile = 20; % percentile to define as F_baseline of cells

% scale_factors = [4.52010933877283,8.76604870159710,5.90995393712342,7.47940285687949];
% scale_factors = baselineStuff.scale_factors;
ensemble_assignments = [1 1 2 2];
numCells = numel(ensemble_assignments);

% reward vars
% threshold_value =   2.5;

range_cursorSound = [-threshold_value , threshold_value];
range_freqOutput = [2000 20000]; % this is set in the teensy code (Ofer made it)
%% Trial Structure settings & Rolling Stats settings
% duration_session = frameRate * 60 * 2; % ADJUSTABLE: change number value (in seconds/minutes)
% duration_trial = 30;
duration_timeout = 5;
duration_threshold = 0.066;
duration_rewardTone = 1.5;
duration_ITI_success = 1;
duration_rewardDelivery = 1;

duration_rollingStats = round(frameRate * 30);

baselinePeriodSTD = 1./scale_factors;
threshold_quiescence = baselinePeriodSTD;

%% == Session Starting & counting ==
if startSession_pref
startSession
end

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
            runningVals = [nan(duration_rollingStats, numCells) ; vals];
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
    soundVolume = 0;
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

% END TRIAL: FAILURE
if CE_trial && counter_trialDuration >= round(frameRate * duration_trial)
    CE_trial = 0;
    ET_timeout = 1;
end

% START TIMEOUT
if ET_timeout
    ET_timeout = 0;
    CE_timeout = 1;
    counter_timeout = 0;
    soundVolume = 0;
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
    counter_trialDuration = NaN;
    CE_trial = 0;
    ET_rewardToneHold = 1;
end

% START REWARD TONE
if ET_rewardToneHold
    ET_rewardToneHold = 0;
    CE_rewardToneHold = 1;
    counter_rewardToneHold = 0;
    soundVolume = 3.3;
    NumOfRewardsAcquired = NumOfRewardsAcquired+1;
end
% COUNT REWARD TONE DURATION
if CE_rewardToneHold
    counter_rewardToneHold = counter_rewardToneHold + 1;
    frequencyOverride = 1;
end
% END REWARD TONE
if CE_rewardToneHold && counter_rewardToneHold >= round(frameRate * duration_rewardTone)
    counter_rewardToneHold = NaN;
    CE_rewardToneHold = 0;
    ET_rewardDelivery = 1;
    frequencyOverride = 0;
end

% START DELIVER REWARD
if ET_rewardDelivery
    ET_rewardDelivery = 0;
    CE_rewardDelivery = 1;
    counter_rewardDelivery = 0;
    soundVolume = 0;
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
%% DATA LOGGING
if ~isnan(counter_frameNum)
    logger(counter_frameNum,1) = CE_waitForBaseline;
    logger(counter_frameNum,2) = CS_quiescence;
    logger(counter_frameNum,3) = ET_trialStart;
    logger(counter_frameNum,4) = CE_trial;
    logger(counter_frameNum,5) = soundVolume;
    logger(counter_frameNum,6) = counter_trialDuration;
    logger(counter_frameNum,7) = CS_threshold;
    logger(counter_frameNum,8) = ET_rewardToneHold; 
    logger(counter_frameNum,9) = CE_rewardToneHold; % reward signals
    logger(counter_frameNum,10) = counter_rewardToneHold;
    logger(counter_frameNum,11) = frequencyOverride;
    logger(counter_frameNum,12) = NaN;
    logger(counter_frameNum,13) = NaN;
    
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
% counter_frameNum
currentLoggerLine = logger(counter_frameNum,:);

%% End Session
if counter_frameNum >= duration_session
    endSession
end

%% FUNCTIONS
    function startSession
        logger = NaN(duration_session,(29+numCells+numCells-1));
        
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
    end
end