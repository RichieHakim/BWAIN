function outputVoltageToTeensy = userFunction_RandomPlayback(source, event, varargin)
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
    
persistent registrationImage refIm_crop_conjFFT_shift loadedCheck_registrationImage...
    
%% Set stuff for motion correction

directory = 'D:\RH_local\data\scanimage data\round 6 experiments\mouse_1_19\zstack';

currentImage = source.hSI.hDisplay.lastFrame{1};


if ~exist('loadedCheck_registrationImage') | isempty(loadedCheck_registrationImage) | loadedCheck_registrationImage ~= 1
    tmp = load([directory , '\stack.mat']);
%     file_fieldName = fieldnames(tmp);
%     registrationImage = eval(['stack.' , file_fieldName{1}]);
    registrationImage = eval(['tmp.stack.' , 'stack_avg']);
    loadedCheck_registrationImage=1;
    
%     disp(size(registrationImage))
    
    clear refIm_crop_conjFFT_shift_centerIdx
    for ii = 1:size(registrationImage,1)
        refIm_crop_conjFFT_shift(ii,:,:) = make_fft_for_MC(squeeze(registrationImage(ii,:,:)));
    end
    
end
    
%% == USER SETTINGS ==
% ROI vars
frameRate = 30;
duration_plotting = 30 * frameRate; % ADJUSTABLE: change number value (in seconds)
duration_session = frameRate * 60 * 60; % ADJUSTABLE: change number value (in seconds/minutes)
win_smooth = 5; % smoothing window (in frames)
F_baseline_prctile = 20; % percentile to define as F_baseline of cells

scale_factors = [1 1 1 1];
scale_factors = [1 1 1 1];
ensemble_assignments = [1 1 2 2];
numCells = numel(ensemble_assignments);

% numFramesToAvgForMotionCorr = 10;

threshold_value =   0.8;

% range_cursorSound = [-threshold_value , threshold_value]*2;
range_cursorSound = [-0.5 , threshold_value]*2;
% range_cursorSound = [-2 10];
range_freqOutput = [2000 18000]; % this is set in the teensy code (Ofer made it)
voltage_at_threshold = 3.1; % this will be the maximum output voltage ([0:voltage_at_threshold])


% reward_duration = 150; % in ms
reward_duration = 64; % in ms
% reward_delay = 200; % in ms
reward_delay = 0; % in ms
LED_duration = .2; % in s
LED_ramp_duration = 0.1; % in s

% directory = [baselineStuff.directory, '\expParams.mat'];
% directory = ['F:\RH_Local', '\expParams.mat'];
%% Trial Structure settings & Rolling Stats settings
duration_trial = 30;
duration_timeout = 5;
duration_threshold = 0.066;
duration_rewardTone = 1.5;
% duration_ITI_success = 1;
duration_ITI_success = 2;
duration_rewardDelivery = 0.25;

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

%% == MOTION CORRECTION ==

numFramesToAvgForMotionCorr = 10;
n_slices = size(registrationImage, 1);

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
cxx = NaN(n_slices, 256);
cyy = NaN(n_slices, 256);
% disp(size(refIm_crop_conjFFT_shift))
% disp(size(squeeze(refIm_crop_conjFFT_shift_centerIdx(3,:,:))))

for z_frame = 3 % RH: ASSUMING THAT 3RD FRAME IS THE MIDDLE FRAME TO ALIGN TO
%     [yShift_tmp , xShift_tmp, cyy_tmp, cxx_tmp] = motionCorrection_ROI(img_MC_moving_rollingAvg , squeeze(registrationImage(z_frame,:,:)) );
    [xShift_tmp, yShift_tmp, cxx_tmp, cyy_tmp] = motionCorrection_ROI(crop_image_for_fft(img_MC_moving_rollingAvg) , [] , squeeze(refIm_crop_conjFFT_shift(z_frame,:,:)));
    xShift(z_frame,:) = squeeze(xShift_tmp);
    yShift(z_frame,:) = squeeze(yShift_tmp);
    cxx(z_frame,:) = cxx_tmp;
    cyy(z_frame,:) = cyy_tmp;
end

MC_corr = max(cxx, [], 2);

% xShift = 0;
% yShift = 0;
% MC_corr = 0;

% img_ROI_corrected{ii} = currentImage((baselineStuff.idxBounds_ROI{ii}(1,2):baselineStuff.idxBounds_ROI{ii}(2,2)) +round(yShift(ii)) ,...
%     (baselineStuff.idxBounds_ROI{ii}(1,1):baselineStuff.idxBounds_ROI{ii}(2,1)) +round(xShift(ii))); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]

% if abs(xShift) >80
%     xShift = 0;
% end
% if abs(yShift) >80
%     yShift = 0;
% end

%% == EXTRACT ROI VALUES ==
if numel(vals) ~= numCells
    vals = [];
end
for ii = 1:numCells
    %     vals(ii,1) = mean(img_ROI_corrected{ii}(:));
    
end
vals = make_fake_dFoF +1;

%% == ROLLING STATS ==
if CE_experimentRunning
    if ~CE_buildingUpStats
        runningVals = logger.decoder.rawVals([(counter_frameNum - duration_rollingStats):counter_frameNum],:);
    end
    if (CE_buildingUpStats == 1) && (counter_frameNum > 1)
        %         runningVals = [nan(duration_rollingStats, numCells) ; logger(1:counter_frameNum, 29:29+numCells-1)];
        runningVals = [nan(duration_rollingStats, numCells) ; logger.decoder.rawVals(1:counter_frameNum, :)];
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
%     CS_quiescence = 1;
    CS_threshold = algorithm_thresholdState(cursor, threshold_value);
%     CS_threshold = mod(counter_frameNum , 90) < 30;
%     CS_threshold = 1;
    
    %  ===== TRIAL STRUCTURE =====
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
        soundVolume = 1;
        %     setSoundVolumeTeensy(soundVolume);
        source.hSI.task_cursorAmplitude.writeDigitalData(soundVolume);
        source.hSI.task_goalAmplitude.writeDigitalData(1);
        source.hSI.task_cursorGoalPos.writeAnalogData(3.2);
                
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
        NumOfRewardsAcquired = NumOfRewardsAcquired + 1;
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

%% Plotting
if CE_experimentRunning
    % %     plotUpdatedOutput([cursor , dFoF.*scale_factors], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 20)
    
    plotUpdatedOutput([cursor , dFoF], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 5)
    
    plotUpdatedOutput2([CE_waitForBaseline*.9, CE_trial*.8, CE_rewardToneHold*.7,...
        CE_rewardDelivery*.6, CE_ITI_successful*.5, CE_timeout*.4,...
        CE_buildingUpStats*.3, CE_experimentRunning*.2]...
        , duration_plotting, frameRate, 'Rewards', 10, 1, ['Num of Rewards:  ' , num2str(NumOfRewardsAcquired)])
    
    %     plotUpdatedOutput3([xShift' yShift'], duration_plotting, frameRate, 'Motion Correction Shifts', 10, 61)
    %
    %     if counter_frameNum > 15
    %         plotUpdatedOutput4([nanmean(logger.motionCorrection.MC_correlation(counter_frameNum-15:counter_frameNum,:),1)], duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 62)
    %     end
    %
    %     if mod(counter_frameNum,60) == 0 && counter_frameNum > 150
    %         plotUpdatedOutput5([nanmean(logger.motionCorrection.MC_correlation(counter_frameNum-150:counter_frameNum,:),1)],...
    %             duration_session, frameRate, 'Motion Correction Correlation All', 10, 15)
    %     end
    %     if mod(counter_frameNum,11) == 0
    % %     if mod(counter_frameNum,1) == 0
    %         plotUpdatedMotionCorrectionImages(baselineStuff.motionCorrectionRefImages.img_MC_reference, img_MC_moving_rollingAvg , img_ROI_corrected, 'Motion Correction')
    %     end
    
    % MC stuff
    plotUpdatedOutput3([xShift'], duration_plotting, frameRate, 'Motion Correction Shifts', 10, 3)
    plotUpdatedOutput4([yShift'], duration_plotting, frameRate, 'Motion Correction Shifts', 10, 3)

%     if counter_frameNum > 15
        %     if counter_frameNum > 1
        plotUpdatedOutput5(MC_corr,...
            duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 3)
%     end

end
%% Teensy Output calculations
if CE_experimentRunning
    if frequencyOverride
        outputVoltageToTeensy = convert_cursor_to_voltage(threshold_value, range_cursorSound, voltage_at_threshold);
    else
        outputVoltageToTeensy = convert_cursor_to_voltage(cursor, range_cursorSound, voltage_at_threshold);
    end
    
%     freqToOutput = 2500* (mod(counter_frameNum,2)+0.5);
    
%     outputVoltageToTeensy = teensyFrequencyToVoltageTransform(freqToOutput , range_freqOutput);

    source.hSI.task_cursorCurrentPos.writeAnalogData(outputVoltageToTeensy);
%     source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(1);
end
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
    logger.trial(counter_frameNum,19) = counter_ITI_successful;
    logger.trial(counter_frameNum,18) = ET_waitForBaseline;
    logger.trial(counter_frameNum,19) = CE_waitForBaseline;
    logger.trial(counter_frameNum,20) = ET_timeout;
    logger.trial(counter_frameNum,21) = CE_timeout;
    logger.trial(counter_frameNum,22) = counter_timeout;
    logger.trial(counter_frameNum,23) = CE_buildingUpStats;
    logger.trial(counter_frameNum,24) = CE_experimentRunning;
    
    logger.decoder.outputs(counter_frameNum,1) = cursor;
%     logger.decoder.outputs(counter_frameNum,2) = freqToOutput;
    logger.decoder.outputs(counter_frameNum,3) = outputVoltageToTeensy;
    logger.decoder.rawVals(counter_frameNum,:) = vals;
    logger.decoder.dFoF(counter_frameNum,:) = dFoF;
    
    %     logger.motionCorrection.xShift(counter_frameNum,:) = xShift;
    %     logger.motionCorrection.yShift(counter_frameNum,:) = yShift;
    %     logger.motionCorrection.MC_correlation(counter_frameNum,:) = MC_corr;
end

%% End Session
if counter_frameNum >= duration_session
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
        dFoF = 0;
        
        NumOfRewardsAcquired = 0;
        
        %         clear logger
        logger.trial = NaN(duration_session,27);
        logger.decoder.outputs = NaN(duration_session,3);
        logger.decoder.rawVals = NaN(duration_session,numCells);
        logger.decoder.dFoF = NaN(duration_session,numCells);
        logger.motionCorrection.xShift = (NaN(duration_session,numCells));
        logger.motionCorrection.yShift = (NaN(duration_session,numCells));
        logger.motionCorrection.MC_correlation = (NaN(duration_session,numCells));
        
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
        loggerNames.trial{23} = 'CE_buildingUpStats';
        loggerNames.trial{24} = 'CE_experimentRunning';
        
        loggerNames.decoder.outputs{1} = 'cursor';
        loggerNames.decoder.outputs{2} = 'freqToOutput';
        loggerNames.decoder.outputs{3} = 'outputVoltageToTeensy';
        
        
        %         saveParams(directory)
    end
    function endSession
        counter_frameNum = NaN;
        CE_experimentRunning = 0;
        
        %         save([directory , '\logger.mat'], 'logger')
        %         saveParams(directory)
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
        
        %         setSoundVolumeTeensy(0);
        source.hSI.task_cursorAmplitude.writeDigitalData(soundVolume);
        source.hSI.task_goalAmplitude.writeDigitalData(0);        
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
        expParams.numFramesToAvgForMotionCorr = numFramesToAvgForMotionCorr;
        
        expParams.loggerNames = loggerNames;
        
        save([directory , '\expParams.mat'], 'expParams')
        %         save([directory , '\motionCorrectionRefImages.mat'], 'motionCorrectionRefImages')
    end
end

%%
function refIm_crop_conjFFT_shift = make_fft_for_MC(refIm)
    refIm_crop = crop_image_for_fft(refIm);
    
    refIm_crop_conjFFT = conj(fft2(refIm_crop));
    refIm_crop_conjFFT_shift = fftshift(refIm_crop_conjFFT);
    
%     disp(size(refIm_crop_conjFFT_shift))


    % size(refIm_crop_conjFFT_shift,1);
    % if mod(size(refIm_crop_conjFFT_shift,1) , 2) == 0
    %     disp('RH WARNING: y length of refIm_crop_conjFFT_shift is even. Something is very wrong')
    % end
    % if mod(size(refIm_crop_conjFFT_shift,2) , 2) == 0
    %     disp('RH WARNING: x length of refIm_crop_conjFFT_shift is even. Something is very wrong')
    % end

%     refIm_crop_conjFFT_shift_centerIdx = ceil(size(refIm_crop_conjFFT_shift)/2);
%     disp(size(refIm_crop_conjFFT_shift_centerIdx))
end
function refIm_crop = crop_image_for_fft(refIm)
    
    refIm = single(refIm);
%     disp(size(refIm))
    % crop_factor = 5;
    crop_size = 256; % MAKE A POWER OF 2! eg 32,64,128,256,512


    length_x = size(refIm,2);
    length_y = size(refIm,1);
    middle_x = size(refIm,2)/2;
    middle_y = size(refIm,1)/2;

    % indRange_y_crop = [round(middle_y - length_y/crop_factor) , round(middle_y + length_y/crop_factor) ];
    % indRange_x_crop = [round(middle_x - length_y/crop_factor) , round(middle_x + length_y/crop_factor) ];

    indRange_y_crop = [round(middle_y - (crop_size/2-1)) , round(middle_y + (crop_size/2)) ];
    indRange_x_crop = [round(middle_x - (crop_size/2-1)) , round(middle_x + (crop_size/2)) ];
    
%     disp(indRange_y_crop)
    refIm_crop = refIm(indRange_y_crop(1) : indRange_y_crop(2) , indRange_x_crop(1) : indRange_x_crop(2)) ;
end