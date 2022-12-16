function [logger , logger_valsROIs  , NumOfRewardsAcquired] =...
    userFunction_BMIv11_withZ(source, event, varargin)
%% Variable stuff
tic
global counter_frameNum counter_trialIdx counter_CS_threshold counter_timeout counter_rewardToneHold...
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
    x_idx_raw y_idx_raw lam_vals...

persistent baselineStuff trialStuff 

%% == USER SETTINGS ==
% SETTINGS: General
frameRate                   = 30;
duration_plotting           = 30 * frameRate; % ADJUSTABLE: change number value (in seconds). Duration of x axis in plots
duration_session            = 30*60*60; % ADJUSTABLE: change number value (in seconds/minutes)
win_smooth                  = 4; % smoothing window (in frames)
show_MC_ref_images          = 0;
threshold_value             = 1.5;

% SETTINGS: Motion correction
numFramesToAvgForMotionCorr = 60;
numFramesToMedForZCorr      = 30*4;
zCorrFrameInterval          = 15; 
interval_z_correction       = 60*frameRate;
max_z_delta                 = 0.5;

% SETTINGS: Cursor
range_cursor = [-threshold_value , threshold_value *1.5];
% range_cursor = [-threshold_value threshold_value];
range_freqOutput = [1000 18000]; % this is set in the teensy code (only here for logging purposes)
voltage_at_threshold = 3.1; % this will be the maximum output voltage ([0:voltage_at_threshold])

% SETTINGS: Reward
reward_duration = 64; % in ms calibrated to 3.6 uL/reward 
reward_delay = 0; % in ms
% reward_delay = 5; % in ms
LED_duration = 0.2; % in s
LED_ramp_duration = 0.1; % in s

% SETTINGS: Mode
% mode = 'baseline';
mode = 'BMI';

% SETTINGS: Trials
% below in unit seconds
duration_trial          = 20;
duration_timeout        = 4;
duration_threshold      = 0.066;
duration_ITI    = 3;
duration_rewardDelivery = 1.00; % before it was .2 10/10/22: Controls how long the reward tone is
threshold_quiescence    = 0;
duration_quiescenceHold = 0.5;  % in seconds

% SETTINGS: Rolling stats
duration_rollingStats       = round(frameRate * 60 * 15);
subSampleFactor_runningVals = 1;
numSamples_rollingStats = round(duration_rollingStats/subSampleFactor_runningVals);
duration_buildingUpStats    = round(frameRate * 60 * 2);

%% IMPORT FILES
currentImage = source.hSI.hDisplay.lastFrame{1};
hash_image = simple_image_hash(currentImage);

% Should be TODAY's directory
directory = 'D:\RH_local\data\BMI_cage_g2F\mouse_g2FB\20221118\analysis_data';
maskPref = 1;                                                                     
borderOuter = 20;                                                                
borderInner = 10;      

if ~isstruct(baselineStuff) && strcmp(mode, 'BMI')
    path_baselineStuff = [directory , '\baselineStuff.mat'];
    load(path_baselineStuff);
    disp(['LOADED baselineStuff from:  ' , path_baselineStuff])
    x_idx_raw = int32(baselineStuff.ROIs.spatial_footprints_tall_warped(:,2));
    y_idx_raw = int32(baselineStuff.ROIs.spatial_footprints_tall_warped(:,3));
    lam_vals = single(baselineStuff.ROIs.spatial_footprints_tall_warped(:,4));
end
if ~isstruct(trialStuff)
    path_trialStuff = [directory , '\trialStuff.mat'];
    load(path_trialStuff);
    disp(['LOADED trialStuff from:  ' , path_trialStuff])
end

% loadedCheck_registrationImage = []
if ~exist('loadedCheck_registrationImage') | isempty(loadedCheck_registrationImage) | loadedCheck_registrationImage ~= 1 | isempty(registrationImage) | isempty(refIm_crop_conjFFT_shift) | isempty(indRange_y_Crop)
    if strcmp(mode, 'BMI')
        type_stack = 'stack_warped';
    elseif strcmp(mode, 'baseline')
        type_stack = 'stack_sparse';
    end
%     type_stack = 'stack_warped';
    tmp = load([directory , '\', type_stack, '.mat']);
    disp(['LOADED zstack from', directory , '\', type_stack, '.mat'])
    registrationImage = eval(['tmp.', type_stack, '.stack_avg']);
    referenceDiffs = eval(['tmp.', type_stack,'.step_size_um']);
    loadedCheck_registrationImage=1;
    

%     clear refIm_crop_conjFFT_shift
    for ii = 1:size(registrationImage,1)
        [refIm_crop_conjFFT_shift(ii,:,:), ~, indRange_y_Crop, indRange_x_Crop] = make_fft_for_MC(squeeze(registrationImage(ii,:,:)));
        refIm_crop(ii,:,:) = registrationImage(ii, indRange_y_Crop(1):indRange_y_Crop(2), indRange_x_Crop(1):indRange_x_Crop(2));
        if maskPref
            refIm_crop_conjFFT_shift_masked(ii,:,:) = maskImage(squeeze(refIm_crop_conjFFT_shift(ii,:,:)), borderOuter, borderInner);
        end
    end
    
    registrationImage = registrationImage(2:4,:,:);
    refIm_crop = refIm_crop(2:4,:,:);
    refIm_crop_conjFFT_shift = refIm_crop_conjFFT_shift(2:4,:,:);
    refIm_crop_conjFFT_shift_masked = refIm_crop_conjFFT_shift_masked(2:4,:,:);
    
    disp('Loaded z-stack')
end

if strcmp(mode, 'BMI')
    numCells = baselineStuff.ROIs.num_cells;
elseif strcmp(mode, 'baseline')
    numCells = 1;
end

%% == Session Starting & counting ==

% == Start Session ==
if isempty(counter_frameNum)
    disp('hi. NEW SESSION STARTED')
    startSession
end
% ======== COMMENT THIS IN/OUT TO START SESSION =======
% startSession
% =====================================================

% counter_frameNum = counter_frameNum + 1;
counter_frameNum = source.hSI.hStackManager.framesDone;

if counter_frameNum == 1
    disp('frameNum = 1')
end

% endSession

% %% SAVING
% saveParams(directory)

% saveParams('F:\RH_Local\Rich data\scanimage data\20191110\mouse 10.13B\expParams.mat')

%% == MOTION CORRECTION ==
% Make a cropped version of the current image for use in motion correction
img_MC_moving = currentImage(indRange_y_Crop(1):indRange_y_Crop(2), indRange_x_Crop(1):indRange_x_Crop(2));

% Track frames for fast 2D motion correction
if ~isa(img_MC_moving_rolling, 'uint16') | size(img_MC_moving_rolling,3) ~= numFramesToAvgForMotionCorr
    img_MC_moving_rolling = zeros([size(img_MC_moving) , numFramesToAvgForMotionCorr], 'uint16');
    disp('making new img_MC_moving_rolling')
end
if counter_frameNum >= 0
    img_MC_moving_rolling(:,:,mod(counter_frameNum , numFramesToAvgForMotionCorr)+1) = img_MC_moving;
end
img_MC_moving_rollingAvg = single(mean(img_MC_moving_rolling,3));


% Track frames for slow Z-axis motion correction
if ~isa(img_MC_moving_rolling_z, 'uint16') | size(img_MC_moving_rolling_z,3) ~= numFramesToMedForZCorr
    img_MC_moving_rolling_z = zeros([size(img_MC_moving) , numFramesToMedForZCorr], 'uint16');
end
if (counter_frameNum >= 0) && (mod(counter_frameNum, zCorrFrameInterval) == 0)
    img_MC_moving_rolling_z(:,:,mod(counter_frameNum/zCorrFrameInterval ,numFramesToMedForZCorr)+1) = img_MC_moving;
end

if ~CE_trial
    [delta, frame_corrs, xShifts, yShifts] = calculate_z_position(img_MC_moving_rolling_z, registrationImage, refIm_crop_conjFFT_shift_masked, referenceDiffs,...
        maskPref, borderOuter, borderInner);
else
    delta=0;
    frame_corrs = [0,0,0];
end

if strcmp(mode, 'BMI')
    [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , [] , baselineStuff.MC.meanImForMC_crop_conjFFT_shift, maskPref, borderOuter, borderInner);
elseif strcmp(mode, 'baseline')
    [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , [] , squeeze(refIm_crop_conjFFT_shift_masked(2,:,:)), maskPref, borderOuter, borderInner);
end
MC_corr = max(cxx);

% delta=0;
% frame_corrs=[0,0,0];
% 
% xShift = 0;
% yShift = 0;
% MC_corr = 0;


if abs(xShift) >60
    xShift = 0;
end
if abs(yShift) >60
    yShift = 0;
end
%% == EXTRACT DECODER Product of Current Image and Decoder Template ==
%     tic
if strcmp(mode, 'BMI')
    x_idx     =  x_idx_raw + xShift;
    y_idx     =  y_idx_raw + yShift;

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
    TA_CF_lam = tall_currentImage .* lam_vals .* idx_safe;
    TA_CF_lam_reshape = reshape(TA_CF_lam , baselineStuff.ROIs.cell_size_max , numCells);
    vals_neurons = nansum( TA_CF_lam_reshape , 1 );
elseif strcmp(mode, 'baseline')
    vals_neurons = NaN;
end
% toc

%% == ROLLING STATS ==
if CE_experimentRunning
    if strcmp(mode, 'BMI')
        if mod(counter_frameNum-1 , subSampleFactor_runningVals) == 0
            next_idx = mod(counter_runningVals-1 , numSamples_rollingStats)+1;
            vals_old = runningVals(next_idx , :);
            runningVals(next_idx,:) = vals_neurons;
            [rolling_var_obj_cells , F_mean , F_var] = rolling_var_obj_cells.step(counter_frameNum , runningVals(next_idx,:) , vals_old);
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
        [rolling_var_obj_cursor , cursor_mean , cursor_var] = rolling_var_obj_cursor.step(counter_frameNum , running_cursor_raw(next_idx) , vals_old);
        counter_runningCursor = counter_runningCursor+1;

        if counter_frameNum >= win_smooth
    %         cursor_brain = mean(logger.decoder(counter_frameNum-(win_smooth-1):counter_frameNum,2));
            cursor_brain = mean((logger.decoder(counter_frameNum-(win_smooth-1):counter_frameNum,2)-cursor_mean)./sqrt(cursor_var));
        else
            cursor_brain = cursor_brain_raw;
        end
    elseif strcmp(mode, 'baseline')
        cursor_brain = -0.1;
    end
%     toc
%% Check for overlap of ROIs and image (VERY SLOW)

    if (CE_rewardDelivery && counter_rewardDelivery<2) && strcmp(mode, 'BMI')
        % % y_tmp = reshape(y_idx, baselineStuff.ROIs.cell_size_max , numCells);
        % % size(y_tmp)
        % CI = zeros(size(currentImage,1), size(currentImage,2));
        % CI(sub2ind(size(currentImage), y_idx , x_idx)) = currentImage(sub2ind(size(currentImage), y_idx , x_idx));

        WI = zeros(size(currentImage,1), size(currentImage,2));
        % size(baselineStuff.ROIs.cellWeightings)
        WI(sub2ind(size(currentImage), y_idx , x_idx)) = repmat(baselineStuff.ROIs.cellWeightings .* F_zscore, 230, 1);
        % WI(sub2ind(size(currentImage), y_idx , x_idx)) = repmat(baselineStuff.ROIs.cellWeightings, 230, 1);
        % 
        LI = zeros(size(currentImage,1), size(currentImage,2));
        LI(sub2ind(size(currentImage), y_idx , x_idx)) = baselineStuff.ROIs.spatial_footprints_tall_warped(:,4);
        % 
        % % CWI = CI .* WI;
        % LCWI = CI .* LI .* WI;
        LWI = LI .* WI;
        plotUpdatedImagesc(LWI , [-0.2,1], 'test')
    end
%     toc
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
    
    CS_quiescence = algorithm_quiescence(cursor_output, threshold_quiescence);
    CS_threshold = algorithm_thresholdState(cursor_output, threshold_value);
    
    %%  ===== TRIAL STRUCTURE =====
    % CE = current epoch
    % ET = epoch transition signal
    % CS = current state
    
    % START BUILDING UP STATS
    if counter_frameNum == 1
        CE_buildingUpStats = 1;
    end
    % BUILDING UP STATS
    if CE_buildingUpStats
        source.hSI.task_cursorAmplitude.writeDigitalData(0);
        source.hSI.task_goalAmplitude.writeDigitalData(0);
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
        %     giveReward2(1, 1, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
            giveReward3(source, 1, 0, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
        %         giveReward3(source, 1, 0, 500, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
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
        source.hSI.task_cursorAmplitude.writeDigitalData(soundVolume);
        source.hSI.task_goalAmplitude.writeDigitalData(0);
        
        if counter_frameNum > (counter_last_z_correction + interval_z_correction)

            if delta ~=0
                clampedDelta = sign(delta) * min(abs(delta), max_z_delta);
                disp(['moving fast Z by one step: ', num2str(clampedDelta)]) %num2str(delta)])
                currentPosition = moveFastZ(source, [], clampedDelta, [], [20,380]);
                delta_moved = clampedDelta;
                counter_last_z_correction = counter_frameNum;
%             elseif (max(abs(frame_corrs(1) - frame_corrs(2)), abs(frame_corrs(3) - frame_corrs(2)))>abs(frame_corrs(1) - frame_corrs(3)))
            elseif abs(frame_corrs(3) - frame_corrs(1)) > abs(max([frame_corrs(1), frame_corrs(3)] - frame_corrs(2)))
                clampedDelta = sign(frame_corrs(1) - frame_corrs(3)) * max_z_delta;
                disp(['moving fast Z by one step: ', num2str(clampedDelta)]) %num2str(delta)])
                currentPosition = moveFastZ(source, [], clampedDelta, [], [20,380]);
                delta_moved = clampedDelta;
                counter_last_z_correction = counter_frameNum;
            end
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
%     toc
end
%% Teensy Output calculations

if CE_experimentRunning
    if frequencyOverride
        voltage_cursorCurrentPos = convert_cursor_to_voltage(threshold_value , range_cursor , voltage_at_threshold);
    else
        voltage_cursorCurrentPos = convert_cursor_to_voltage(cursor_output , range_cursor, voltage_at_threshold);
    end
%     voltage_cursorCurrentPos = (mod(counter_frameNum,2)+0.5);
    source.hSI.task_cursorCurrentPos.writeAnalogData(double(voltage_cursorCurrentPos));

    freqToOutput = convert_voltage_to_frequency(voltage_cursorCurrentPos , 3.3 , range_freqOutput); % for logging purposes only. function should mimic (exactly) the voltage to frequency transformation on teensy
end
% toc

% save([directory , '\logger.mat'], 'logger')
% saveParams(directory)
% disp(['Logger & Params Saved: frameCounter = ' num2str(counter_frameNum)]);
%% Plotting

if counter_frameNum>1
    %     plotUpdatedOutput([cursor , dFoF.*scale_factors], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 20)
    
%         plotUpdatedOutput([cursor , dFoF], duration_plotting, frameRate, 'Cursor , E1 , E2', 10, 59)
    
    plotUpdatedOutput2([CE_waitForBaseline*0.1, CE_trial*0.2,...
        CE_rewardDelivery*0.3, CE_timeout*0.4, CE_buildingUpStats*0.5, fakeFeedback_inUse*0.6]...
        , duration_plotting, frameRate, 'Rewards', 10, 22, ['# Rewards: ' , num2str(NumOfRewardsAcquired) , ' ; # Timeouts: ' , num2str(NumOfTimeouts)])
    
    plotUpdatedOutput3([xShift' yShift'], duration_plotting, frameRate, 'Motion Correction Shifts', 10, 11)
    
    if counter_frameNum > 25
        %     if counter_frameNum > 1
        plotUpdatedOutput4([nanmean(logger.motionCorrection(counter_frameNum-15:counter_frameNum,3),1)], duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 12)
        %         plotUpdatedOutput4([nanmean(logger.motionCorrection.MC_correlation(counter_frameNum-15:counter_frameNum,:),1)], duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 12)
    end
    
    %     plotUpdatedOutput4(dFoF(1:10) , duration_plotting , frameRate , 'dFoF' , 20, 50)
    plotUpdatedOutput6(cursor_output , duration_plotting , frameRate , 'cursor' , 10,3)
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
    
    plotUpdatedOutput7(frame_corrs,...
        duration_plotting, frameRate, 'Z Frame Correlations', 10, 10)
    
end
% toc
%% DATA LOGGING
if ~isnan(counter_frameNum)
    logger_valsROIs(counter_frameNum,:) = vals_neurons; %already done above
    
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
    logger.timeSeries(counter_frameNum,27) = hash_image;
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
    
    logger.motionCorrection(counter_frameNum,1) = xShift;
    logger.motionCorrection(counter_frameNum,2) = yShift;
    logger.motionCorrection(counter_frameNum,3) = MC_corr;
    logger.motionCorrection(counter_frameNum,4) = source.hSI.hFastZ.currentFastZs{1}.targetPosition;
    logger.motionCorrection(counter_frameNum,5) = delta_moved;
    logger.motionCorrection(counter_frameNum,6:8) = frame_corrs; 

%     toc
end

%% End Session

if  counter_frameNum == round(duration_session * 0.90)
%     source.hSI.task_cursorAmplitude.writeDigitalData(0);
%     source.hSI.task_goalAmplitude.writeDigitalData(0);
    saveLogger(directory);
    saveParams(directory);
%     source.hSI.task_cursorAmplitude.writeDigitalData(1);
%     source.hSI.task_goalAmplitude.writeDigitalData(1);
end
% counter_frameNum
if  counter_frameNum == round(duration_session * 0.98)
    endSession
end
% source.hSI.task_cursorAmplitude.writeDigitalData(0);
% source.hSI.task_goalAmplitude.writeDigitalData(0);

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
        loadedCheck_registrationImage = [];  % set in order to force import of stack.mat
        
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
        CE_waitForBaseline = 0;
        ET_timeout = 0;
        CE_timeout = 0;
        counter_timeout = 0;
        counter_last_z_correction = 0;
        
        counter_frameNum = 0;
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
        loggerNames.motionCorrection{4} = 'current_z_position';
        loggerNames.motionCorrection{5} = 'deltaMoved';
        loggerNames.motionCorrection{6} = 'z_correlation_1';
        loggerNames.motionCorrection{7} = 'z_correlation_2';
        loggerNames.motionCorrection{8} = 'z_correlation_3';
        
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

        logger_valsROIs = NaN(duration_session , numCells);
        runningVals = NaN(numSamples_rollingStats , numCells);
        running_cursor_raw = NaN(numSamples_rollingStats , 1);
        
        rolling_var_obj_cells = rolling_var_and_mean();
        rolling_var_obj_cells = rolling_var_obj_cells.set_key_properties(size(runningVals) , duration_rollingStats);
        rolling_var_obj_cursor = rolling_var_and_mean();
        rolling_var_obj_cursor = rolling_var_obj_cursor.set_key_properties([1,1] , duration_rollingStats);
        
        counter_runningVals = 1;
        counter_runningCursor = 1;

        saveParams(directory)
    end

    function endSession
        disp('SESSION OVER')
        counter_frameNum = NaN;
        CE_experimentRunning = 0;
        
        saveLogger(directory)
        saveParams(directory)
        disp('=== Loggers and expParams saved ===')
        
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
        CE_waitForBaseline = 0;
        ET_timeout = 0;
        CE_timeout = 0;
        counter_timeout = 0;
        
        %         counter_frameNum = 0;
        CE_buildingUpStats = 0;
        %         CE_experimentRunning = 0;
        cursor_output = 0;
        
        %         setSoundVolumeTeensy(0);
        source.hSI.task_cursorAmplitude.writeDigitalData(0);
        source.hSI.task_goalAmplitude.writeDigitalData(0);
        
    end

    function saveLogger(directory)
        disp("Saving logger.mat...")
        save([directory , '\logger.mat'], 'logger');
        save([directory , '\logger_valsROIs.mat'], 'logger_valsROIs');
    end

    function saveParams(directory)
        expParams.frameRate = frameRate;
        expParams.duration_session = duration_session;
        expParams.duration_trial = duration_trial;
        expParams.win_smooth = win_smooth;
        expParams.duration_threshold = duration_threshold;
        expParams.threshold_value = threshold_value;
        expParams.range_cursor = range_cursor;
        expParams.range_freqOutput = range_freqOutput;
        expParams.voltage_at_threshold = voltage_at_threshold;
        expParams.duration_timeout = duration_timeout;
        expParams.numCells = numCells;
        expParams.directory = directory;
        expParams.duration_rollingStats = duration_rollingStats;
        expParams.duration_rollingStats = duration_buildingUpStats;
        expParams.subSampleFactor_runningVals = subSampleFactor_runningVals;
        expParams.threshold_quiescence = threshold_quiescence;
        expParams.duration_ITI = duration_ITI;
        expParams.duration_rewardDelivery = duration_rewardDelivery;
        expParams.duration_quiescenceHold = duration_quiescenceHold;
        expParams.reward_duration = reward_duration; % in ms
        expParams.reward_delay = reward_delay;
        expParams.LED_duration = LED_duration;
        expParams.LED_ramp_duration = LED_ramp_duration;
        expParams.numFramesToAvgForMotionCorr = numFramesToAvgForMotionCorr;
        
        expParams.image_hash_function = 'hash = sum(sum(image,1).^2)';
        
        expParams.loggerNames = loggerNames;
        
        expParams.baselineStuff = baselineStuff;
        
        save([directory , '\expParams.mat'], 'expParams')
        %         save([directory , '\motionCorrectionRefImages.mat'], 'motionCorrectionRefImages')
    end

    function [refIm_crop_conjFFT_shift, refIm_crop, indRange_y_crop, indRange_x_crop] = make_fft_for_MC(refIm)
        refIm = single(refIm);
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
    
        refIm_crop = refIm(indRange_y_crop(1) : indRange_y_crop(2) , indRange_x_crop(1) : indRange_x_crop(2)) ;

        refIm_crop_conjFFT = conj(fft2(refIm_crop));
        refIm_crop_conjFFT_shift = fftshift(refIm_crop_conjFFT);

        % size(refIm_crop_conjFFT_shift,1);
        % if mod(size(refIm_crop_conjFFT_shift,1) , 2) == 0
        %     disp('RH WARNING: y length of refIm_crop_conjFFT_shift is even. Something is very wrong')
        % end
        % if mod(size(refIm_crop_conjFFT_shift,2) , 2) == 0
        %     disp('RH WARNING: x length of refIm_crop_conjFFT_shift is even. Something is very wrong')
        % end

%         refIm_crop_conjFFT_shift_centerIdx = ceil(size(refIm_crop_conjFFT_shift)/2);
    end

    function [delta,frame_corrs, xShifts, yShifts] = calculate_z_position(img_MC_moving_rolling_z, registrationImage, refIm_crop_conjFFT_shift, referenceDiffs, maskPref, borderOuter, borderInner)
        image_toUse = mean(img_MC_moving_rolling_z, 3);
        [delta, frame_corrs, xShifts, yShifts] = zCorrection(image_toUse, registrationImage, ...
            refIm_crop_conjFFT_shift, referenceDiffs, maskPref, borderOuter, borderInner);
        
    end

    function currentPosition = moveFastZ(source, evt, delta, position, range_position)
        
        if ~exist('range_position')
            range_position = [0, 200];
        end
        
        fastZDevice = source.hSI.hFastZ.currentFastZs{1};
        %Select the FastZ device (you likely have just one, so index at 1)

        currentPosition = fastZDevice.targetPosition;
        
        if ~exist('position')
            newPosition = currentPosition + delta;
        elseif isempty(position)
            newPosition = currentPosition + delta;
        else
            newPosition = position;
        end
        % scalar finite number indicating depth within the lower and upper travel bounds set by the user.
        
        if range_position(1) > newPosition | range_position(2) < newPosition
            error(['RH ERROR: newPosition if out of range. Range: ', range_position, ' Attempted position: ', newPosition])
        end
            
%         force = true;
        % do the move even if it is a grab or loop acquisition. Don't try using this with a stack acquisition.

    %     source.hSI.hFastZ.move(fastZDevice, position,  force);
    
        source.hSI.hFastZ.move(fastZDevice, newPosition);

    end
%     toc
end