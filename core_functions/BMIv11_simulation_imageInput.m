function [logger_output , loggerNames_output , logger_valsROIs_output  , NumOfRewardsAcquired] =...
    BMIv11_simulation_imageInput(currentImage, frameNum , baselineStuff , trialStuff, zstack , last_frameNum , threshold_value , threshold_quiescence, num_frames_total)
%% Variable stuff
tic
global counter_trialIdx counter_CS_threshold counter_timeout counter_rewardToneHold...
    counter_rewardDelivery counter_ITI_withZ...
    CE_buildingUpStats CE_experimentRunning CE_waitForBaseline CE_trial CE_timeout CE_rewardToneHold...
    CE_rewardDelivery CE_ITI_withZ...
    ET_waitForBaseline ET_trialStart ET_timeout ET_rewardToneHold ET_rewardDelivery ET_ITI_withZ...
    frequencyOverride...
    NumOfRewardsAcquired NumOfTimeouts trialNum soundVolume...
    logger loggerNames logger_valsROIs...
    runningVals running_cursor_raw counter_runningVals counter_runningCursor...
    rolling_var_obj_cells rolling_var_obj_cursor loadedCheck_registrationImage...
    counter_last_z_correction...
    counter_quiescenceHold...
    pe shifter...
    rolling_z_mean_obj...
    params data rois sm...

% persistent baselineStuff
%% IMPORT DATA
if frameNum == 1
    params = struct();
    data = struct();
    rois = struct();
    sm = struct();
end

frameNum = frameNum;
% frameNum = source.hSI.hStackManager.framesDone;
% data.currentImage = source.hSI.hDisplay.lastFrame{1};
data.currentImage = currentImage;
data.currentImage_gpu = gpuArray(data.currentImage);
data.hash_image = simple_image_hash(data.currentImage);  %% slower on gpu
% hash_image = gather(simple_image_hash(data.currentImage_gpu));

%% == USER SETTINGS ==
if frameNum == 1
    % SETTINGS: General
    
    % SETTINGS: TIMING
    params.timing.frameRate          = 30;
    params.timing.duration_plotting  = 30 * params.timing.frameRate; % ADJUSTABLE: change number value (in seconds). Duration of x axis in plots
    params.timing.duration_session   = num_frames_total; % ADJUSTABLE: change number value (in seconds/minutes)
    
    % SETTINGS: Motion correction
    params.MC.numFrames_avgWin_zCorr      = 30*2;
    params.MC.intervalFrames_zCorr        = 5; 
    params.MC.min_interval_z_correction   = 20*params.timing.frameRate;
    params.MC.max_delta_z_correction      = 1;
    params.MC.bandpass_freqs              = [1/64, 1/4];
    params.MC.bandpass_orderButter        = 3;
    params.MC.device                      = 'cuda';
    params.MC.frame_shape_yx              = int64([512,512]);

    % SETTINGS: Cursor
    params.cursor.threshold_value      = 1.7;
    params.cursor.win_smooth_cursor    = 3; % smoothing window (in frames)
    params.cursor.bounds_cursor        = [-params.cursor.threshold_value , params.cursor.threshold_value *1.5];
    params.cursor.range_freqOutput     = [1000 18000]; % this is set in the teensy code (only here for logging purposes)
    params.cursor.voltage_at_threshold = 3.1; % this will be the maximum output voltage ([0:voltage_at_threshold])

    % SETTINGS: Reward
    params.reward.reward_duration = 64; % in ms calibrated to 3.6 uL/reward 
    params.reward.reward_delay = 0; % in ms
    
    % SETTINGS: Mode
%     params.mode = 'baseline';
    params.mode = 'BMI';

    % SETTINGS: Trials
    % below in unit seconds
    params.trial.duration_trial          = 20;
    params.trial.duration_timeout        = 4;
    params.trial.duration_threshold      = 0.066;
    params.trial.duration_ITI            = 3;
    params.trial.duration_rewardDelivery = 1.00; % before it was .2 10/10/22: Controls how long the reward tone is
    params.trial.threshold_quiescence    = 0;
    params.trial.duration_quiescenceHold = 0.5;  % in seconds
    params.trial.duration_buildingUpStats    = round(params.timing.frameRate * 60 * 1);

    % SETTINGS: Rolling stats
    params.rollingStats.duration_rolling       = round(params.timing.frameRate * 60 * 15);
    
end

% % SETTINGS: General
% frameRate                   = 30;
% duration_plotting           = 30 * frameRate; % ADJUSTABLE: change number value (in seconds). Duration of x axis in plots
% duration_session            = num_frames_total; % ADJUSTABLE: change number value (in seconds/minutes)
% win_smooth_cursor           = 3; % smoothing window (in frames)
% % threshold_value             = 1.7;
% 
% % SETTINGS: Motion correction
% params.MC.numFrames_avgWin_zCorr      = 30*2;
% params.MC.intervalFrames_zCorr          = 5; 
% interval_z_correction       = 20*frameRate;
% max_z_delta                 = 1;
% frame_shape_yx           = int64([512,1024]);
% 
% % SETTINGS: Cursor
% range_cursor = [-threshold_value , threshold_value *1.5];
% % range_cursor = [-threshold_value threshold_value];
% range_freqOutput = [1000 18000]; % this is set in the teensy code (only here for logging purposes)
% voltage_at_threshold = 3.1; % this will be the maximum output voltage ([0:voltage_at_threshold])
% 
% % SETTINGS: Reward
% reward_duration = 64; % in ms calibrated to 3.6 uL/reward 
% reward_delay = 0; % in ms
% % reward_delay = 5; % in ms
% LED_duration = 0.2; % in s
% LED_ramp_duration = 0.1; % in s
% 
% % SETTINGS: Mode
% % mode = 'baseline';
% mode = 'BMI';
% 
% % SETTINGS: Trials
% % below in unit seconds
% duration_trial          = 20;
% duration_timeout        = 4;
% duration_threshold      = 0.066;
% duration_ITI    = 3;
% duration_rewardDelivery = 1.00; % before it was .2 10/10/22: Controls how long the reward tone is
% threshold_quiescence    = 0;
% duration_quiescenceHold = 0.5;  % in seconds
% 
% % SETTINGS: Rolling stats
% duration_rollingStats       = round(frameRate * 60 * 15);
% numSamples_rollingStats = round(duration_rollingStats);
% duration_buildingUpStats    = round(frameRate * 60 * 1);

%% INITIALIZE EXPERIMENT

if frameNum == 1 && strcmp(params.mode, 'BMI')
    %%% Motion correction python code prep
    try
        pe = pyenv('Version', 'C:\ProgramData\Miniconda\envs\matlab_env\python');  %% prepare python environment
    catch
        disp('failed to initalize Python environment. The environment may already by loaded')
    end
    py.importlib.import_module('bph.motion_correction');

    im = baselineStuff.MC.meanIm;
    s_y = floor((size(im,1)-params.MC.frame_shape_yx(1))/2) + 1;
    s_x = floor((size(im,2)-params.MC.frame_shape_yx(2))/2) + 1;
    data.MC.idx_im_MC_crop_y = s_y:s_y+params.MC.frame_shape_yx(1)-1;
    data.MC.idx_im_MC_crop_x = s_x:s_x+params.MC.frame_shape_yx(2)-1;
    data.MC.im_refIm_MC_2D = gpuArray(single(im(data.MC.idx_im_MC_crop_y, data.MC.idx_im_MC_crop_x)));

    im_zstack = eval(['zstack', '.stack_avg']);
    im_zstack = single(im_zstack(:, data.MC.idx_im_MC_crop_y, data.MC.idx_im_MC_crop_x));
    data.MC.stepSize_zstack = eval(['zstack','.step_size_um']);
    data.MC.n_slices_zstack = size(im_zstack, 1);
    
    % Initialize the shifter clas
    shifter = py.bph.motion_correction.Shifter_rigid(params.MC.device);
    shifter.make_mask(py.tuple(params.MC.frame_shape_yx), py.tuple(params.MC.bandpass_freqs), params.MC.bandpass_orderButter);
    shifter.preprocess_template_images(gather(single(cat(1, permute(data.MC.im_refIm_MC_2D, [3,1,2]), im_zstack))), py.int(0));

    disp('Loaded z-stack')

    
    rois.x_idx_raw = gpuArray(int32(baselineStuff.ROIs.spatial_footprints_tall_warped(:,2)));
    rois.y_idx_raw = gpuArray(int32(baselineStuff.ROIs.spatial_footprints_tall_warped(:,3)));
    rois.lam_vals = gpuArray(single(baselineStuff.ROIs.spatial_footprints_tall_warped(:,4)));
    rois.lam_vals(isnan(rois.lam_vals)) = 0;

    rois.Fneu_x_idx_raw = gpuArray(int32(baselineStuff.ROIs.Fneu_masks_tall_warped(:,2)));
    rois.Fneu_y_idx_raw = gpuArray(int32(baselineStuff.ROIs.Fneu_masks_tall_warped(:,3)));
    rois.Fneu_lam_vals = gpuArray(single(baselineStuff.ROIs.Fneu_masks_tall_warped(:,4)));
    rois.Fneu_lam_vals(isnan(rois.Fneu_lam_vals)) = 0;
%     rois.Fneu_lam_vals = gpuArray(int16(rois.Fneu_lam_vals));

    data.MC.im_buffer_rolling_z = gpuArray(zeros([size(data.MC.im_refIm_MC_2D) , params.MC.numFrames_avgWin_zCorr], 'int16'));
    data.MC.counter_buffer_rolling_z = 0;

    
end

if strcmp(params.mode, 'BMI')
    rois.numCells = baselineStuff.ROIs.num_cells;
elseif strcmp(params.mode, 'baseline')
    rois.numCells = 1;
end

%% == Session Starting & counting ==

% == Start Session ==
if frameNum == 1
    disp('hi. NEW SESSION STARTED')
    startSession
    disp('frameNum = 1')
end
% ======== COMMENT THIS IN/OUT TO START SESSION ===========================
% startSession
% =========================================================================

%% == MOTION CORRECTION ==
% % FASTER ON CPU THAN GPU
% % Make a cropped version of the current image for use in motion correction
data.MC.img_MC_2d_moving = data.currentImage_gpu(data.MC.idx_im_MC_crop_y(1):data.MC.idx_im_MC_crop_y(end), data.MC.idx_im_MC_crop_x(1):data.MC.idx_im_MC_crop_x(end));

% Track frames for slow Z-axis motion correction
if (frameNum >= 0) && (mod(frameNum, params.MC.intervalFrames_zCorr) == 0)
    data.MC.counter_buffer_rolling_z = data.MC.counter_buffer_rolling_z + 1;
    data.MC.im_buffer_rolling_z_mean = rolling_z_mean_obj.update_mean(rolling_z_mean_obj.idx_new, data.MC.img_MC_2d_moving, data.MC.im_buffer_rolling_z(:,:,mod(data.MC.counter_buffer_rolling_z ,params.MC.numFrames_avgWin_zCorr)+1), rolling_z_mean_obj.win_size, rolling_z_mean_obj.mean_old);

    rolling_z_mean_obj.mean_old = data.MC.im_buffer_rolling_z_mean;
    rolling_z_mean_obj.idx_new = rolling_z_mean_obj.idx_new + 1;

    data.MC.im_buffer_rolling_z(:,:,mod(data.MC.counter_buffer_rolling_z ,params.MC.numFrames_avgWin_zCorr)+1) = data.MC.img_MC_2d_moving;
elseif frameNum < params.MC.intervalFrames_zCorr
    data.MC.im_buffer_rolling_z_mean = data.MC.im_buffer_rolling_z(:,:,1);
end

out = shifter.find_translation_shifts(gather(data.MC.img_MC_2d_moving), py.int(0));  %% 0-indexed
shifts_yx          = int32(out{1}.numpy());
data.MC.yShift     = shifts_yx(1);
data.MC.xShift     = shifts_yx(2);
data.MC.maxCorr_2d = single(out{2}.numpy());

out = shifter.find_translation_shifts(gather(data.MC.im_buffer_rolling_z_mean), py.list(int64([1:data.MC.n_slices_zstack])));  %% 0-indexed
data.MC.maxCorr_z = single(out{2}.numpy());
[maxVal, maxArg] = max(data.MC.maxCorr_z);
data.MC.delta_z = (ceil(data.MC.n_slices_zstack/2)-maxArg) * data.MC.stepSize_zstack; 

% data.MC.xShift = 0;
% data.MC.yShift = 0;
% data.MC.maxCorr_2d = 0;
% data.MC.maxCorr_z = zeros(1,5);
% data.MC.delta_z = 0;

if abs(data.MC.xShift) >60
    data.MC.xShift = 0;
end
if abs(data.MC.yShift) >60
    data.MC.yShift = 0;
end

%% == EXTRACT DECODER Product of Current Image and Decoder Template ==
% FASTER ON GPU
if strcmp(params.mode, 'BMI')
    % New extractor
    x_idx     = rois.x_idx_raw + data.MC.xShift;
    y_idx     = rois.y_idx_raw + data.MC.yShift;

    y_idx = max(min(y_idx, double(params.MC.frame_shape_yx(1))), 1);
    x_idx = max(min(x_idx, double(params.MC.frame_shape_yx(2))), 1);
    
    Fneu_x_idx     =  rois.Fneu_x_idx_raw + data.MC.xShift;
    Fneu_y_idx     =  rois.Fneu_y_idx_raw + data.MC.yShift;

    Fneu_y_idx = max(min(Fneu_y_idx, double(params.MC.frame_shape_yx(1))), 1);
    Fneu_x_idx = max(min(Fneu_x_idx, double(params.MC.frame_shape_yx(2))), 1);

    % Calculate current frame F / Fneu: lam should be normalized to sum=1,
    % then to get F, F=image(pixels) @ lam (its a sum)
    tall_currentImage_F = single(data.currentImage_gpu(sub2ind(size(data.currentImage_gpu), y_idx , x_idx)));
    TA_CF_lam = tall_currentImage_F .* rois.lam_vals;
    TA_CF_lam_reshape = reshape(TA_CF_lam , baselineStuff.ROIs.cell_size_max , rois.numCells);
    
    % Calculate Fneu: 'lam' is specified by s2p as just the pixels because
    % the lambda values are all 1. Fneu = (mean(image(pixels))). So we just
    % pre normalize the lam values and take a sum
    tall_currentImage_Fneu = single(data.currentImage_gpu(sub2ind(size(data.currentImage_gpu), Fneu_y_idx , Fneu_x_idx)));
    TA_CF_Fneu_lam = tall_currentImage_Fneu .* rois.Fneu_lam_vals;
    TA_CF_Fneu_lam_reshape = reshape(TA_CF_Fneu_lam , baselineStuff.ROIs.neuropil_size_max , rois.numCells);
    
%     % Subtract 0.7 * Fneu from F
%     vals_neurons = ones(1,numCells);
%     vals_neurons = nansum( TA_CF_lam_reshape , 1 );
    data.vals_neurons = nansum( TA_CF_lam_reshape , 1 ) - nanmean(TA_CF_Fneu_lam_reshape, 1) * 0.7;
    
    data.vals_neurons = gather(data.vals_neurons);

elseif strcmp(params.mode, 'baseline')
    data.vals_neurons = NaN;
end

%% == ROLLING STATS ==
data.cursor_brain_raw = NaN;
sm.fakeFeedback_inUse = NaN;
if CE_experimentRunning
    if strcmp(params.mode, 'BMI')
        next_idx = mod(counter_runningVals-1 , params.rollingStats.duration_rolling)+1;
        vals_old = runningVals(next_idx , :);
        runningVals(next_idx,:) = data.vals_neurons;
        % 20230126 Now, frameNum follows ScanImage Frames Done
%             [rolling_var_obj_cells , F_mean , F_var] = rolling_var_obj_cells.step(frameNum , runningVals(next_idx,:) , vals_old);
        [rolling_var_obj_cells , F_mean , F_var] = rolling_var_obj_cells.step(runningVals(next_idx,:) , vals_old);
        counter_runningVals = counter_runningVals+1;
        if frameNum == 1
            F_mean = data.vals_neurons;
            F_var = ones(size(F_mean));
        end

        F_std = sqrt(F_var);
        %     F_std = nanstd(runningVals , [] , 1);
        %     F_std = ones(size(vals_smooth));
        F_std(F_std < 0.01) = inf;
        %     F_mean = nanmean(runningVals , 1);
        F_zscore = single((data.vals_neurons-F_mean)./F_std);
        F_zscore(isnan(F_zscore)) = 0;

        cursor_brain_raw = F_zscore * baselineStuff.ROIs.cellWeightings';
        logger.decoder(frameNum,2) = cursor_brain_raw;

        next_idx = mod(counter_runningCursor-1 , params.rollingStats.duration_rolling)+1;
        vals_old = running_cursor_raw(next_idx);
        running_cursor_raw(next_idx) = cursor_brain_raw;
%         [rolling_var_obj_cursor , cursor_mean , cursor_var] = rolling_var_obj_cursor.step(frameNum , running_cursor_raw(next_idx) , vals_old);
        [rolling_var_obj_cursor , cursor_mean , cursor_var] = rolling_var_obj_cursor.step(running_cursor_raw(next_idx) , vals_old);
        counter_runningCursor = counter_runningCursor+1;

        if frameNum >= params.cursor.win_smooth_cursor
    %         cursor_brain = mean(logger.decoder(frameNum-(params.cursor.win_smooth_cursor-1):frameNum,2));
            cursor_brain = mean((logger.decoder(frameNum-(params.cursor.win_smooth_cursor-1):frameNum,2)-cursor_mean)./sqrt(cursor_var));
        else
            cursor_brain = cursor_brain_raw;
        end
    elseif strcmp(mode, 'baseline')
        cursor_brain = NaN;
    end
    
%% Check for overlap of ROIs and image (VERY SLOW)
% % % y_tmp = reshape(y_idx, baselineStuff.ROIs.cell_size_max , rois.numCells);
% % % size(y_tmp)
% % CI = zeros(size(data.currentImage,1), size(data.currentImage,2));
% % CI(sub2ind(size(data.currentImage), y_idx , x_idx)) = data.currentImage(sub2ind(size(data.currentImage), y_idx , x_idx));
% 
% WI = zeros(size(data.currentImage,1), size(data.currentImage,2));
% % size(baselineStuff.ROIs.cellWeightings)
% WI(sub2ind(size(data.currentImage), y_idx , x_idx)) = repmat(baselineStuff.ROIs.cellWeightings .* F_zscore, 230, 1);
% 
% LI = zeros(size(data.currentImage,1), size(data.currentImage,2));
% LI(sub2ind(size(data.currentImage), y_idx , x_idx)) = baselineStuff.ROIs.spatial_footprints_tall_warped(:,4);
% 
% % CWI = CI .* WI;
% % LCWI = CI .* LI .* WI;
% LWI = LI .* WI;
% plotUpdatedImagesc(LWI , [-1,4], 'test')

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
    CS_threshold = algorithm_thresholdState(cursor_output, params.cursor.threshold_value);
    
    %%  ===== TRIAL STRUCTURE =====
    % CE = current epoch
    % ET = epoch transition signal
    % CS = current state
    
    % START BUILDING UP STATS
    if frameNum == 1
        CE_buildingUpStats = 1;
        %     soundVolume = 0;
        %     setSoundVolumeTeensy(soundVolume);
    end
    % BUILDING UP STATS
    if CE_buildingUpStats
    end
    % END BUILDING UP STATS
    if CE_buildingUpStats && frameNum > params.trial.duration_buildingUpStats
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
        %         disp(['Logger & Params Saved: frameCounter = ' num2str(frameNum)]);
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
        
        if frameNum > (counter_last_z_correction + interval_z_correction)

            
            if delta ~=0
                clampedDelta = sign(delta) * min(abs(delta), max_z_delta);
                disp(['moving fast Z by one step: ', num2str(clampedDelta)]) %num2str(delta)])
%                 currentPosition = moveFastZ(source, [], clampedDelta, [], [20,380]);
                delta_moved = gather(clampedDelta);
                counter_last_z_correction = frameNum;
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
    
end
%% Teensy Output calculations

if CE_experimentRunning
    if frequencyOverride
        voltage_cursorCurrentPos = convert_cursor_to_voltage(params.cursor.threshold_value , params.cursor.bounds_cursor , params.cursor.voltage_at_threshold);
    else
        voltage_cursorCurrentPos = convert_cursor_to_voltage(cursor_output , params.cursor.bounds_cursor, params.cursor.voltage_at_threshold);
    end
%     voltage_cursorCurrentPos = (mod(frameNum,2)+0.5);
%     source.hSI.task_cursorCurrentPos.writeAnalogData(voltage_cursorCurrentPos);

    freqToOutput = convert_voltage_to_frequency(voltage_cursorCurrentPos , 3.3 , params.cursor.range_freqOutput); % for logging purposes only. function should mimic (exactly) the voltage to frequency transformation on teensy
end

% save([directory , '\logger.mat'], 'logger')
% saveParams(directory)
% disp(['Logger & Params Saved: frameCounter = ' num2str(frameNum)]);

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
%     %     if frameNum > 15
%     %         %     if frameNum > 1
%     %         plotUpdatedOutput4([nanmean(logger.motionCorrection.MC_correlation(frameNum-15:frameNum,:),1)], duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 62)
%     %         %         plotUpdatedOutput4([nanmean(logger.motionCorrection.MC_correlation(frameNum-15:frameNum,:),1)], duration_plotting, frameRate, 'Motion Correction Correlation Rolling', 10, 12)
%     %     end
%
%     %     plotUpdatedOutput4(dFoF(1:10) , duration_plotting , frameRate , 'dFoF' , 20, 50)

%     plotUpdatedOutput6(cursor_output , duration_plotting , frameRate , 'cursor' , 10,5)
%     drawnow

%     %     plotUpdatedOutput7(logger.motionCorrection.MC_correlation(frameNum-1) , duration_plotting , frameRate , 'Motion Correction Correlation' , 10,54)
%
%
%     %     if mod(frameNum,300) == 0 && frameNum > 300
%     %         plotUpdatedOutput5([nanmean(logger.motionCorrection.MC_correlation(frameNum-300:frameNum,:),1)],...
%     %             params.timing.duration_session, frameRate, 'Motion Correction Correlation All', 10, 50)
%     %     end
%
%
%     plotUpdatedOutput7(frame_corrs,...
%         duration_plotting, frameRate, 'Z Frame Correlations', 10, 10)
%     drawnow
%
% end

% % if mod(frameNum,30*60*5) == 0
% if frameNum == round(params.timing.duration_session * 0.9)...
%         || frameNum == round(params.timing.duration_session * 0.95)...
%         || frameNum == round(params.timing.duration_session * 0.98)...
%         || frameNum == round(params.timing.duration_session * 0.99)
%     save([directory , '\logger.mat'], 'logger')
%     disp(['Logger Saved: frameCounter = ' num2str(frameNum)]);
% end

% frameNum

%% DATA LOGGING
if ~isnan(frameNum)
    logger_valsROIs(frameNum,:) = data.vals_neurons; %already done above
    
    logger.timeSeries(frameNum,1) = frameNum;
    logger.timeSeries(frameNum,2) = CS_quiescence;
    logger.timeSeries(frameNum,3) = ET_trialStart;
    logger.timeSeries(frameNum,4) = CE_trial;
    logger.timeSeries(frameNum,5) = soundVolume;
    logger.timeSeries(frameNum,6) = counter_trialIdx;
    logger.timeSeries(frameNum,7) = CS_threshold;
    logger.timeSeries(frameNum,8) = ET_rewardToneHold; % reward signals
    logger.timeSeries(frameNum,9) = CE_rewardToneHold;
    logger.timeSeries(frameNum,10) = counter_rewardToneHold;
    logger.timeSeries(frameNum,11) = frequencyOverride;
    logger.timeSeries(frameNum,12) = ET_rewardDelivery;
    logger.timeSeries(frameNum,13) = CE_rewardDelivery;
    logger.timeSeries(frameNum,14) = counter_rewardDelivery;
    logger.timeSeries(frameNum,15) = ET_ITI_withZ;
    logger.timeSeries(frameNum,16) = CE_ITI_withZ;
    logger.timeSeries(frameNum,17) = counter_ITI_withZ;
    logger.timeSeries(frameNum,18) = ET_waitForBaseline;
    logger.timeSeries(frameNum,19) = CE_waitForBaseline;
    logger.timeSeries(frameNum,20) = ET_timeout;
    logger.timeSeries(frameNum,21) = CE_timeout;
    logger.timeSeries(frameNum,22) = counter_timeout;
    logger.timeSeries(frameNum,23) = CE_buildingUpStats;
    logger.timeSeries(frameNum,24) = CE_experimentRunning;
    logger.timeSeries(frameNum,25) = NumOfRewardsAcquired;
    logger.timeSeries(frameNum,26) = NumOfTimeouts;
    logger.timeSeries(frameNum,27) = data.hash_image;
    logger.timeSeries(frameNum,28) = trialNum;
    logger.timeSeries(frameNum,29) = fakeFeedback_inUse;
    logger.timeSeries(frameNum,30) = trialType_cursorOn;
    logger.timeSeries(frameNum,31) = trialType_feedbackLinked;
    logger.timeSeries(frameNum,32) = trialType_rewardOn;
    
    
    logger.timers(frameNum,1) = now;
    logger.timers(frameNum,2) = toc;
    
    logger.decoder(frameNum,1) = cursor_brain;
    logger.decoder(frameNum,2) = cursor_brain_raw;   % this is computed above
    logger.decoder(frameNum,3) = cursor_output;
    logger.decoder(frameNum,4) = freqToOutput; % note that this is just approximate, since calculation is done on teensy
    logger.decoder(frameNum,5) = voltage_cursorCurrentPos;
    
%     logger.motionCorrection(frameNum,1) = xShift;
%     logger.motionCorrection(frameNum,2) = yShift;
%     logger.motionCorrection(frameNum,3) = MC_corr;
    logger.motionCorrection(frameNum,1) = gather(data.MC.xShift);
    logger.motionCorrection(frameNum,2) = gather(data.MC.yShift);
    logger.motionCorrection(frameNum,3) = gather(data.MC.maxCorr_2d(1));
%     logger.motionCorrection(frameNum,4) = source.hSI.hFastZ.currentFastZs{1}.targetPosition;
    logger.motionCorrection(frameNum,5) = delta_moved;
    logger.motionCorrection(frameNum,6:10) = data.MC.maxCorr_z(1:end); 
    
%     toc  
end

% %% End Session
% if frameNum >= params.timing.duration_session
%     endSession
% end

if frameNum == last_frameNum
    logger_output = logger;
    loggerNames_output = loggerNames;
    logger_valsROIs_output = logger_valsROIs;
else
    logger_output = [];
    loggerNames_output = [];
    logger_valsROIs_output = [];
end

%% FUNCTIONS
    function updateLoggerTrials_START % calls at beginning of a trial
        logger.trials(trialNum,1) = trialNum;
        logger.trials(trialNum,2) = now;
        logger.trials(trialNum,3) = frameNum;
        logger.trials(trialNum,4) = trialType_cursorOn;
        logger.trials(trialNum,5) = trialType_feedbackLinked;
        logger.trials(trialNum,6) = trialType_rewardOn;
    end
    function updateLoggerTrials_END(success_outcome) % calls at end of a trial
        logger.trials(trialNum,7) = trialNum;
        logger.trials(trialNum,8) = now;
        logger.trials(trialNum,9) = frameNum;
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
        
        %         frameNum = 0;
        CE_buildingUpStats = 1;
        CE_experimentRunning = 1;
        cursor_brain = 0;
        cursor_brain_raw = 0;
        cursor_output = 0;
        
        NumOfRewardsAcquired = 0;
        NumOfTimeouts = 0;
        trialNum = 1;

        loggerNames.timeSeries{1} = 'frameNum';
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
        loggerNames.motionCorrection{8} = 'z_correlation_4';
        loggerNames.motionCorrection{8} = 'z_correlation_5';
        
        loggerNames.trials{1} = 'trialNum_trialStart';
        loggerNames.trials{2} = 'time_now_trialStart';
        loggerNames.trials{3} = 'frameNum_trialStart';
        loggerNames.trials{4} = 'trialType_cursorOn';
        loggerNames.trials{5} = 'trialType_feedbackLinked';
        loggerNames.trials{6} = 'trialType_rewardOn';
        loggerNames.trials{7} = 'trialNum_trialEnd';
        loggerNames.trials{8} = 'time_now_trialEnd';
        loggerNames.trials{9} = 'frameNum_trialEnd';
        loggerNames.trials{10} = 'success_outcome';

        %         clear logger
        logger.timeSeries = NaN(params.timing.duration_session, length(loggerNames.timeSeries));
        logger.timers = NaN(params.timing.duration_session, length(loggerNames.timers));
        logger.decoder = NaN(params.timing.duration_session, length(loggerNames.decoder));
        logger.motionCorrection = NaN(params.timing.duration_session,  length(loggerNames.motionCorrection));
        logger.trials = NaN(size(trialStuff.condTrials,1),  length(loggerNames.trials));

        logger_valsROIs = nan(params.timing.duration_session , rois.numCells);
        runningVals = nan(params.rollingStats.duration_rolling , rois.numCells);
        running_cursor_raw = nan(params.rollingStats.duration_rolling , 1);
        
        rolling_var_obj_cells = rolling_var_and_mean();
        rolling_var_obj_cells = rolling_var_obj_cells.set_key_properties(size(runningVals) , params.rollingStats.duration_rolling);
        rolling_var_obj_cursor = rolling_var_and_mean();
        rolling_var_obj_cursor = rolling_var_obj_cursor.set_key_properties([1,1] , params.rollingStats.duration_rolling);
        
        rolling_z_mean_obj = rolling_var_and_mean();
        rolling_z_mean_obj = rolling_z_mean_obj.set_key_properties(size(permute(data.MC.im_buffer_rolling_z, [3,1,2])), params.MC.numFrames_avgWin_zCorr);
        
        counter_runningVals = 1;
        counter_runningCursor = 1;

        %         saveParams(directory)
    end

%     function endSession
%         disp('SESSION OVER')
%         frameNum = NaN;
%         CE_experimentRunning = 0;
%
%         save([directory , '\logger.mat'], 'logger')
%         save([directory , '\logger_valsROIs.mat'], 'logger_valsROIs')
%         saveParams(directory)
%         disp('=== Loggers and expParams saved ===')
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
%         %         frameNum = 0;
%         CE_buildingUpStats = 0;
%         %         CE_experimentRunning = 0;
%         cursor_output = 0;
%
%         %         setSoundVolumeTeensy(0);
%         source.hSI.task_cursorAmplitude.writeDigitalData(0);
%         source.hSI.task_goalAmplitude.writeDigitalData(0);
%         
%     end

%     function saveParams(directory)
%         expParams.frameRate = frameRate;
%         expParams.duration_session = params.timing.duration_session;
%         expParams.duration_trial = duration_trial;
%         expParams.win_smooth_cursor = params.cursor.win_smooth_cursor;
%         %         expParams.F_baseline_prctile = F_baseline_prctile;
%         expParams.duration_threshold = duration_threshold;
%         expParams.threshold_value = threshold_value;
%         expParams.range_cursor = params.cursor.bounds_cursor;
%         expParams.range_freqOutput = range_freqOutput;
%         expParams.voltage_at_threshold = voltage_at_threshold;
%         expParams.duration_timeout = duration_timeout;
%         expParams.numCells = rois.numCells;
%         expParams.directory = directory;
%         expParams.duration_rollingStats = params.rollingStats.duration_rolling;
%         expParams.duration_buildingUpStats = params.trial.duration_buildingUpStats;
%         expParams.threshold_quiescence = threshold_quiescence;
%         expParams.duration_rewardTone = duration_rewardTone;
%         expParams.duration_ITI_success = duration_ITI_success;
%         expParams.duration_rewardDelivery = duration_rewardDelivery;
%         expParams.duration_quiescenceHold = duration_quiescenceHold;
%         expParams.reward_duration = reward_duration; % in ms
%         expParams.reward_delay = reward_delay;
%         expParams.LED_duration = LED_duration;
%         expParams.LED_ramp_duration = LED_ramp_duration;
%
%         expParams.image_hash_function = 'hash = sum(sum(image,1).^2)';
% 
%         expParams.loggerNames = loggerNames;
%
%         expParams.baselineStuff = baselineStuff;
%
%         save([directory , '\expParams.mat'], 'expParams')
%         %         save([directory , '\motionCorrectionRefImages.mat'], 'motionCorrectionRefImages')
%     end


%     function [refIm_crop_conjFFT_shift, refIm_crop, indRange_y_crop, indRange_x_crop] = make_fft_for_MC(refIm)
%         refIm = single(refIm);
%         % crop_factor = 5;
%         crop_size = 256; % MAKE A POWER OF 2! eg 32,64,128,256,512
% 
%         length_x = size(refIm,2);
%         length_y = size(refIm,1);
%         middle_x = size(refIm,2)/2;
%         middle_y = size(refIm,1)/2;
% 
%         % indRange_y_crop = [round(middle_y - length_y/crop_factor) , round(middle_y + length_y/crop_factor) ];
%         % indRange_x_crop = [round(middle_x - length_y/crop_factor) , round(middle_x + length_y/crop_factor) ];
% 
%         indRange_y_crop = [round(middle_y - (crop_size/2-1)) , round(middle_y + (crop_size/2)) ];
%         indRange_x_crop = [round(middle_x - (crop_size/2-1)) , round(middle_x + (crop_size/2)) ];
%     
%         refIm_crop = refIm(indRange_y_crop(1) : indRange_y_crop(2) , indRange_x_crop(1) : indRange_x_crop(2)) ;
% 
%         refIm_crop_conjFFT = conj(fft2(refIm_crop));
%         refIm_crop_conjFFT_shift = fftshift(refIm_crop_conjFFT);
% 
%         % size(refIm_crop_conjFFT_shift,1);
%         % if mod(size(refIm_crop_conjFFT_shift,1) , 2) == 0
%         %     disp('RH WARNING: y length of refIm_crop_conjFFT_shift is even. Something is very wrong')
%         % end
%         % if mod(size(refIm_crop_conjFFT_shift,2) , 2) == 0
%         %     disp('RH WARNING: x length of refIm_crop_conjFFT_shift is even. Something is very wrong')
%         % end
% 
% %         refIm_crop_conjFFT_shift_centerIdx = ceil(size(refIm_crop_conjFFT_shift)/2);
%     end

    function [delta,frame_corrs, xShifts, yShifts] = calculate_z_position(im_buffer_rolling_z, registrationImage, refIm_crop_conjFFT_shift, stepSize_zstack, maskPref, borderOuter, borderInner)
%         image_toUse = prctile(data.MC.im_buffer_rolling_z, zCorrPtile, 3);

        image_toUse = mean(data.MC.im_buffer_rolling_z, 3);
%         figure; imagesc(image_toUse)
%         size(data.MC.im_buffer_rolling_z)
%         figure; imagesc(squeeze(data.MC.im_buffer_rolling_z(:,:,1)))

        [delta, frame_corrs, xShifts, yShifts] = zCorrection(image_toUse, registrationImage, ...
            refIm_crop_conjFFT_shift, stepSize_zstack, maskPref, borderOuter, borderInner);
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