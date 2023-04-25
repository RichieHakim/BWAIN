function [logger , logger_valsROIs] =...
    userFunction_BMIv11_withZ(source, event, varargin)
%% Variable stuff
tic
global logger loggerNames logger_valsROIs logger_dFoFROIs logger_dFoFbaseROIs...
    pe shifter roll_ptile rolling_var_obj_cells rolling_var_obj_cursor rolling_z_mean_obj...
    params data rois sm...
    frameNum...
    baselineStuff trialStuff...
    
% persistent  

%% IMPORT DATA

frameNum = source.hSI.hStackManager.framesDone;

if frameNum == 1
    params = struct();
    data = struct();
    rois = struct();
    sm = struct();
end

data.currentImage = int32(source.hSI.hDisplay.lastFrame{1});
% disp(class(data.currentImage))
% data.currentImage = currentImage;
data.currentImage_gpu = gpuArray(data.currentImage);
data.hash_image = simple_image_hash(data.currentImage);  %% slower on gpu
% hash_image = gather(simple_image_hash(data.currentImage_gpu));
data.MC.current_position_z = source.hSI.hFastZ.currentFastZs{1}.targetPosition;
% data.MC.current_position_z = 0;

%% == USER SETTINGS ==
if frameNum == 1
    % SETTINGS: General
    params.paths.directory = 'D:\RH_local\data\cage_0315\mouse_0315N\20230425\analysis_data';
    params.paths.expSettings = 'D:\RH_local\data\cage_0315\mouse_0315N\20230423\analysis_data\day0_analysis\expSettings.mat';

%     params.paths.directory = 'D:\RH_local\data\cage_0322\mouse_0322R\20230425\analysis_data';
%     params.paths.expSettings = 'D:\RH_local\data\cage_0322\mouse_0322R\20230424\analysis_data\expSettings.mat';  %% Set to false to not pull expSettings
    
    % SETTINGS: TIMING
    params.timing.frameRate          = 30;
    params.timing.duration_plotting  = 30 * params.timing.frameRate; % ADJUSTABLE: change number value (in seconds). Duration of x axis in plots
%     params.timing.duration_session   = params.timing.frameRate*60*60; % ADJUSTABLE: change number value (in seconds/minutes)
    params.timing.duration_session   = source.hSI.hStackManager.framesPerSlice; % ADJUSTABLE: change number value (in seconds/minutes)
    
    if params.timing.duration_session == inf % when you click FOCUS...
        params.timing.duration_session = params.timing.frameRate*60*60; % Just a random number that is long enough
    end
    
    disp(['duration_session :  ', num2str(params.timing.duration_session)])
    
    % SETTINGS: Motion correction
    params.MC.numFrames_avgWin_zCorr      = 30*2;
    params.MC.intervalFrames_zCorr        = 5;
    params.MC.min_interval_z_correction   = 20*params.timing.frameRate;
    params.MC.max_delta_z_correction      = 0.5;
    params.MC.bandpass_freqs              = [1/64, 1/4];
    params.MC.bandpass_orderButter        = 3;
    params.MC.device                      = 'cuda';
    params.MC.frame_shape_yx              = int64([512,512]);
    
    % SETTINGS: Cursor: Only works for params.trial.block_trial == false
    params.cursor.factor_to_use = 1;
    params.cursor.angle_power = 2.0;
 
    params.cursor.threshold_reward     = 1.5;
    params.cursor.thresh_quiescence_cursorDecoder = 0.15;
    params.cursor.thresh_quiescence_cursorMag = 0;
    params.cursor.win_smooth_cursor    = 1; % smoothing window (in frames)
    params.cursor.bounds_cursor        = [-params.cursor.threshold_reward , params.cursor.threshold_reward *1.5];
    params.cursor.range_freqOutput     = [1000 18000]; % this is set in the teensy code (only here for logging purposes)
    params.cursor.voltage_at_threshold = 3.1; % this will be the maximum output voltage ([0:voltage_at_threshold])
    
    % SETTINGS: Mode
%     params.mode = 'baseline';
    params.mode = 'BMI';
    
    % SETTINGS: Trials
    % below in unit seconds
    params.trial.reward_duration = 52; % 01/14/2023 in ms calibrated to 2.5 uL/reward
%     params.trial.reward_duration = 26; % 03/21/2023 in ms calibrated to 2.5 uL/reward, 1:1 water-diluted soymilk
    params.trial.reward_delay = 0; % in ms
    params.trial.LED_duration = 0.2; % in s
    params.trial.LED_ramp_duration = 0.1; % in s
    params.trial.duration_trial          = 20;
    params.trial.duration_timeout        = 4;
    
%     params.trial.duration_threshold      = 0.066; % in seconds
    params.trial.duration_threshold      = 3; % in frames
    
    params.trial.duration_ITI            = 3; % in seconds
    params.trial.duration_rewardDelivery = 1.00; % before it was .2 10/10/22: Controls how long the reward tone is
    
%     params.trial.duration_quiescenceHold = 0.5;  % in seconds
    params.trial.duration_quiescenceHold      = 3; % in frames
    
    params.trial.duration_buildingUpStats    = round(params.timing.frameRate * 60 * 1);
    
    %     % 20230327 Block Structure
    params.blocks.block_trial        = true; % If true, Trial is block-structurized by parameters defined in baselineStuff.
    
    % SETTINGS: Rolling stats
    params.dFoF.duration_rolling = round(params.timing.frameRate * 60 * 10);
    params.dFoF.interval_update  = 10;
    params.dFoF.device           = 'cuda';
    params.dFoF.ptile            = 30;
    params.dFoF.frac_neuropil    = 0.7;
    params.dFoF.additive_offset  = 0;
    params.dFoF.thresh_violation_F_baseline = 100;
%     params.dFoF.thresh_violation_dFoF = 50;
    params.dFoF.thresh_violation_dFoF = 15;
    
    if isscalar(params.paths.expSettings)==false
        if isfile(params.paths.expSettings)
            clear expSettings
%             load(params.paths.expSettings);
            load(params.paths.expSettings, 'expSettings'); % To avoid adding variable to a static workspace
%             disp(num2str(exist('expSettings') > 0))
            assert(exist('expSettings') > 0, 'RH ERROR: Imported file from params.paths.expSettings, but variable name is not expSettings');
            
            % overwrite all parameters contained within expSettings
            params = overwrite_struct_fields(params, expSettings);
        else
            error('RH ERROR: params.paths.expSettings is not false, but does not point to a valid file')
        end
    end
end

%% INITIALIZE EXPERIMENT
if frameNum == 1 
    if strcmp(params.mode, 'BMI')
        path_baselineStuff = [params.paths.directory , '\baselineStuff.mat'];
        load(path_baselineStuff);
        disp(['LOADED baselineStuff from:  ' , path_baselineStuff])
    end
    
    path_trialStuff = [params.paths.directory , '\trialStuff.mat'];
    load(path_trialStuff);
    disp(['LOADED trialStuff from:  ' , path_trialStuff])
    
    if strcmp(params.mode, 'BMI')
        type_stack = 'stack_warped';
    elseif strcmp(params.mode, 'baseline')
        type_stack = 'stack_sparse';
    end
    zstack = load([params.paths.directory , '\', type_stack, '.mat']);
    
    %     baselineStuff = baselineStuff_in;
    %     trialStuff = trialStuff_in;
    
    %%% Motion correction python code prep
    try
        pe = pyenv('Version', 'C:\ProgramData\Miniconda\envs\matlab_env\python');  %% prepare python environment
    catch
        disp('failed to initalize Python environment. The environment may already by loaded')
    end
    py.importlib.import_module('bph.motion_correction');
    py.importlib.import_module('rp.rolling_percentile');
    
    s_y = floor((size(data.currentImage,1)-params.MC.frame_shape_yx(1))/2) + 1;
    s_x = floor((size(data.currentImage,2)-params.MC.frame_shape_yx(2))/2) + 1;
    data.MC.idx_im_MC_crop_y = s_y:s_y+params.MC.frame_shape_yx(1)-1;
    data.MC.idx_im_MC_crop_x = s_x:s_x+params.MC.frame_shape_yx(2)-1;
    
    data.im_zstack = eval(['zstack.', type_stack, '.stack_avg']);
    data.im_zstack = single(data.im_zstack(:, data.MC.idx_im_MC_crop_y, data.MC.idx_im_MC_crop_x));
    data.MC.stepSize_zstack = eval(['zstack.', type_stack, '.step_size_um']);
    data.MC.n_slices_zstack = size(data.im_zstack, 1);
    data.MC.idx_middle_frame = ceil(data.MC.n_slices_zstack/2);
    if strcmp(params.mode, 'BMI')
        im = baselineStuff.MC.meanIm;
        data.MC.im_refIm_MC_2D = gpuArray(single(im(data.MC.idx_im_MC_crop_y, data.MC.idx_im_MC_crop_x)));
    elseif strcmp(params.mode, 'baseline')
        im = squeeze(data.im_zstack(data.MC.idx_middle_frame, :,:));
        data.MC.im_refIm_MC_2D = gpuArray(single(im));
    end
    
    % Initialize the shifter class
    shifter = py.bph.motion_correction.Shifter_rigid(params.MC.device);
    shifter.make_mask(py.tuple(params.MC.frame_shape_yx), py.tuple(params.MC.bandpass_freqs), params.MC.bandpass_orderButter);
    shifter.preprocess_template_images(gather(single(cat(1, permute(data.MC.im_refIm_MC_2D, [3,1,2]), data.im_zstack))), py.int(0));
    
    data.MC.im_buffer_rolling_z = gpuArray(zeros([size(data.MC.im_refIm_MC_2D) , params.MC.numFrames_avgWin_zCorr], 'int32'));
%     data.MC.im_buffer_rolling_z = gpuArray(zeros([size(data.MC.im_refIm_MC_2D) , params.MC.numFrames_avgWin_zCorr], 'int16'));
    data.MC.counter_buffer_rolling_z = 0;
    
    if strcmp(params.mode, 'BMI')
        rois.numCells = baselineStuff.ROIs.num_cells;
        rois.x_idx_raw = gpuArray(single(baselineStuff.ROIs.spatial_footprints_tall_warped(:,2)));
        rois.y_idx_raw = gpuArray(single(baselineStuff.ROIs.spatial_footprints_tall_warped(:,3)));
        rois.lam_vals = gpuArray(single(baselineStuff.ROIs.spatial_footprints_tall_warped(:,4)));
        rois.lam_vals(isnan(rois.lam_vals)) = 1;
        rois.F_idx_nan = (1 - isnan(rois.x_idx_raw) .* rois.x_idx_raw);

        rois.Fneu_x_idx_raw = gpuArray(single(baselineStuff.ROIs.Fneu_masks_tall_warped(:,2)));
        rois.Fneu_y_idx_raw = gpuArray(single(baselineStuff.ROIs.Fneu_masks_tall_warped(:,3)));
        rois.Fneu_lam_vals = gpuArray(single(baselineStuff.ROIs.Fneu_masks_tall_warped(:,4)));
        rois.Fneu_lam_vals(isnan(rois.Fneu_lam_vals)) = 1;
        %     rois.Fneu_lam_vals = gpuArray(int16(rois.Fneu_lam_vals));
        rois.Fneu_idx_nan = (1 - isnan(rois.Fneu_x_idx_raw) .* rois.Fneu_x_idx_raw);
    elseif strcmp(params.mode, 'baseline')
        rois.numCells = 1;
    end
    
    roll_ptile = py.rp.rolling_percentile.Rolling_percentile_online(...
        int64(params.dFoF.duration_rolling / params.dFoF.interval_update), int64(rois.numCells), 20, 'cuda');
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
shifts_yx          = single(int32(out{1}.numpy()));
% shifts_yx          = single(int16(out{1}.numpy()));
data.MC.yShift     = shifts_yx(1);
data.MC.xShift     = shifts_yx(2);
data.MC.maxCorr_2d = single(out{2}.numpy());

out = shifter.find_translation_shifts(gather(data.MC.im_buffer_rolling_z_mean), py.list(int64([1:data.MC.n_slices_zstack])));  %% 0-indexed
data.MC.maxCorr_z = single(out{2}.numpy());
[maxVal, maxArg] = max(data.MC.maxCorr_z);
data.MC.delta_z = (data.MC.idx_middle_frame-maxArg) * data.MC.stepSize_zstack;

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
    
    y_idx = max(min(y_idx, single(size(data.currentImage,1))), 1);
    x_idx = max(min(x_idx, single(size(data.currentImage,2))), 1);
    
    Fneu_x_idx     =  rois.Fneu_x_idx_raw + data.MC.xShift;
    Fneu_y_idx     =  rois.Fneu_y_idx_raw + data.MC.yShift;
    
    Fneu_y_idx = max(min(Fneu_y_idx, double(size(data.currentImage,1))), 1);
    Fneu_x_idx = max(min(Fneu_x_idx, double(size(data.currentImage,2))), 1);
    
    % Calculate current frame F / Fneu: lam should be normalized to sum=1,
    % then to get F, F=image(pixels) @ lam (its a sum)
    tall_currentImage_F = single(data.currentImage_gpu(sub2ind(size(data.currentImage_gpu), y_idx , x_idx))) .* rois.F_idx_nan;
    TA_CF_lam = tall_currentImage_F .* rois.lam_vals;
    TA_CF_lam_reshape = reshape(TA_CF_lam , baselineStuff.ROIs.cell_size_max , rois.numCells);
    
    % Calculate Fneu: 'lam' is specified by s2p as just the pixels because
    % the lambda values are all 1. Fneu = (mean(image(pixels))). So we just
    % pre normalize the lam values and take a sum
    tall_currentImage_Fneu = single(data.currentImage_gpu(sub2ind(size(data.currentImage_gpu), Fneu_y_idx , Fneu_x_idx))) .* rois.Fneu_idx_nan;
    TA_CF_Fneu_lam = tall_currentImage_Fneu .* rois.Fneu_lam_vals;
    TA_CF_Fneu_lam_reshape = reshape(TA_CF_Fneu_lam , baselineStuff.ROIs.neuropil_size_max , rois.numCells);
    
    %     % Subtract 0.7 * Fneu from F
    %     vals_neurons = ones(1,numCells);
    %     vals_neurons = nansum( TA_CF_lam_reshape , 1 );
    data.ROIs.vals_neurons = nansum( TA_CF_lam_reshape , 1 ) - nanmean(TA_CF_Fneu_lam_reshape, 1) * params.dFoF.frac_neuropil;
    data.ROIs.vals_neurons = gather(data.ROIs.vals_neurons);
    
elseif strcmp(params.mode, 'baseline')
    data.ROIs.vals_neurons = NaN;
end

%% == ROLLING STATS ==
data.ROIs.cursor_brain_raw = NaN;
sm.fakeFeedback_inUse = NaN;
if sm.CE_experimentRunning
    if strcmp(params.mode, 'BMI')
        % 03/27/2023 block-trial preparation
        if params.blocks.block_trial
            if ~sm.CE_trial && ((frameNum - sm.blocks.frameNum_blockStart >= sm.blocks.block_timecap * params.timing.frameRate) || (sm.NumOfRewardsAcquired - sm.blocks.rewardNum_blockStart >= sm.blocks.block_rewardcap))
                disp(['Starting new block.'])
                temp_blockNum = sm.blocks.blockNum + 1;
                sm.blocks = baselineStuff.block_sequence.blocks(temp_blockNum);
                sm.cursor = baselineStuff.cursors(sm.blocks.cursor_to_use);
                sm.blocks.blockNum = temp_blockNum;
                sm.blocks.frameNum_blockStart = frameNum;
                sm.blocks.rewardNum_blockStart = sm.NumOfRewardsAcquired;
                disp(['Current Block: ', num2str(sm.blocks.blockNum)])
                disp(sm.blocks)
                disp(sm.cursor)
            end
        else
            % Set to defaults
            sm.cursor = params.cursor;
    %         sm.blocks = params.blocks;
        end
        %         next_idx = mod(data.ROIs.counter_runningVals-1 , params.dFoF.duration_rolling)+1;
        %         vals_old = data.ROIs.runningVals(next_idx , :);
        %         data.ROIs.runningVals(next_idx,:) = data.vals_neurons;
        %         % 20230126 Now, frameNum follows ScanImage Frames Done
        % %             [rolling_var_obj_cells , F_mean , F_var] = rolling_var_obj_cells.step(frameNum , data.ROIs.runningVals(next_idx,:) , vals_old);
        %         [rolling_var_obj_cells , F_mean , F_var] = rolling_var_obj_cells.step(data.ROIs.runningVals(next_idx,:) , vals_old);
        %         data.ROIs.counter_runningVals = data.ROIs.counter_runningVals+1;
        %         if frameNum == 1
        %             F_mean = data.vals_neurons;
        %             F_var = ones(size(F_mean));
        %         end
        %
        %         F_std = sqrt(F_var);
        %         %     F_std = nanstd(data.ROIs.runningVals , [] , 1);
        %         %     F_std = ones(size(vals_smooth));
        %         F_std(F_std < 0.01) = inf;
        %         %     F_mean = nanmean(data.ROIs.runningVals , 1);
        %         F_zscore = single((data.vals_neurons-F_mean)./F_std);
        %         F_zscore(isnan(F_zscore)) = 0;
        
        data.ROIs.vals_neurons_offset = data.ROIs.vals_neurons + params.dFoF.additive_offset;
        if (mod(frameNum, params.dFoF.interval_update) == 0) || frameNum < 3
            data.ROIs.F_baseline = single(roll_ptile.step(py.numpy.array(data.ROIs.vals_neurons_offset)).cpu().numpy());
        end
        data.ROIs.dFoF = (data.ROIs.vals_neurons_offset - data.ROIs.F_baseline) ./ data.ROIs.F_baseline;
        %         data.ROIs.factors = (data.ROIs.dFoF / (norm(v1 , axis=0, keepdims=True))).T  @ (v2  / norm(v2 , axis=0, keepdims=True))
        
        % 20230313 Online Trace Quality Metric
        data.ROIs.violations = (data.ROIs.F_baseline < params.dFoF.thresh_violation_F_baseline) | (data.ROIs.dFoF > params.dFoF.thresh_violation_dFoF);
        data.ROIs.dFoF(data.ROIs.violations) = NaN;
        
        % % 20230312 angle decoder application
        % Norm of each decoder in factor_space should be 1
        data.ROIs.decoder_magnitudes = nansum(data.ROIs.dFoF' .* baselineStuff.factor_space);
        data.ROIs.decoder_angles = data.ROIs.decoder_magnitudes / norm(data.ROIs.decoder_magnitudes);
        data.ROIs.cursor_positions = (abs(data.ROIs.decoder_angles) .^ sm.cursor.angle_power) .* data.ROIs.decoder_magnitudes;

        data.ROIs.cursor_brain_raw = data.ROIs.cursor_positions(sm.cursor.factor_to_use);
        data.ROIs.cursor_brain = data.ROIs.cursor_brain_raw;
        
        
        %         data.ROIs.cursor_brain_raw = F_zscore * baselineStuff.ROIs.cellWeightings';
        %         logger.decoder(frameNum,2) = data.ROIs.cursor_brain_raw;
        %
        %         next_idx = mod(data.ROIs.counter_runningCursor-1 , params.dFoF.duration_rolling)+1;
        %         vals_old = data.ROIs.running_cursor_raw(next_idx);
        %         data.ROIs.running_cursor_raw(next_idx) = data.ROIs.cursor_brain_raw;
        % %         [rolling_var_obj_cursor , cursor_mean , cursor_var] = rolling_var_obj_cursor.step(frameNum , data.ROIs.running_cursor_raw(next_idx) , vals_old);
        %         [rolling_var_obj_cursor , cursor_mean , cursor_var] = rolling_var_obj_cursor.step(data.ROIs.running_cursor_raw(next_idx) , vals_old);
        %         data.ROIs.counter_runningCursor = data.ROIs.counter_runningCursor+1;
        %
        %         if frameNum >= sm.cursor.win_smooth_cursor
        %     %         data.ROIs.cursor_brain = mean(logger.decoder(frameNum-(sm.cursor.win_smooth_cursor-1):frameNum,2));
        %             data.ROIs.cursor_brain = nanmean((logger.decoder(frameNum-(sm.cursor.win_smooth_cursor-1):frameNum,2)-cursor_mean)./sqrt(cursor_var));
        %         else
        %             data.ROIs.cursor_brain = data.ROIs.cursor_brain_raw;
        %         end
    elseif strcmp(params.mode, 'baseline')
        data.ROIs.dFoF = NaN;
        data.ROIs.F_baseline = NaN;
        data.ROIs.decoder_magnitudes = NaN;
        data.ROIs.decoder_angles = NaN;
        data.ROIs.cursor_positions = NaN;
        data.ROIs.cursor_brain = NaN;
    end
    
    %% Check for overlap of ROIs and image (VERY SLOW)
    if 0
        if (sm.CE_rewardDelivery && sm.counter_rewardDelivery<2) && strcmp(params.mode, 'BMI')
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
    end
    %% Trial prep 
    sm.trialType_cursorOn = trialStuff.condTrialBool(sm.trialNum,1);
    sm.trialType_feedbackLinked = trialStuff.condTrialBool(sm.trialNum,2);
    sm.trialType_rewardOn = trialStuff.condTrialBool(sm.trialNum,3);
    
    if sm.CE_trial && (~sm.trialType_feedbackLinked) && (~isnan(sm.counter_trialIdx))
        data.ROIs.cursor_output = trialStuff.fakeFeedback.fakeCursors(sm.trialNum, sm.counter_trialIdx+1);
        sm.fakeFeedback_inUse = 1;
    else
        data.ROIs.cursor_output = data.ROIs.cursor_brain;
        sm.fakeFeedback_inUse = 0;
    end
   
    
    %     sm.CS_quiescence = algorithm_quiescence(data.ROIs.cursor_brain, sm.cursor.threshold_quiescence);
    % Assumption: The last decoder is avgVector
    if strcmp(params.mode, 'BMI')
        sm.CS_quiescence = algorithm_cursor_quiescence(data.ROIs.cursor_positions, data.ROIs.decoder_magnitudes, sm.cursor.factor_to_use, sm.cursor.thresh_quiescence_cursorDecoder, sm.cursor.thresh_quiescence_cursorMag);
    elseif strcmp(params.mode, 'baseline')
%         Just a dummy variable
        sm.CS_quiescence = 1;
    end
    sm.CS_threshold = algorithm_thresholdState(data.ROIs.cursor_output, sm.cursor.threshold_reward);
    
    %%  ===== TRIAL STRUCTURE =====
    % CE = current epoch
    % ET = epoch transition signal
    % CS = current state
    
    % START BUILDING UP STATS
    if frameNum == 1
        sm.CE_buildingUpStats = 1;
        %     sm.soundVolume = 0;
        %     setSoundVolumeTeensy(sm.soundVolume);
    end
    % BUILDING UP STATS
    if sm.CE_buildingUpStats
        source.hSI.task_cursorAmplitude.writeDigitalData(0);
        source.hSI.task_goalAmplitude.writeDigitalData(0);
    end
    % END BUILDING UP STATS
    if sm.CE_buildingUpStats && frameNum > params.trial.duration_buildingUpStats
        sm.CE_buildingUpStats = 0;
        sm.ET_waitForBaseline = 1;
    end
        
    % START WAIT FOR BASELINE
    if sm.ET_waitForBaseline
        sm.ET_waitForBaseline = 0;
        sm.CE_waitForBaseline = 1;
        sm.soundVolume = 0;
        %     setSoundVolumeTeensy(sm.soundVolume);
        sm.counter_quiescenceHold = 0;
    end
    % WAIT FOR BASELINE
    if sm.CE_waitForBaseline
        if strcmp(params.mode, 'BMI')
            if sm.CS_quiescence == 1
                sm.counter_quiescenceHold = sm.counter_quiescenceHold + 1;
            else
                sm.counter_quiescenceHold = 0;
            end
        elseif strcmp(params.mode, 'baseline')
            if rand > 0.6
                sm.counter_quiescenceHold = sm.counter_quiescenceHold + 1;
            end
        end
    end
    % END WAIT FOR BASELINE (QUIESCENCE ACHIEVED)
    %     if sm.CE_waitForBaseline && sm.CS_quiescence
    %         sm.CE_waitForBaseline = 0;
    %         sm.ET_trialStart = 1;
    %     end
    
    % 2022/10/10 Let them HOLD the quiescence
    if sm.CE_waitForBaseline && (sm.counter_quiescenceHold > params.trial.duration_quiescenceHold)
        sm.CE_waitForBaseline = 0;
        sm.ET_trialStart = 1;
    end
    
    % START TRIAL
    if sm.ET_trialStart
        sm.ET_trialStart = 0;
        sm.CE_trial = 1;
        sm.counter_trialIdx = 0;
        sm.counter_CS_threshold = 0;
        
        updateLoggerTrials_START
        if ~sm.trialType_feedbackLinked
            data.ROIs.cursor_output = trialStuff.fakeFeedback.fakeCursors(sm.trialNum, sm.counter_trialIdx+1);
            sm.fakeFeedback_inUse = 1;
        end
        if sm.trialType_cursorOn
            sm.soundVolume = 1;
        else
            sm.soundVolume = 0;
        end
        %     setSoundVolumeTeensy(sm.soundVolume);
        source.hSI.task_cursorAmplitude.writeDigitalData(sm.soundVolume);
        source.hSI.task_goalAmplitude.writeDigitalData(1);
        source.hSI.task_cursorGoalPos.writeAnalogData(3.2);
        
        sm.frequencyOverride = 0;
    end
    % COUNT TRIAL DURATION & COUNT THRESHOLD DURATIONS
    if sm.CE_trial
        sm.counter_trialIdx = sm.counter_trialIdx + 1;
        
        if sm.CS_threshold
            sm.counter_CS_threshold = sm.counter_CS_threshold + 1;
        else
            sm.counter_CS_threshold = 0;
        end
    end
    
    % END TRIAL: FAILURE
    if sm.CE_trial && sm.counter_trialIdx >= round(params.timing.frameRate * params.trial.duration_trial)
        sm.CE_trial = 0;
        sm.ET_timeout = 1;
        sm.counter_trialIdx = NaN;
        sm.fakeFeedback_inUse = 0;
        updateLoggerTrials_END(0)
        sm.trialNum = sm.trialNum+1;
    end
    % START TIMEOUT
    if sm.ET_timeout
        sm.ET_timeout = 0;
        sm.CE_timeout = 1;
        sm.counter_timeout = 0;
        sm.soundVolume = 0;
        sm.NumOfTimeouts = sm.NumOfTimeouts + 1;
        %     setSoundVolumeTeensy(sm.soundVolume);
        source.hSI.task_cursorAmplitude.writeDigitalData(sm.soundVolume);
        source.hSI.task_goalAmplitude.writeDigitalData(0);
    end
    % COUNT TIMEOUT DURATION
    if sm.CE_timeout
        sm.counter_timeout = sm.counter_timeout + 1;
    end
    % END TIMEOUT
    if sm.CE_timeout && sm.counter_timeout >= round(params.timing.frameRate * params.trial.duration_timeout)
        sm.CE_timeout = 0;
        sm.ET_ITI_withZ = 1;
    end
    
    % END TRIAL: THRESHOLD REACHED
    if sm.CE_trial && sm.counter_CS_threshold >= params.trial.duration_threshold
        updateLoggerTrials_END(1)
        sm.CE_trial = 0;
        %     ET_rewardToneHold = 1;
        sm.ET_rewardDelivery = 1;
        sm.trialNum = sm.trialNum+1;
        sm.counter_trialIdx = NaN;
        sm.fakeFeedback_inUse = 0;
    end
    
    % START DELIVER REWARD
    if sm.ET_rewardDelivery
        sm.ET_rewardDelivery = 0;
        sm.CE_rewardDelivery = 1;
        sm.counter_rewardDelivery = 0;
        sm.frequencyOverride = 1;
        %     sm.soundVolume = 0;
        %     setSoundVolumeTeensy(sm.soundVolume);
        if sm.trialType_rewardOn
            %     giveReward2(1, 1, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
            giveReward3(source, 1, 0, params.trial.reward_duration, params.trial.reward_delay, params.trial.LED_duration, 1, params.trial.LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
            %         giveReward3(source, 1, 0, 500, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
        end
        sm.NumOfRewardsAcquired = sm.NumOfRewardsAcquired + 1;
        
        %         save([params.paths.directory , '\logger.mat'], 'logger')
        %         saveParams(params.paths.directory)
        %         disp(['Logger & Params Saved: frameCounter = ' num2str(frameNum)]);
    end
    % COUNT DELIVER REWARD
    if sm.CE_rewardDelivery
        sm.counter_rewardDelivery = sm.counter_rewardDelivery + 1;
    end
    % END DELIVER REWARD
    if sm.CE_rewardDelivery && sm.counter_rewardDelivery >= round(params.timing.frameRate * params.trial.duration_rewardDelivery)
        sm.CE_rewardDelivery = 0;
        sm.ET_ITI_withZ = 1;
        sm.frequencyOverride = 0;
    end
    
    sm.delta_moved = 0; % place holder to potentially be overwritten by 'moveFastZ' function below
    % START INTER-TRIAL-INTERVAL (POST-REWARD): WITH Z-CORRECTION
    if sm.ET_ITI_withZ
        sm.ET_ITI_withZ = 0;
        sm.CE_ITI_withZ = 1;
        sm.counter_ITI_withZ = 0;
        sm.soundVolume = 0;
        source.hSI.task_cursorAmplitude.writeDigitalData(sm.soundVolume);
        source.hSI.task_goalAmplitude.writeDigitalData(0);
        
        if frameNum > (sm.counter_last_z_correction + params.MC.min_interval_z_correction)
            if data.MC.delta_z ~=0
                clampedDelta = sign(data.MC.delta_z) * min(abs(data.MC.delta_z), params.MC.max_delta_z_correction);
                [~] = moveFastZ(source, [], clampedDelta, [], [20,380]);
                disp(['moving fast Z by one step: ', num2str(clampedDelta), ', new position: ', num2str(data.MC.current_position_z + clampedDelta)]) %num2str(data.MC.delta_z)])
                sm.delta_moved = clampedDelta;
                sm.counter_last_z_correction = frameNum;
            elseif abs(data.MC.maxCorr_z(data.MC.idx_middle_frame + 1) - data.MC.maxCorr_z(data.MC.idx_middle_frame - 1)) > abs(max([data.MC.maxCorr_z(data.MC.idx_middle_frame - 1), data.MC.maxCorr_z(data.MC.idx_middle_frame + 1)] - data.MC.maxCorr_z(data.MC.idx_middle_frame)))
                clampedDelta = sign(data.MC.maxCorr_z(data.MC.idx_middle_frame - 1) - data.MC.maxCorr_z(data.MC.idx_middle_frame + 1)) * params.MC.max_delta_z_correction;
                [~] = moveFastZ(source, [], clampedDelta, [], [20,380]);
                disp(['moving fast Z by one step: ', num2str(clampedDelta), ', new position: ', num2str(data.MC.current_position_z + clampedDelta)]) %num2str(data.MC.delta_z)])
                sm.delta_moved = clampedDelta;
                sm.counter_last_z_correction = frameNum;
            end
        end
    end
    % COUNT INTER-TRIAL-INTERVAL (POST-REWARD)
    if sm.CE_ITI_withZ
        sm.counter_ITI_withZ = sm.counter_ITI_withZ + 1;
    end
    % END INTER-TRIAL-INTERVAL
    if sm.CE_ITI_withZ && sm.counter_ITI_withZ >= round(params.timing.frameRate * params.trial.duration_ITI)
        sm.counter_ITI_withZ = NaN;
        sm.CE_ITI_withZ = 0;
        sm.ET_waitForBaseline = 1;
    end
    
end
%% Teensy Output calculations

if sm.CE_experimentRunning
    if sm.frequencyOverride
        data.voltage_cursorCurrentPos = convert_cursor_to_voltage(sm.cursor.threshold_reward , sm.cursor.bounds_cursor , sm.cursor.voltage_at_threshold);
    else
        data.voltage_cursorCurrentPos = convert_cursor_to_voltage(data.ROIs.cursor_output , sm.cursor.bounds_cursor, sm.cursor.voltage_at_threshold);
    end
%         data.voltage_cursorCurrentPos = (mod(frameNum,2)+0.5);
    source.hSI.task_cursorCurrentPos.writeAnalogData(double(data.voltage_cursorCurrentPos));
    
    data.freqToOutput = convert_voltage_to_frequency(data.voltage_cursorCurrentPos , 3.3 , sm.cursor.range_freqOutput); % for logging purposes only. function should mimic (exactly) the voltage to frequency transformation on teensy
end

% save([params.paths.directory , '\logger.mat'], 'logger')
% saveParams(params.paths.directory)
% disp(['Logger & Params Saved: frameCounter = ' num2str(frameNum)]);

%% Plotting

if frameNum>1
    
    plotUpdatedOutput2([sm.CE_waitForBaseline*0.1, sm.CE_trial*0.2,...
        sm.CE_rewardDelivery*0.3, sm.CE_timeout*0.4, sm.CE_buildingUpStats*0.5, sm.fakeFeedback_inUse*0.6],...
        params.timing.duration_plotting, params.timing.frameRate, 'Rewards', 10, 22, ['# Rewards: ' , num2str(sm.NumOfRewardsAcquired) , ' ; # Timeouts: ' , num2str(sm.NumOfTimeouts)])
    plotUpdatedOutput3([data.MC.xShift' data.MC.yShift'], params.timing.duration_plotting, params.timing.frameRate, 'Motion Correction Shifts', 10, 11)
    
    if frameNum > 25
        plotUpdatedOutput4(nanmean(logger.motionCorrection(frameNum-15:frameNum,3),1),...
            params.timing.duration_plotting, params.timing.frameRate, 'Motion Correction Correlation Rolling', 10, 12)
    end
    
    plotUpdatedOutput6([data.ROIs.cursor_output, data.ROIs.cursor_brain],...
        params.timing.duration_plotting, params.timing.frameRate, ['cursor_output, ', 'cursor_brain'] , 10,3)
    
    
    if mod(frameNum,30) == 0 && frameNum > 300
        plotUpdatedOutput5([nanmean(logger.motionCorrection(frameNum-300:frameNum-1,3),1)],...
            params.timing.duration_session, params.timing.frameRate, 'Motion Correction Correlation All', 10, 1)
    end
    
    plotUpdatedOutput7(data.MC.maxCorr_z,...
        params.timing.duration_plotting, params.timing.frameRate, 'Z Frame Correlations', 10, 10)
    
    % % Below are too slow for online calculation
    %     plotUpdatedOutput(data.ROIs.dFoF(:,1:10) + [1:10]*2, params.timing.duration_plotting, params.timing.frameRate, 'neurons', 1, 1),
    %     plotUpdatedOutput8(data.ROIs.decoder_magnitudes + [1:length(data.ROIs.decoder_magnitudes)]*100, params.timing.duration_plotting, params.timing.frameRate, 'magnitudes', 5, 1),
    %     plotUpdatedOutput9(data.ROIs.decoder_angles + [1:length(data.ROIs.decoder_angles)]*1.5, params.timing.duration_plotting, params.timing.frameRate, 'angles', 5, 1),
    
    %     drawnow
end

%% DATA LOGGING
if ~isnan(frameNum)
    logger_valsROIs(frameNum,:) = data.ROIs.vals_neurons; %already done above
    logger_dFoFROIs(frameNum,:) = data.ROIs.dFoF;
    logger_dFoFbaseROIs(frameNum,:) = data.ROIs.F_baseline;
    
    logger.timeSeries(frameNum,1) = frameNum;
    logger.timeSeries(frameNum,2) = sm.CS_quiescence;
    logger.timeSeries(frameNum,3) = sm.ET_trialStart;
    logger.timeSeries(frameNum,4) = sm.CE_trial;
    logger.timeSeries(frameNum,5) = sm.soundVolume;
    logger.timeSeries(frameNum,6) = sm.counter_trialIdx;
    logger.timeSeries(frameNum,7) = sm.CS_threshold;
    logger.timeSeries(frameNum,8) = sm.ET_rewardToneHold; % reward signals
    logger.timeSeries(frameNum,9) = sm.CE_rewardToneHold;
    logger.timeSeries(frameNum,10) = sm.counter_rewardToneHold;
    logger.timeSeries(frameNum,11) = sm.frequencyOverride;
    logger.timeSeries(frameNum,12) = sm.ET_rewardDelivery;
    logger.timeSeries(frameNum,13) = sm.CE_rewardDelivery;
    logger.timeSeries(frameNum,14) = sm.counter_rewardDelivery;
    logger.timeSeries(frameNum,15) = sm.ET_ITI_withZ;
    logger.timeSeries(frameNum,16) = sm.CE_ITI_withZ;
    logger.timeSeries(frameNum,17) = sm.counter_ITI_withZ;
    logger.timeSeries(frameNum,18) = sm.ET_waitForBaseline;
    logger.timeSeries(frameNum,19) = sm.CE_waitForBaseline;
    logger.timeSeries(frameNum,20) = sm.ET_timeout;
    logger.timeSeries(frameNum,21) = sm.CE_timeout;
    logger.timeSeries(frameNum,22) = sm.counter_timeout;
    logger.timeSeries(frameNum,23) = sm.CE_buildingUpStats;
    logger.timeSeries(frameNum,24) = sm.CE_experimentRunning;
    logger.timeSeries(frameNum,25) = sm.NumOfRewardsAcquired;
    logger.timeSeries(frameNum,26) = sm.NumOfTimeouts;
    logger.timeSeries(frameNum,27) = data.hash_image;
    logger.timeSeries(frameNum,28) = sm.trialNum;
    logger.timeSeries(frameNum,29) = sm.trialNum * sm.CE_trial;
    logger.timeSeries(frameNum,30) = sm.fakeFeedback_inUse;
    logger.timeSeries(frameNum,31) = sm.trialType_cursorOn;
    logger.timeSeries(frameNum,32) = sm.trialType_feedbackLinked;
    logger.timeSeries(frameNum,33) = sm.trialType_rewardOn;
    logger.timeSeries(frameNum,34) = sm.counter_last_z_correction;
    logger.timeSeries(frameNum,35) = sm.delta_moved;
    logger.timeSeries(frameNum,36) = sm.blocks.blockNum;
    logger.timeSeries(frameNum,37) = sm.cursor.factor_to_use;
    logger.timeSeries(frameNum,38) = sm.cursor.angle_power;
    logger.timeSeries(frameNum,39) = sm.cursor.threshold_reward;
    logger.timeSeries(frameNum,40) = sm.cursor.thresh_quiescence_cursorDecoder;
    logger.timeSeries(frameNum,41) = sm.cursor.thresh_quiescence_cursorMag;
    logger.timeSeries(frameNum,42) = sm.cursor.win_smooth_cursor;
    
    logger.timers(frameNum,1) = now;
    logger.timers(frameNum,2) = toc;
    
    logger.decoder(frameNum,1) = data.ROIs.cursor_brain;
    logger.decoder(frameNum,2) = data.ROIs.cursor_brain_raw;   % this is computed above
    logger.decoder(frameNum,3) = data.ROIs.cursor_output;
    logger.decoder(frameNum,4) = data.freqToOutput; % note that this is just approximate, since calculation is done on teensy
    logger.decoder(frameNum,5) = data.voltage_cursorCurrentPos;
    if strcmp(params.mode, 'BMI')
        logger.decoder(frameNum,6) = data.ROIs.decoder_angles(sm.cursor.factor_to_use);   % this is computed above
    end
    logger.decoder(frameNum,7) = data.ROIs.decoder_angles(end);   % this is computed above
    
    logger.factor_space.magnitudes(frameNum,:) = data.ROIs.decoder_magnitudes;
    logger.factor_space.angles(frameNum,:)     = data.ROIs.decoder_angles;
    logger.factor_space.decoders(frameNum,:)     = data.ROIs.cursor_positions;

    
    logger.motionCorrection(frameNum,1) = gather(data.MC.xShift);
    logger.motionCorrection(frameNum,2) = gather(data.MC.yShift);
    logger.motionCorrection(frameNum,3) = gather(data.MC.maxCorr_2d(1));
    logger.motionCorrection(frameNum,4) = data.MC.current_position_z;
    logger.motionCorrection(frameNum,5) = sm.delta_moved;
    logger.motionCorrection(frameNum,6:10) = data.MC.maxCorr_z(1:end);
end

%% End Session
% if  sm.CE_experimentRunning && (frameNum == round(params.timing.duration_session * 0.90))
%     %     source.hSI.task_cursorAmplitude.writeDigitalData(0);
%     %     source.hSI.task_goalAmplitude.writeDigitalData(0);
%     saveLogger(params.paths.directory);
%     saveParams(params.paths.directory);
%     %     source.hSI.task_cursorAmplitude.writeDigitalData(1);
%     %     source.hSI.task_goalAmplitude.writeDigitalData(1);
% end

    
if sm.CE_experimentRunning && ((sm.NumOfRewardsAcquired == 250) || (frameNum == round(params.timing.duration_session * 0.98)) || (sm.blocks.blockNum > params.blocks.blockNum_cap))
    sm.CE_experimentRunning = 0;
    endSession
end

%% FUNCTIONS
    function updateLoggerTrials_START % calls at beginning of a trial
        logger.trials(sm.trialNum,1) = sm.trialNum;
        logger.trials(sm.trialNum,2) = now;
        logger.trials(sm.trialNum,3) = frameNum;
        logger.trials(sm.trialNum,4) = sm.trialType_cursorOn;
        logger.trials(sm.trialNum,5) = sm.trialType_feedbackLinked;
        logger.trials(sm.trialNum,6) = sm.trialType_rewardOn;
        logger.trials(sm.trialNum,7) = sm.blocks.blockNum;
        logger.trials(sm.trialNum,8) = sm.cursor.factor_to_use;
    end
    function updateLoggerTrials_END(success_outcome) % calls at end of a trial
        logger.trials(sm.trialNum,9) = sm.trialNum;
        logger.trials(sm.trialNum,10) = now;
        logger.trials(sm.trialNum,11) = frameNum;
        logger.trials(sm.trialNum,12) = success_outcome;
    end
    function startSession
        % If block-trial, INITIALIZE PARAMETERS
        if strcmp(params.mode, 'BMI') && params.blocks.block_trial
            params.blocks.blockNum_cap       = baselineStuff.block_sequence.blockNum_cap; % After this number of block trials, finish the exp.

            sm.blocks = baselineStuff.block_sequence.blocks(1);
            sm.cursor = baselineStuff.cursors(sm.blocks.cursor_to_use);
            sm.blocks.blockNum = 1; % 1-INDEXED!!!
            sm.blocks.frameNum_blockStart =  params.trial.duration_buildingUpStats;
            sm.blocks.rewardNum_blockStart = 0;
            
            disp(sm.blocks)
            disp(sm.cursor)
        else
            sm.blocks  = struct();
            sm.blocks.blockNum = 0;
            params.blocks.blockNum_cap = inf;
            sm.cursor = params.cursor();
            disp("No Block Trial: Load Default Parameter")
            disp(sm.cursor)
        end
        
        % INITIALIZE VARIABLES
        sm.CE_waitForBaseline = 0;
        sm.CS_quiescence = 0;
        sm.ET_trialStart = 0;
        sm.CE_trial = 0;
        sm.soundVolume = 0;
        sm.counter_trialIdx = 0;
        sm.CS_threshold = 0;
        sm.ET_rewardToneHold = 0; % reward signals
        sm.CE_rewardToneHold = 0;
        sm.counter_rewardToneHold = 0;
        sm.frequencyOverride = 0;
        sm.ET_rewardDelivery = 0;
        sm.CE_rewardDelivery = 0;
        sm.counter_rewardDelivery = 0;
        sm.ET_ITI_withZ = 0;
        sm.CE_ITI_withZ = 0;
        sm.counter_ITI_withZ = 0;
        sm.ET_waitForBaseline = 0;
        sm.CE_waitForBaseline = 0;
        sm.ET_timeout = 0;
        sm.CE_timeout = 0;
        sm.counter_timeout = 0;
        sm.counter_last_z_correction = 0;
        
        %         frameNum = 0;
        sm.CE_buildingUpStats = 1;
        sm.CE_experimentRunning = 1;
        data.ROIs.cursor_brain = 0;
        data.ROIs.cursor_brain_raw = 0;
        data.ROIs.cursor_output = 0;
        
        sm.NumOfRewardsAcquired = 0;
        sm.NumOfTimeouts = 0;
        sm.trialNum = 1;
        
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
        loggerNames.timeSeries{29} = 'trialNum*CE_trial';
        loggerNames.timeSeries{30} = 'fakeFeedback_inUse';
        loggerNames.timeSeries{31} = 'trialType_cursorOn';
        loggerNames.timeSeries{32} = 'trialType_feedbackLinked';
        loggerNames.timeSeries{33} = 'trialType_rewardOn';
        loggerNames.timeSeries{34} = 'counter_last_z_correction';
        loggerNames.timeSeries{35} = 'delta_moved';
        loggerNames.timeSeries{36} = 'blockNum';
        loggerNames.timeSeries{37} = 'cursor_factor_to_use';
        loggerNames.timeSeries{38} = 'cursor_angle_power';
        loggerNames.timeSeries{39} = 'cursor_threshold_reward';
        loggerNames.timeSeries{40} = 'cursor_thresh_quiescence_cursorDecoder';
        loggerNames.timeSeries{41} = 'cursor_thresh_quiescence_cursorMag';
        loggerNames.timeSeries{42} = 'cursor_win_smooth_cursor';
        
        loggerNames.timers{1} = 'time_now';
        loggerNames.timers{2} = 'tic_toc';
        
        loggerNames.decoder{1} = 'cursor_brain';
        loggerNames.decoder{2} = 'cursor_brain_raw';
        loggerNames.decoder{3} = 'cursor_output';
        loggerNames.decoder{4} = 'freqToOutput';
        loggerNames.decoder{5} = 'voltage_cursorCurrentPos';
        loggerNames.decoder{6} = 'decoder_angles';
        loggerNames.decoder{7} = 'avgVec_angle';
        
        loggerNames.motionCorrection{1} = 'xShift';
        loggerNames.motionCorrection{2} = 'yShift';
        loggerNames.motionCorrection{3} = 'MC_correlation';
        loggerNames.motionCorrection{4} = 'current_position_z';
        loggerNames.motionCorrection{5} = 'deltaMoved';
        loggerNames.motionCorrection{6} = 'z_correlation_1';
        loggerNames.motionCorrection{7} = 'z_correlation_2';
        loggerNames.motionCorrection{8} = 'z_correlation_3';
        loggerNames.motionCorrection{9} = 'z_correlation_4';
        loggerNames.motionCorrection{10} = 'z_correlation_5';
        
        loggerNames.trials{1} = 'trialNum_trialStart';
        loggerNames.trials{2} = 'time_now_trialStart';
        loggerNames.trials{3} = 'frameNum_trialStart';
        loggerNames.trials{4} = 'trialType_cursorOn';
        loggerNames.trials{5} = 'trialType_feedbackLinked';
        loggerNames.trials{6} = 'trialType_rewardOn';
        loggerNames.trials{7} = 'blockNum';
        loggerNames.trials{8} = 'factor_to_use';
        loggerNames.trials{9} = 'trialNum_trialEnd';
        loggerNames.trials{10} = 'time_now_trialEnd';
        loggerNames.trials{11} = 'frameNum_trialEnd';
        loggerNames.trials{12} = 'success_outcome';
        
        %         clear logger
        logger.timeSeries = NaN(params.timing.duration_session, length(loggerNames.timeSeries));
        logger.timers = NaN(params.timing.duration_session, length(loggerNames.timers));
        logger.decoder = NaN(params.timing.duration_session, length(loggerNames.decoder));
        logger.motionCorrection = NaN(params.timing.duration_session,  length(loggerNames.motionCorrection));
        logger.trials = NaN(size(trialStuff.condTrials,1),  length(loggerNames.trials));
        logger.factor_space = struct();
        if strcmp(params.mode, 'BMI')
            logger.factor_space.magnitudes = NaN(params.timing.duration_session, size(baselineStuff.factor_space, 2));
            logger.factor_space.angles     = NaN(params.timing.duration_session, size(baselineStuff.factor_space, 2));
            logger.factor_space.decoders     = NaN(params.timing.duration_session, size(baselineStuff.factor_space, 2));
        end
        
        
        % 20230325 Log class single
        logger_valsROIs = single(nan(params.timing.duration_session , rois.numCells));
%         logger_valsROIs = nan(params.timing.duration_session , rois.numCells);
        
        % 20230320 online dFoF logging
        % 20230325 Log class single
        logger_dFoFROIs = single(nan(params.timing.duration_session , rois.numCells));
        logger_dFoFbaseROIs = single(nan(params.timing.duration_session , rois.numCells));
        
        data.ROIs.runningVals = nan(params.dFoF.duration_rolling , rois.numCells);
        data.ROIs.running_cursor_raw = nan(params.dFoF.duration_rolling , 1);
        
        rolling_var_obj_cells = rolling_var_and_mean();
        rolling_var_obj_cells = rolling_var_obj_cells.set_key_properties(size(data.ROIs.runningVals) , params.dFoF.duration_rolling);
        rolling_var_obj_cursor = rolling_var_and_mean();
        rolling_var_obj_cursor = rolling_var_obj_cursor.set_key_properties([1,1] , params.dFoF.duration_rolling);
        
        rolling_z_mean_obj = rolling_var_and_mean();
        rolling_z_mean_obj = rolling_z_mean_obj.set_key_properties(size(permute(data.MC.im_buffer_rolling_z, [3,1,2])), params.MC.numFrames_avgWin_zCorr);
        
        data.ROIs.counter_runningVals = 1;
        data.ROIs.counter_runningCursor = 1;
        
        saveParams(params.paths.directory)
    end

    function endSession
        disp('SESSION OVER')
        frameNum = NaN;
        sm.CE_experimentRunning = 0;
        
        saveLogger(params.paths.directory)
        saveParams(params.paths.directory)
        disp('=== Loggers and expParams saved ===')
        
        
        sm.CE_waitForBaseline = 0;
        sm.CS_quiescence = 0;
        sm.ET_trialStart = 0;
        sm.CE_trial = 0;
        sm.soundVolume = 0;
        sm.counter_trialIdx = 0;
        sm.CS_threshold = 0;
        sm.ET_rewardToneHold = 0; % reward signals
        sm.CE_rewardToneHold = 0;
        sm.counter_rewardToneHold = 0;
        sm.frequencyOverride = 0;
        sm.ET_rewardDelivery = 0;
        sm.CE_rewardDelivery = 0;
        sm.counter_rewardDelivery = 0;
        sm.ET_ITI_withZ = 0;
        sm.CE_ITI_withZ = 0;
        sm.counter_ITI_withZ = 0;
        sm.ET_waitForBaseline = 0;
        sm.CE_waitForBaseline = 0;
        sm.ET_timeout = 0;
        sm.CE_timeout = 0;
        sm.counter_timeout = 0;
        
        %         frameNum = 0;
        sm.CE_buildingUpStats = 0;
        %         sm.CE_experimentRunning = 0;
        data.ROIs.cursor_output = 0;
        
        %         setSoundVolumeTeensy(0);
        source.hSI.task_cursorAmplitude.writeDigitalData(0);
        source.hSI.task_goalAmplitude.writeDigitalData(0);
        
    end

    function saveLogger(directory)
        disp("Saving logger.mat...")
        save([directory , '\logger.mat'], 'logger');
        save([directory , '\logger_valsROIs.mat'], 'logger_valsROIs');
        save([directory , '\logger_dFoFROIs.mat'], 'logger_dFoFROIs');
        save([directory , '\logger_dFoFbaseROIs.mat'], 'logger_dFoFbaseROIs');
    end

    function saveParams(directory)
        expParams.params = params;
        expParams.directory = directory;
        
        expParams.image_hash_function = 'hash = sum(sum(image,1).^2)';
        
        expParams.loggerNames = loggerNames;
        
        expParams.baselineStuff = baselineStuff; % Too Big!
        
%         save([directory , '\expParams.mat'], 'expParams')
%         save([directory , '\expParams.mat'], 'expParams', '-v7')
        save([directory , '\expParams.mat'], 'expParams', '-v7.3')
        %         save([directory , '\motionCorrectionRefImages.mat'], 'motionCorrectionRefImages')
    end

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

%     function [delta,frame_corrs, xShifts, yShifts] = calculate_z_position(img_MC_moving_rolling_z, registrationImage, refIm_crop_conjFFT_shift, referenceDiffs, maskPref, borderOuter, borderInner)
%         image_toUse = mean(img_MC_moving_rolling_z, 3);
%         [delta, frame_corrs, xShifts, yShifts] = zCorrection(image_toUse, registrationImage, ...
%             refIm_crop_conjFFT_shift, referenceDiffs, maskPref, borderOuter, borderInner);
%
%     end

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