function userFunction_Alignment_3D_1(source, event, varargin)
%% Variable stuff
global pe shifter rolling_z_mean_obj...
    params data rois...
    frameNum...

%% IMPORT DATA
% Updated 03/14/2023, following userFunction_BMIv11_withZ
frameNum = source.hSI.hStackManager.framesDone;

if frameNum == 1
    params = struct();
    data = struct();
end

data.currentImage = int32(source.hSI.hDisplay.lastFrame{1});
data.currentImage_gpu = gpuArray(data.currentImage);
data.MC.current_position_z = source.hSI.hFastZ.currentFastZs{1}.targetPosition;

%% == USER SETTINGS ==
if frameNum == 1
    % SETTINGS: General
    params.directory = 'D:\RH_local\data\cage_0315\mouse_0315N\20230423\analysis_data';
%     params.directory = 'D:\RH_local\data\cage_0322\mouse_0322R\20230424\analysis_data';
    
    % SETTINGS: TIMING
    params.timing.frameRate          = 30;
    params.timing.duration_plotting  = 30 * params.timing.frameRate; % ADJUSTABLE: change number value (in seconds). Duration of x axis in plots

    % SETTINGS: Motion correction
    params.MC.numFrames_avgWin_zCorr      = 30*2;
    params.MC.intervalFrames_zCorr        = 5;
    params.MC.min_interval_z_correction   = 20*params.timing.frameRate;
    params.MC.max_delta_z_correction      = 0.5;
    params.MC.bandpass_freqs              = [1/64, 1/4];
    params.MC.bandpass_orderButter        = 3;
    params.MC.device                      = 'cuda';
    params.MC.frame_shape_yx              = int64([512,512]);
end

%% INITIALIZE EXPERIMENT
if frameNum == 1 
%     type_stack = 'stack_warped';
    type_stack = 'stack_sparse';
    zstack = load([params.directory , '\', type_stack, '.mat']);
    
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

    im = squeeze(data.im_zstack(data.MC.idx_middle_frame, :,:));
    data.MC.im_refIm_MC_2D = gpuArray(single(im));
    
    % Initialize the shifter class
    shifter = py.bph.motion_correction.Shifter_rigid(params.MC.device);
    shifter.make_mask(py.tuple(params.MC.frame_shape_yx), py.tuple(params.MC.bandpass_freqs), params.MC.bandpass_orderButter);
    shifter.preprocess_template_images(gather(single(cat(1, permute(data.MC.im_refIm_MC_2D, [3,1,2]), data.im_zstack))), py.int(0));
    
    data.MC.im_buffer_rolling_z = gpuArray(zeros([size(data.MC.im_refIm_MC_2D) , params.MC.numFrames_avgWin_zCorr], 'int32'));
    data.MC.counter_buffer_rolling_z = 0;
end

%% == Session Starting & counting ==

% == Start Session ==
if frameNum == 1
    disp('hi. ALIGNMENT SESSION STARTED')
    rolling_z_mean_obj = rolling_var_and_mean();
    rolling_z_mean_obj = rolling_z_mean_obj.set_key_properties(size(permute(data.MC.im_buffer_rolling_z, [3,1,2])), params.MC.numFrames_avgWin_zCorr);
    disp('frameNum = 1')
end

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
data.MC.yShift     = shifts_yx(1);
data.MC.xShift     = shifts_yx(2);
data.MC.maxCorr_2d = single(out{2}.numpy());

out = shifter.find_translation_shifts(gather(data.MC.im_buffer_rolling_z_mean), py.list(int64([1:data.MC.n_slices_zstack])));  %% 0-indexed
data.MC.maxCorr_z = single(out{2}.numpy());

if abs(data.MC.xShift) >60
    data.MC.xShift = 0;
end
if abs(data.MC.yShift) >60
    data.MC.yShift = 0;
end

%% Plotting

if frameNum>1
    plotUpdatedOutput3([data.MC.xShift' data.MC.yShift'], params.timing.duration_plotting, params.timing.frameRate, 'Motion Correction Shifts', 10, 11)
    plotUpdatedOutput7(data.MC.maxCorr_z,...
        params.timing.duration_plotting, params.timing.frameRate, 'Z Frame Correlations', 10, 10)
end