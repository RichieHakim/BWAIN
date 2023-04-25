function userFunction_calibrate_tilt(source, event, varargin)
%% Variable stuff
global pe shifter rolling_z_mean_obj...
    params data rois...
    frameNum...
    sm...
    fastZDevice...

%% IMPORT DATA
data.currentImage = int32(source.hSI.hDisplay.lastFrame{1});
data.currentImage_gpu = gpuArray(data.currentImage);
data.MC.current_position_z = source.hSI.hFastZ.currentFastZs{1}.targetPosition;

%% == USER SETTINGS ==
frameNum = source.hSI.hStackManager.framesDone;

    % SETTINGS: General
    params.directory = 'D:\RH_local\data\BMI_cage_g8Test\mouse_g8t\20230314\analysis_data';
    
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
    
    params.positions_dots_calibration = [256-100, 512-200; 256+100, 512-200; 256-100, 512+200; 256+100, 512+200];
    params.z_start = 70;
    params.z_end = 90;
    params.relaxation_step_frames = 20;
    params.averaging_step_frames = 15;
    params.relaxation_flyback_frames = 15;
    params.delta_z_interval = 10;  % in microns
    
if frameNum == 1
    sm = struct();
    sm.ET_takingStack = 0;
    sm.CE_takingStack = 0;
    sm.ET_flyback = 1;
    sm.CE_flyback = 0;
    sm.counter_relaxation_step = 0;
    sm.counter_averaging_step = 0;
    sm.counter_relaxation_flyback = 0;
    sm.counter_frameIter = 0;
    
    fastZDevice = source.hSI.hFastZ.currentFastZs{1};
end

%% == TAKE STACK

data.stack.z_position_current = fastZDevice.targetPosition;

% sm.counter_averaging_step
% sm.counter_relaxation_step

if sm.ET_takingStack
    sm.ET_takingStack = 0;
    sm.CE_takingStack = 1;

    sm.ET_relaxation = 0;
    sm.CE_relaxation = 0;
    sm.counter_relaxation_step = 0;
    sm.counter_averaging_step = 0;
        
%     disp('ET_takingStack')
end
if sm.CE_takingStack
%     disp('CE_takingStack')
    if sm.counter_relaxation_step == params.relaxation_step_frames
%         disp('taking frame')
        frame = data.currentImage;
        
        if sm.counter_averaging_step == params.averaging_step_frames
            moveFastZ(source, event, params.delta_z_interval);
            sm.counter_averaging_step = 0;
            sm.counter_relaxation_step = 0;
        else
            sm.counter_averaging_step = sm.counter_averaging_step + 1;
        end

        if data.stack.z_position_current > params.z_end
            sm.CE_takingStack = 0;
            sm.ET_flyback = 1;
        end
    else
%         disp('relaxing')
        sm.counter_relaxation_step = sm.counter_relaxation_step + 1;
    end
end

if sm.ET_flyback
    sm.ET_flyback = 0;
    sm.CE_flyback = 1;
    source.hSI.hFastZ.move(fastZDevice, params.z_start);
    sm.counter_relaxation_flyback = 0;
%     disp('flyback')
end
if sm.CE_flyback
    if sm.counter_relaxation_flyback == params.relaxation_flyback_frames
        sm.CE_flyback = 0;
        sm.ET_takingStack = 1;
    else
        sm.counter_relaxation_flyback = sm.counter_relaxation_flyback + 1;
    end
end


idx_linear_dots = [double(data.currentImage(sub2ind([512,1024], params.positions_dots_calibration(:,1), params.positions_dots_calibration(:,2))))];

%% Plotting
% size({max(data.currentImage, [], 1)})
if frameNum>1
%     plotUpdatedOutput3([double(data.currentImage(256-100,512-200)), double(data.currentImage(256+100,512-200)), double(data.currentImage(256-100,512+200)), double(data.currentImage(256+100,512+200))], params.timing.duration_plotting, params.timing.frameRate, '', 1, 1)
    plotUpdatedOutput3(idx_linear_dots, params.timing.duration_plotting, params.timing.frameRate, '', 1, 1);
    plotUpdatingPlot({max(data.currentImage, [], 1), data.currentImage(:, 512), data.currentImage(256, :)'}, 0)
end

end
function currentPosition = moveFastZ(source, evt, delta, position, range_position)

    if ~exist('range_position')
        range_position = [0, 400];
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
