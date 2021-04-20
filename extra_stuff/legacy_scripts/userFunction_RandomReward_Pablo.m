function outputVoltageToTeensy = userFunction_RandomReward_Pablo(source, event, varargin)
%% Variable stuff
persistent counter_frameNum frameIdx counter_rewards baselineStuff

%define parameters of reward delivery
directory           = 'D:\RH_local\data\scanimage data\round 4 experiments\mouse 8.9b\20201216\';
lights_flag         = true;
mc_flag             = true;
num_mins            = 45;                   % in mins
num_rewards         = 45;                  % number of rewards to deliver
reward_duration     = 200;                  % in ms
reward_delay    	= 200;                  % in ms
LED_duration        = 1;                    % in s
LED_ramp_duration   = 0.1;                  % in s
frameRate           = 30;
duration_session    = frameRate * num_mins * 60;  % ADJUSTABLE: change number value (in seconds/minutes)
npos                = 72 - 2*10;            %72 positions but 10 buffered on each side
cursor_pos          = 1*npos/3;             %set the cursor 1/3 of the way down
target_pos          = 2*npos/3;             %set the target 2/3 of the way down

%establish the current frame and randomly choose the frames for reward
%delivery
if isempty(counter_frameNum)
   counter_frameNum = 0;
   counter_rewards  = 0;
   rng(1)
   frameIdx         = sort(randi(duration_session,num_rewards,1),'ascend');
end
counter_frameNum    = counter_frameNum + 1;

%if frame is randomly selected, give reward
if ismember(counter_frameNum,frameIdx) %mod(counter_frameNum,frames_per_reward) == 1 
    giveReward3(source, 1, 0, reward_duration, reward_delay, LED_duration, 1, LED_ramp_duration); % in ms. This is in the START section so that it only delivers once
    counter_rewards = counter_rewards+1;
end
fprintf('frameNum: %d      total rewards: %d     ',counter_frameNum,counter_rewards)
%if we are showing lights, pass analog output to arduino
if lights_flag
    cursor_voltage = double((cursor_pos/npos)*5);
    target_voltage = double((target_pos/npos)*5);
    
    source.hSI.task_FrequencyOutputVoltage.writeAnalogData(cursor_voltage);
    source.hSI.task_AmplitudeOutputVoltage.writeAnalogData(target_voltage);
end

%if we are taking a long recording, monitor the motion drift with motion
%correction to images stored in baselineStuff
if mc_flag
    currentImage = source.hSI.hDisplay.lastFrame{1};

    if ~isstruct(baselineStuff) 
        load([directory , '\baselineStuff.mat']);
        figure; imagesc(baselineStuff.MC.meanIm)             %show the reference Image we are basing off of
    end
    
        img_MC_moving = baselineStuff.MC.meanImForMC_crop;


        img_MC_moving_rolling = img_MC_moving;


    img_MC_moving = currentImage(baselineStuff.MC.indRange_y_crop(1):baselineStuff.MC.indRange_y_crop(2)  ,...
        baselineStuff.MC.indRange_x_crop(1):baselineStuff.MC.indRange_x_crop(2)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]

    img_MC_moving_rolling(:,:,end+1) = img_MC_moving;
    img_MC_moving_rollingAvg = mean(img_MC_moving_rolling,3);
    
    [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop , baselineStuff.MC.meanImForMC_crop_conjFFT_shift);
    % [xShift , yShift, cxx, cyy] = motionCorrection_ROI(img_MC_moving_rollingAvg , baselineStuff.MC.meanImForMC_crop);
    MC_corr = max(cxx);
    if abs(xShift) >40
        xShift = 0;
    end
    if abs(yShift) >40
        yShift = 0;
    end

    fprintf('xShift: %d      yShift: %d\n',xShift,yShift)
end

end