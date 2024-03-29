%% Set conditions params
% TODO: Prepare more logger files. Pull from randomly chosen logger file

maxNumTrials = 500; % always make the max number of trials way more than what is possible

condition_names{1} = 'cursorOn';
condition_names{2} = 'feedbackLinked';

condition_names{3} = 'rewardOn';
numConditions = length(condition_names);

% Default setting: For baseline session.
threshold = 1.5;
goal_threshAchieved = 0.4; 
experiment_name = 'random_playback';

% % Scale to get a set percentage of threshold hits
% % threshold = 1.0; %% mouse_0916N
% threshold = 1.6; %% mouse_0908
% goal_threshAchieved = 0.4; % goal of what fraction of fake cursor trials reach threshold
% experiment_name = 'show_intention';

% dir = 'D:\RH_local\data\cage_0916\mouse_0916N\20231030\analysis_data';
% dir = 'D:\RH_local\data\cage_0908\mouse_0908\20231030\analysis_data';
dir = 'D:\RH_local\data\cage_0914\mouse_0914\20231030\analysis_data';
%% Make fake decoder data for each trial
% Import a logger to steal decoder output for fake feedback decoder
path_fakeData = 'D:\RH_local\fake_data_for_trialStuff\logger.mat';
load(path_fakeData)

sample_rate = 30; % in Hz, rough imaging speed
Fs = sample_rate;
maxTrialDuration = 20 * Fs; % Make this the actual trial duration

fakeDecoderData = logger.decoder(:,1);
fakeDecoderData = fakeDecoderData(isnan(fakeDecoderData)==0);
%% Reshuffling optimization
thresh_check_avg = inf;
thresh_check_avg_first10 = inf;

obj_fun = @(thresh_avg) (thresh_avg - goal_threshAchieved);
criterion_fun = @(thresh_avg , thresh) abs(obj_fun(thresh_avg)) < thresh;

cc = 1;
loss_rolling = NaN(10000,1);
acceptable_onsets = find(fakeDecoderData(1:end-maxTrialDuration) < 0);
idx_toUpdate = [1:maxNumTrials];
fakeCursors = NaN(maxNumTrials, maxTrialDuration);
while ~(criterion_fun(thresh_check_avg_first10 , 0.01) && criterion_fun(thresh_check_avg , 0.03))
    acceptable_onsets_shuffled = acceptable_onsets(randperm(length(acceptable_onsets)));
    for ii = 1:length(idx_toUpdate)
        fakeCursors(idx_toUpdate(ii),:) = fakeDecoderData(acceptable_onsets_shuffled(ii):acceptable_onsets_shuffled(ii)+maxTrialDuration-1);
    end

    thresh_check = max(fakeCursors,[],2) > threshold;
    thresh_check_avg = mean(thresh_check);
    thresh_check_avg_first10 = mean(thresh_check(1:10));
    loss_rolling(cc) = ((obj_fun(thresh_check_avg_first10))  +  (obj_fun(thresh_check_avg)));

    side_tooHigh = thresh_check_avg > goal_threshAchieved; % 0 for sub thresh trials, 1 for supra thresh trials
    idx_toUpdate = find(thresh_check == side_tooHigh);
    
    if mod(cc,5)==0
        disp(['working... iter: ' , num2str(cc), ' loss: ', num2str(loss_rolling(cc)), ' frac_thresh: ', num2str(thresh_check_avg), ' frac_threshFirst10: ', num2str(thresh_check_avg_first10)])
    end
    cc = cc+1;
end


%% 20230818 Make success fakeCursor trials longer than 5 seconds
minTrialDuration = 5 * Fs;
[fake_val, fake_cross] = max(max(fakeCursors - threshold, 0) > 0, [], 2);
% disp(num2str(sum((fake_cross ~= 1) & (fake_cross < minTrialDuration))))
trial_tooShort = find((fake_cross ~= 1) & (fake_cross < minTrialDuration));
trial_fail = find(fake_cross == 1);

for ii=1:length(trial_tooShort)
    trial_index = trial_tooShort(ii);
    % Add random variation so that not every trial ends at 5 seconds
    numframe_shift = minTrialDuration - fake_cross(trial_index) + randi(4 * Fs);
    fakeCursors(trial_index,:) = circshift(fakeCursors(trial_index,:), numframe_shift);
    % Randomly choose failed trials to fill frames in
    fillin = trial_fail(randi(length(trial_fail)));
    fakeCursors(trial_index,1:numframe_shift) = fakeCursors(fillin, 1:numframe_shift);
end

% [post_fake_val, post_fake_cross] = max(max(fakeCursors - threshold, 0) > 0, [], 2);
% disp(num2str(sum((post_fake_cross ~= 1) & (post_fake_cross < minTrialDuration))))
% figure()
% plot(post_fake_cross - 1)

%%
fakeCursors_rs = reshape(fakeCursors',1,[])';
figure;
plot([1/Fs:length(fakeCursors_rs)]/Fs, fakeCursors_rs)
%% plotting
figure('Position' , [100,300 , 1000,800]);

subplot(2,2,1)
plot(loss_rolling)
ylabel('loss')
xlabel('iter #')

subplot(2,2,2);
plot(fakeDecoderData); hold on; plot(acceptable_onsets,fakeDecoderData(acceptable_onsets))
title('plot 1: full cursor , plot2: acceptable trial starts')

subplot(2,2,3); 
imagesc(fakeCursors(1:100,:))

% scale_factor = mean(mean(fakeCursors_scaled ./ fakeCursors))
thresh_check_avg
thresh_check_avg_first10
subplot(2,2,4);
plot(thresh_check)
ylabel('thresh_check')
xlabel('iter #')

trialStuff.fakeFeedback.path_fakeData = path_fakeData;
trialStuff.fakeFeedback.fakeDecoderInputData = fakeDecoderData;
trialStuff.fakeFeedback.acceptable_onsets = acceptable_onsets;

trialStuff.fakeFeedback.fakeCursors = fakeCursors;
trialStuff.fakeFeedback.threshold = threshold;
trialStuff.fakeFeedback.goal_threshAchieved = goal_threshAchieved;

% trialStuff.fakeFeedback.fakeCursors = fakeCursors_scaled;
trialStuff.fakeFeedback.thresh_check_avg = thresh_check_avg;
trialStuff.fakeFeedback.thresh_check_avg_first10 = thresh_check_avg_first10;
% trialStuff.fakeFeedback.scale_factor = scale_factor;


%% EXPERIMENTS:
% run only one of the following cell blocks

%% Experiment: goal-directed
%     experiment_name = 'goal_directed_contingency_degridation_blocks';

if strcmpi(experiment_name, 'goal_directed_contingency_degridation_blocks')
    disp('goal_directed_contingency_degridation_blocks')
    cond_bool = [[1,1,1] ;...
        [1,1,0]];
    numConditions = size(cond_bool,1);

    cond_all = [ones(75,1)*1 ; ones(75,1)*2 ; ones(350,1)*1]; % index to use for each trial

    clear conditions_trials
    for ii = 1:numConditions
        conditions_trials(cond_all==ii,:) = repmat(cond_bool(ii,:) , sum(cond_all==ii),1);
    end

    figure; imagesc(conditions_trials)

    trialStuff.expType = experiment_name;
    trialStuff.condTrialBool = conditions_trials;
    trialStuff.condBool = cond_bool;
    trialStuff.condTrials = cond_all;
    trialStuff.condNames = condition_names;
    trialStuff.condProbs = 'N/A';
    trialStuff.homogeneousBlockSize = 'N/A';
%% Experiment: show forward model cells
% - This format makes a matrix where each row is a trial, and each column is
% a condition (ie cursorOn, feedbackLinked, rewardOn)
% - I added the constraint that homogenizes the distribution of conditions
%     experiment_name = 'show_forward_model';
elseif strcmpi(experiment_name, 'show_forward_model')
    disp('show_forward_model')
    cond_bool = [[1,1,1] ;...
                 [0,1,1] ;...
                 [1,0,0]];
    numConditions = size(cond_bool,1);

    prob_cond = [0.8;0.1;0.1];

    hbs = 10; % homogeneous_block_size. MUST be factor of maxNumTrials. ALSO, hbs*prob_cond MUST be all integers

    criterion_fun = @(cond_all,cond) abs(mean(cond_all == cond) - prob_cond(cond)) ==0;

    cond_all = zeros(maxNumTrials,1);
    tmp_conds = [];
    for jj = 1:numConditions
        tmp_conds = [tmp_conds;ones(prob_cond(jj)*hbs,1)*jj];
    end
    for ii= 0:(maxNumTrials/hbs)-1
        criterion_output = zeros(numConditions,1);
        while sum(criterion_output) < 3
            cond_all(1+(ii*hbs):((ii+1)*hbs)) = tmp_conds(randperm(length(tmp_conds)));
            for jj = 1:numConditions
                criterion_output(jj) = criterion_fun(cond_all(1+(ii*hbs):((ii+1)*hbs)),jj);
            end
        end
    end
    figure; plot(cond_all)

    clear conditions_trials
    for ii = 1:numConditions
        conditions_trials(cond_all==ii,:) = repmat(cond_bool(ii,:) , sum(cond_all==ii),1);
    end

    figure; imagesc(conditions_trials)
    figure; plot(smoothdata([cond_all==1,cond_all==2,cond_all==3],1,'sgolay',20))

    trialStuff.expType = experiment_name;
    trialStuff.condTrialBool = conditions_trials;
    trialStuff.condBool = cond_bool;
    trialStuff.condTrials = cond_all;
    trialStuff.condNames = condition_names;
    trialStuff.condProbs = prob_cond;
    trialStuff.homogeneousBlockSize = hbs;
%% Experiment: show 'intention'
% - This format makes a matrix where each row is a trial, and each column is
% a condition (ie cursorOn, feedbackLinked, rewardOn)
% - I added the constraint that homogenizes the distribution of conditions
%     experiment_name = 'show_intention';
elseif strcmpi(experiment_name, 'show_intention')
    disp('show_intention')
    cond_bool = [[1,1,1] ;...
                 [1,0,1] ;...
                 [0,1,0]];
    numConditions = size(cond_bool,1);

    prob_cond = [0.9;0.05;0.05];

    hbs = 20; % homogeneous_block_size. MUST be factor of maxNumTrials. ALSO, hbs*prob_cond MUST be all integers

    criterion_fun = @(cond_all,cond) abs(mean(cond_all == cond) - prob_cond(cond)) ==0;

    cond_all = zeros(maxNumTrials,1);
    tmp_conds = [];
    for jj = 1:numConditions
        tmp_conds = [tmp_conds;ones(prob_cond(jj)*hbs,1)*jj];
    end
    for ii= 0:(maxNumTrials/hbs)-1
        criterion_output = zeros(numConditions,1);
        while sum(criterion_output) < 3
            cond_all(1+(ii*hbs):((ii+1)*hbs)) = tmp_conds(randperm(length(tmp_conds)));
            for jj = 1:numConditions
                criterion_output(jj) = criterion_fun(cond_all(1+(ii*hbs):((ii+1)*hbs)),jj);
            end
        end
    end
    figure; plot(cond_all)

    clear conditions_trials
    for ii = 1:numConditions
        conditions_trials(cond_all==ii,:) = repmat(cond_bool(ii,:) , sum(cond_all==ii),1);
    end

    figure; imagesc(conditions_trials)
    figure; plot(smoothdata([cond_all==1,cond_all==2,cond_all==3],1,'sgolay',20))

    trialStuff.expType = experiment_name;
    trialStuff.condTrialBool = conditions_trials;
    trialStuff.condBool = cond_bool;
    trialStuff.condTrials = cond_all;
    trialStuff.condNames = condition_names;
    trialStuff.condProbs = prob_cond;
    trialStuff.homogeneousBlockSize = hbs;

%% Experiment: dissociate sensory
% - This format makes a matrix where each row is a trial, and each column is
% a condition (ie cursorOn, feedbackLinked, rewardOn)
% - I added the constraint that homogenizes the distribution of conditions
%     experiment_name = 'dissociate_sensory';
elseif strcmpi(experiment_name, 'dissociate_sensory')
    disp('dissociate_sensory')
    cond_bool = [[1,1,1] ;...
                 [1,0,1] ;...
                 [0,1,1]];
    numConditions = size(cond_bool,1);

    prob_cond = [0.8;0.1;0.1];

    hbs = 20; % homogeneous_block_size. MUST be factor of maxNumTrials. ALSO, hbs*prob_cond MUST be all integers

    criterion_fun = @(cond_all,cond) abs(mean(cond_all == cond) - prob_cond(cond)) ==0;

    cond_all = zeros(maxNumTrials,1);
    tmp_conds = [];
    for jj = 1:numConditions
        tmp_conds = [tmp_conds;ones(prob_cond(jj)*hbs,1)*jj];
    end
    for ii= 0:(maxNumTrials/hbs)-1
        criterion_output = zeros(numConditions,1);
        while sum(criterion_output) < 3
            cond_all(1+(ii*hbs):((ii+1)*hbs)) = tmp_conds(randperm(length(tmp_conds)));
            for jj = 1:numConditions
                criterion_output(jj) = criterion_fun(cond_all(1+(ii*hbs):((ii+1)*hbs)),jj);
            end
        end
    end
    figure; plot(cond_all)

    clear conditions_trials
    for ii = 1:numConditions
        conditions_trials(cond_all==ii,:) = repmat(cond_bool(ii,:) , sum(cond_all==ii),1);
    end

    figure; imagesc(conditions_trials)
    figure; plot(smoothdata([cond_all==1,cond_all==2,cond_all==3],1,'sgolay',20))

    trialStuff.expType = experiment_name;
    trialStuff.condTrialBool = conditions_trials;
    trialStuff.condBool = cond_bool;
    trialStuff.condTrials = cond_all;
    trialStuff.condNames = condition_names;
    trialStuff.condProbs = prob_cond;
    trialStuff.homogeneousBlockSize = hbs;



%% Experiment: Random Playback
% - This format makes a matrix where each row is a trial, and each column is
% a condition (ie cursorOn, feedbackLinked, rewardOn)
% - I added the constraint that homogenizes the distribution of conditions
%     experiment_name = 'random_playback';
elseif strcmpi(experiment_name, 'random_playback')
    disp('random_playback')
    cond_bool = [[1,0,1]];
    numConditions = size(cond_bool,1);

    cond_all = [ones(500,1)*1];

    clear conditions_trials
    for ii = 1:numConditions
        conditions_trials(cond_all==ii,:) = repmat(cond_bool(ii,:) , sum(cond_all==ii),1);
    end

    figure; imagesc(conditions_trials)

    trialStuff.expType = experiment_name;
    trialStuff.condTrialBool = conditions_trials;
    trialStuff.condBool = cond_bool;
    trialStuff.condTrials = cond_all;
    trialStuff.condNames = condition_names;
    trialStuff.condProbs = 'N/A';
    trialStuff.homogeneousBlockSize = 'N/A';
else
    error('experiment is not defined')
end
%%
% % dir = 'D:\RH_local\data\cage_0916\mouse_0916N\20231027\analysis_data';
% % dir = 'D:\RH_local\data\cage_0908\mouse_0908\20231027\analysis_data';
% dir = 'D:\RH_local\data\cage_0914\mouse_0914\20231027\analysis_data';
save([dir , '\trialStuff.mat'] , 'trialStuff')
disp(['Saved trialStuff to:  ' ,dir]) 


% %%
% Fs = 30;
% f_highPass = 2; %% in hz
% trace = logger.decoder(:,1);
% noise_goal = noiseAmplitude(trace, Fs, f_highPass);
% 
% %%
% % fn_loss = @(x) (noise_goal - noiseAmplitude(trace/2 + x*randn(size(trace, 1), 1), Fs, f_highPass))^2;
% fn_loss = @(x) (noise_goal - noiseAmplitude(addNoise(trace/2, x, Fs, f_highPass), Fs, f_highPass))^2;
% 
% % options = optimset('Display', 'iter', 'MaxIter', 10, 'TolX', 1e-0);
% options = optimset('MaxIter', 10, 'MaxFunEvals', 10, 'TolX', 1e-2);
% 
% opt_noise_val = fminbnd(fn_loss,0,2, options)
% 
% %%
% figure;
% plot(opt_noise_val*randn(size(trace,1), 1) + trace/2)
% hold on
% plot(trace)
% 
% figure;
% plot(safeHighPass(opt_noise_val*randn(size(trace,1), 1) + trace/2, Fs, f_highPass))
% hold on
% plot(safeHighPass(trace, Fs, f_highPass))
% 
% %%
% var(safeHighPass(opt_noise_val*randn(size(trace,1), 1) + trace/2, Fs, f_highPass))
% var(safeHighPass(trace, Fs, f_highPass))
% %%
% figure;
% hold on
% plot(trace)
% plot(fakeCursors_scaled)
%%
% figure;
% hold on
% for ii = [0:0.01:2]
% % plot(ii, fn_loss(ii), '.')
% % fn_loss(ii)
% highpass(x, f_hp, Fs);
% end

%%

% %%
% 
% % Parameters: Scaling optimization
% learning_rate = 0.01;
% 
% thresh_check_avg = inf;
% thresh_check_avg_first10 = inf;
% 
% obj_fun = @(thresh_avg) (thresh_avg - goal_threshAchieved);
% criterion_fun = @(thresh_avg , thresh) abs(obj_fun(thresh_avg)) < thresh;
% 
% % Parameters: Noise optimization
% Fs = 30;
% f_highPass = 2; %% in hz
% noise_goal = noiseAmplitude(fakeDecoderData, Fs, f_highPass);
% % fn_loss = @(x) (noise_goal - noiseAmplitude(addNoise(trace*scale_factor, x, Fs, f_highPass), Fs, f_highPass))^2;
% options_noiseOptimizer = optimset('MaxIter', 10, 'MaxFunEvals', 100, 'TolX', 1e-1);
% 
% 
% 
% % the thing below looks weird because it is weird. It is basically randomly
% % selecting time points from the fakeData cursor trace and then scaling
% % them all by a single constant factor. The scaling factor is optimized
% % over an objective function that makes the average number of threshold
% % crossing events == goal_thresholdAchieved. It does this both for the
% % average of the first 10 frames as well as the average of all of them.
% % There will be situations where it's actually impossible to meet both
% % objectives, so if the wheels are just spinning (set as just a large
% % iteration number), it resamples the fakeCursors and tries again.
% % To troubleshoot, it's useful to look at the loss_rolling curve
% cc = 1;
% loss_rolling = NaN(10000,1);
% while ~(criterion_fun(thresh_check_avg_first10 , 0.01) && criterion_fun(thresh_check_avg , 0.03))
%     if mod(cc,500)==1
%         acceptable_onsets = find(fakeDecoderData(1:end-maxTrialDuration) < 0);
%         acceptable_onsets_shuffled = acceptable_onsets(randperm(length(acceptable_onsets)));
%         clear fakeCursors
%         for ii = 1:maxNumTrials
%             %     acceptable_onsets(ii)
%             fakeCursors(ii,:) = fakeDecoderData(acceptable_onsets_shuffled(ii):acceptable_onsets_shuffled(ii)+maxTrialDuration);
%         end
% %         cc=1;
%         fakeCursors_scaled = fakeCursors;
%     end
% 
%     %     abs(thresh_check_avg_first10 - goal_threshAchieved)
%     thresh_check = max(fakeCursors_scaled,[],2) > threshold;
%     thresh_check_avg = mean(thresh_check);
%     thresh_check_avg_first10 = mean(thresh_check(1:10));
%         
%     loss_rolling(cc) = ((obj_fun(thresh_check_avg_first10))  +  (obj_fun(thresh_check_avg)));
%     
%     fakeCursors_scaled = fakeCursors_scaled * (1 - learning_rate*loss_rolling(cc));
%     
% %     if mod(cc,20)==0
% % %         fn_loss = @(x) (noise_goal - noiseAmplitude(addNoise(fakeCursors_scaled*scale_factor, x, Fs, f_highPass), Fs, f_highPass))^2;
% %         fn_loss = @(x) (noise_goal - noiseAmplitude(addNoise(reshape(fakeDecoderData',1,[])'*scale_factor, x, Fs, f_highPass), Fs, f_highPass))^2;
% %         opt_noise_val = fminbnd(fn_loss,0,2,options_noiseOptimizer);
% %         fakeCursors_scaled = addNoise(reshape(fakeCursors_scaled',1,[])', opt_noise_val, Fs, f_highPass);
% %         fakeCursors_scaled = reshape(fakeCursors_scaled,[],500)';
% % 
% % %         fakeCursors_scaled = addNoise(fakeCursors_scaled, opt_noise_val, Fs, f_highPass);
% %     end
%     
%     if mod(cc,50)==0
%         disp(['working... iter: ' , num2str(cc), ' loss: ', num2str(loss_rolling(cc)), ' frac_thresh: ', num2str(thresh_check_avg), ' frac_threshFirst10: ', num2str(thresh_check_avg_first10)])
%     end
%     cc = cc+1;
% end
% 
% % fakeCursors_beforenoise = fakeCursors_scaled;
% % 
% % scale_factor = nanmean(mean(fakeCursors_scaled ./ fakeCursors));
% % 
% % % noise amplitude optimization
% % Fs = 30;
% % f_highPass = 1; %% in hz
% % trace = logger.decoder(:,1);
% % noise_goal = noiseAmplitude(trace, Fs, f_highPass);
% % 
% % % fn_loss = @(x) (noise_goal - noiseAmplitude(addNoise(trace*scale_factor, x, Fs, f_highPass), Fs, f_highPass))^2
% % 
% % 
% % fn_loss = @(x) (noise_goal - noiseAmplitude(addNoise(trace*scale_factor, x, Fs, f_highPass), Fs, f_highPass))^2;
% % %     (goal_threshAchieved - (sum(max(reshape(addNoise(reshape(fakeCursors_scaled',1,[])', x, Fs, f_highPass), [], 500),[],2)>threshold)/maxNumTrials))^2;
% % 
% % % options = optimset('Display', 'iter', 'MaxIter', 10, 'TolX', 1e-0);
% % options_noiseOptimizer = optimset('MaxIter', 10, 'MaxFunEvals', 100, 'TolX', 1e-2);
% % 
% % opt_noise_val = fminbnd(fn_loss,0,2,options_noiseOptimizer);
% % 
% % fakeCursors_scaled = addNoise(reshape(fakeCursors_scaled',1,[])', opt_noise_val, Fs, f_highPass);
% % fakeCursors_scaled = reshape(fakeCursors_scaled,[],500)';
% % 
% % fakeCursors_afternoise = fakeCursors_scaled;
% % 
% % % final fakeCursors_scaled iter
% % thresh_check_avg = inf;
% % thresh_check_avg_first10 = inf;
% % disp("Optimizing the cursor after noise addition...")
% % cc=1;
% % while ~(criterion_fun(thresh_check_avg_first10 , 0.01) & criterion_fun(thresh_check_avg , 0.03))
% %     thresh_check = max(fakeCursors_scaled,[],2) > threshold;
% %     thresh_check_avg = mean(thresh_check);
% %     thresh_check_avg_first10 = mean(thresh_check(1:10));
% %     
% %     loss_rolling(cc) = ((obj_fun(thresh_check_avg_first10))  +  (obj_fun(thresh_check_avg)));
% %     
% %     fakeCursors_scaled = fakeCursors_scaled * (1 - learning_rate*loss_rolling(cc));
% %     if mod(cc, 100)==0
% %         disp(['post noise scaling. iter: ', num2str(cc), ', loss: ', num2str(loss_rolling(cc))])
% %     end
% %     cc = cc+1;
% % end

%%
function trace_out = addNoise(trace_in, gain, Fs, f_hp)
    trace_out = trace_in + gain * safeHighPass(trace_in, Fs, f_hp);
end

function [v] = noiseAmplitude(x, Fs, f_hp)
    v = var(safeHighPass(x, Fs, f_hp));
end

function [v] = safeHighPass(x, Fs, f_hp)
    x(isnan(x)) = 0;
    v = highpass(x, f_hp, Fs);
%     v = highpass(x, f_hp, Fs, 'ImpulseResponse', 'fir');
end
