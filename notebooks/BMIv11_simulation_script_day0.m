% Run simulation

% import Fall.mat file
dir_Fall = 'D:\RH_local\data\BMI_cage_1511_4\mouse_1511L\20230201\analysis_data\suite2p\plane0';
fileName_Fall = 'Fall.mat';

path_Fall = [dir_Fall , '\' , fileName_Fall];
Fall = load(path_Fall);

%%
% Import weights computed from Day 0

dir_analysis_day0 = 'D:\RH_local\data\BMI_cage_1511_4\mouse_1511L\20230201\analysis_data';
% fileName_weightsDay0 = 'weights_day0.mat';
% 
% path_weightsDay0 = [dir_analysis_day0 , '\' , fileName_weightsDay0];
% weights_day0 = load(path_weightsDay0);
% 
% clear dir_Fall fileName_Fall path_Fall ...
%     dir_weightsDay0 fileName_weightsDay0 path_weightsDay0

%% Import trialStuff for today's experiment
fileName_trialStuff = 'trialStuff.mat';

dir_analysis_dayN =  'D:\RH_local\data\BMI_cage_1511_4\mouse_1511L\20230201\analysis_data';
path_trialStuff = [dir_analysis_dayN , '\' , fileName_trialStuff];
load(path_trialStuff);

%% Import baselineStuff from Day 0 or N 
% fileName_baselineStuff = 'baselineStuff_day0.mat';
fileName_baselineStuff = 'baselineStuff.mat';


path_baselineStuff = [dir_analysis_day0 , '\' , fileName_baselineStuff];
% path_baselineStuff = [dir_analysis_dayN , '\' , 'baselineStuff.mat'];
load(path_baselineStuff);

% fileName_baselineStuff = 'baselineStuff.mat';
% 
% path_baselineStuff = [dir_analysis_dayN , '\' , fileName_baselineStuff];
% % path_baselineStuff = [dir_analysis_dayN , '\' , 'baselineStuff.mat'];
% load(path_baselineStuff);

%% Import zstack if using image based simulation
fileName_stack = 'stack_warped.mat';

path_stack = [dir_analysis_day0 , '\' , fileName_stack];
load(path_stack);

stack = stack_warped;

%% do the damn thing
% I should probably check that the F values are the same if I do the
% extraction or just use F from s2p

%% Import movie (optional)
% Should be in day N-1 or day 0 folder
directory_movie = 'D:\RH_local\data\BMI_cage_1511_4\mouse_1511L\20230201\scanimage_data\exp';
fileName_movie = 'exp';
% fileName_movie = 'baseline';

frames_totalExpected = 108000;
frames_perFile = 1000;

ds_factor = 5; % downsampling
ds_preference = 0;

% lastImportedFileIdx = 0;
% clear movie_all

if exist('lastImportedFileIdx') == 0
    lastImportedFileIdx = 0;
end
if exist('ind_current') == 0
    ind_current = 0;
end

% % scanAngleMultiplier = [1.7, 1.1];
% scanAngleMultiplier = [1.5, 0.8];
% pixelsPerDegree = [26.2463 , 31.3449] .* scanAngleMultiplier; % expecting 1024 x 512

filesExpected = ceil(frames_totalExpected/frames_perFile);

import ScanImageTiffReader.ScanImageTiffReader
ImportWarning_Waiting_Shown = 0;
ImportWarning_OpenAccess_Shown = 0;
clear movie_chunk;
% clear movie_all;
% while lastImportedFileIdx < filesExpected
for iter_file = 1:filesExpected
    if ImportWarning_Waiting_Shown == 0
        disp('Looking for files to import')
        ImportWarning_Waiting_Shown = 1;
    end
    
    dirProps = dir([directory_movie , '\', fileName_movie, '*.tif']);
    
    if size(dirProps,1) > 0
        fileNames = str2mat(dirProps.name);
        fileNames_temp = fileNames;
        fileNames_temp(:,[1:numel(fileName_movie), end-3:end]) = [];
        fileNums = str2num(fileNames_temp);
        
        if size(fileNames,1) > lastImportedFileIdx
            if fopen([directory_movie, '\', fileNames(lastImportedFileIdx+1,:)]) ~= -1
                
                disp(['===== Importing:    ', fileNames(lastImportedFileIdx+1,:), '====='])
                %                 movie_chunk = bigread5([directory_movie, '\', fileNames(lastImportedFileIdx+1,:)]);
                reader = ScanImageTiffReader([directory_movie, '\', fileNames(lastImportedFileIdx+1,:)]);
                movie_chunk = permute(reader.data(),[2,1,3]);
                
                %                 if ds_preference
                %                     clear movie_chunk_ds
                %                     movie_chunk_ds = imresize3(movie_chunk, [size(movie_chunk,1), size(movie_chunk,2), round(size(movie_chunk,3)/ds_factor)]);
                %                     saveastiff(movie_chunk_ds, [directory_today, '\downsampled\ds_', fileNames(lastImportedFileIdx+1,:)]);
                %                     if ~exist('movie_all_ds')
                %                         movie_all_ds = movie_chunk_ds;
                %                     else
                %                         movie_all_ds = cat(3, movie_all_ds, movie_chunk_ds);
                %                     end
                %                 end
                if ~exist('movie_all')
                    movie_all = zeros(size(movie_chunk,1), size(movie_chunk,2), frames_totalExpected , 'int16');
                    movie_all(:,:, ind_current+1:ind_current+size(movie_chunk,3)) = movie_chunk;
                else
                    movie_all(:,:, ind_current+1:ind_current+size(movie_chunk,3)) = movie_chunk;
                end
                ind_current = ind_current+size(movie_chunk,3);
                
                disp(['Completed import'])
                lastImportedFileIdx = lastImportedFileIdx + 1;
                ImportWarning_Waiting_Shown = 0;
                ImportWarning_OpenAccess_Shown = 0;
                
            else if ImportWarning_OpenAccess_Shown == 0
                    disp('New file found, waiting for access to file')
                    ImportWarning_OpenAccess_Shown = 1;
                end
            end
        end
    end
end

%% concatenate movie_chunks
% disp('Concatenating movie_all')
% movie_all = cell2mat(movie_chunk);
% disp(' == COMPLETE == ')

%%
cellNumsToUse =     baselineStuff.cellNumsToUse;
%% Simulation (new)

% F_double = double(Fall.F);

num_frames = size(movie_all, 3);
num_frames = 1000;
% num_frames = size(F_double, 2);

threshold_reward = 1.7;
threshold_quiescence = 0;
duration_quiescenceHold = 0.5; % in second
% duration_quiescenceHold = 0;

t = NaN(1,num_frames);
for ii = 1:num_frames
% for ii = 1:1000
        tic
%         ii
    if ii<num_frames
%         BMIv11_simulation(F_double(cellNumsToUse,ii)' , ii , baselineStuff , trialStuff, num_frames , threshold_reward , threshold_quiescence, duration_quiescenceHold, num_frames);
        BMIv11_simulation_imageInput(movie_all(:,:,ii) , ii , baselineStuff, trialStuff, stack, num_frames , threshold_reward , threshold_quiescence, num_frames);
    else
%         [logger , logger_valsROIs2 , numRewardsAcquired] = BMIv11_simulation(F_double(cellNumsToUse,ii)' , ii , baselineStuff , trialStuff, num_frames , threshold_reward , threshold_quiescence, duration_quiescenceHold, num_frames);
        [logger , loggerNames , logger_valsROIs,  numRewardsAcquired] = BMIv11_simulation_imageInput(movie_all(:,:,ii) , ii , baselineStuff, trialStuff, stack, num_frames , threshold_reward , threshold_quiescence, num_frames);
    end
        t(ii) = toc;
    if mod(ii,1000)==0 || ii==1 || ii==2
%     if 1
        fprintf('%s\n' , [num2str(ii) , '/' , num2str(num_frames)])
    end
end
% toc
%%
figure;
hold on;
histogram(t*1000, [0.0:0.0002:0.02]*1000);
xlim([0,0.02*1000])
xlabel(['script time (ms)'])
disp(['mean: ', num2str(nanmean(t)), 'std: ', num2str(nanstd(t))])

%%
Fs = 30; % fps
% duration_inMinutes = num_frames / (Fs*60);
duration_inMinutes = ii / (Fs*60);
disp(['num of rewards acquired =  ' , num2str(numRewardsAcquired) , ' rewards'])
reward_rate_per_min = numRewardsAcquired/duration_inMinutes;
disp(['num of rewards acquired =  ' , num2str(reward_rate_per_min) , ' rewards / min'])
disp(['total duration =  ' , num2str(duration_inMinutes) , ' min'])
disp(['total number of trials = ' , num2str(size(logger.trials,1))])

%%
figure; imagesc(logger.timeSeries(:,[2,4,5,7,13,16,19,21,30,31,32])')
figure; plot(logger.timeSeries(:,[2,4,5,7,13,16,19,21,30,31,32]) + [0,2,4,6,8,10,12,14,16,18,20])
reward_times = find(diff(logger.timeSeries((1:num_frames),13))>0.5);
figure; plot(logger.decoder(1:num_frames,4).*logger.timeSeries(1:num_frames,5)/10000)
hold on; plot(logger.decoder(1:num_frames,3))
hold on; plot(logger.timeSeries(1:num_frames,31:32))
hold on; plot([1,length(logger.timeSeries(1:num_frames,1))] , [threshold_reward , threshold_reward])
hold on; plot(reward_times, ones(size(reward_times))*0.2 , '.' , 'MarkerSize' , 20)

%%
% figure; plot((1:num_frames) / 30 , logger.decoder.outputs(1:num_frames,1))
figure; 
% plot(zscore(weights_day0.regression_output.regression_reconstruction))
% hold on
% plot(weights_day0.regression_output.regression_goalSignal-0.55)
plot(logger.decoder(1:num_frames,1))
%% save
save_dir =  'D:\RH_local\data\BMI_cage_1511_4\mouse_1511L\20230125\analysis_data';
save([save_dir , '\' ,'numRewardsAcquired_PC3'] , 'numRewardsAcquired');
save([save_dir , '\' , 'reward_rate_per_min_PC3'] , 'reward_rate_per_min');
save([save_dir , '\' , 'logger_simulation_PC3'] , 'logger');



