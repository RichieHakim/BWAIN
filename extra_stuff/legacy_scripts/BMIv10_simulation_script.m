%% Run simulation

% import Fall.mat file
dir_Fall = 'D:\RH_local\data\scanimage data\round 4 experiments\mouse 11.5\20210121\suite2p\plane0';
fileName_Fall = 'Fall.mat';

path_Fall = [dir_Fall , '\' , fileName_Fall];
Fall = load(path_Fall);
%%

dir_weightsDay0 = 'D:\RH_local\data\scanimage data\round 4 experiments\mouse 11.5\20210121';
fileName_weightsDay0 = 'weights_dayN.mat';

path_weightsDay0 = [dir_weightsDay0 , '\' , fileName_weightsDay0];
weights_day0 = load(path_weightsDay0);

clear dir_Fall fileName_Fall path_Fall ...
    dir_weightsDay0 fileName_weightsDay0 path_weightsDay0
%% do the damn thing
% I should probably check that the F values are the same if I do the
% extraction or just use F from s2p
%% Import movie (optional)
% Should be in day N-1 or day 0 folder
directory_movie = 'D:\RH_local\data\scanimage data\round 4 experiments\mouse 11.5\20210121';
fileName_movie = 'exp_00001';

frames_totalExpected = 60000;
frames_perFile = 1000;

ds_factor = 5; % downsampling
ds_preference = 0;

% lastImportedFileIdx = 0;
% clear movie_all

if exist('lastImportedFileIdx') == 0
    lastImportedFileIdx = 0;
end

% % scanAngleMultiplier = [1.7, 1.1];
% scanAngleMultiplier = [1.5, 0.8];
% pixelsPerDegree = [26.2463 , 31.3449] .* scanAngleMultiplier; % expecting 1024 x 512

filesExpected = ceil(frames_totalExpected/frames_perFile);

import ScanImageTiffReader.ScanImageTiffReader
ImportWarning_Waiting_Shown = 0;
ImportWarning_OpenAccess_Shown = 0;
cc=1; clear movie_chunk;
while lastImportedFileIdx < filesExpected
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
                movie_chunk{:,:,cc} = permute(reader.data(),[2,1,3]);
                
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
                    movie_all = movie_chunk{:,:,cc};
                else
                    movie_all = cat(3, movie_all, movie_chunk{:,:,cc});
                end
                cc=cc+1;
                
                
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
% %% concatenate movie_chunks
% disp('Concatenating movie_all')
% movie_all = cell2mat(movie_chunk);
% disp(' == COMPLETE == ')
%%
cellNumsToUse =     logical(weights_day0.iscell_custom==1);
cellWeightings =    weights_day0.weights';
% cellNumsToUse = output.day2;
% cellWeightings = output.weight;

% cell_size_max = 165; % in pixels

% numCells = length(cellNumsToUse); % in 1-indexed (matlab) indices



%% Simulation (new)

% make raw input vals for pseudo BMI code function

% - first make a rolling F_baseline
% - it should be heavily approximated for speed
% - subsampling can be done both on the number of prctile calls done as well
% as the number of timepoints used in the prctile call
% - same goes for std

% tic
F_double = double(Fall.F);
% num_frames = size(F_double,2);
num_frames = size(movie_all,3);

% num_frames = 30000;
% num_frames = size(movie_all,3);
threshold_reward = 0.08;
threshold_quiescence = 0;

for ii = 1:num_frames
    %     tic
    %     ii
    if ii<num_frames
        %         BMIv10_simulation(F_double(cellNumsToUse,ii)' , ii , cellWeightings , num_frames , threshold_reward , threshold_quiescence);
        BMIv10_simulation_imageInput(movie_all(:,:,ii) , ii , baselineStuff,num_frames , threshold_reward , threshold_quiescence);
        
    else
        %          [logger , logger_valsROIs , numRewardsAcquired] = BMIv10_simulation(F_double(cellNumsToUse,ii)' , ii , cellWeightings , num_frames , threshold_reward , threshold_quiescence);
        [logger , logger_valsROIs,logger_valsROIs_output_zscored,  numRewardsAcquired] = BMIv10_simulation_imageInput(movie_all(:,:,ii) , ii , baselineStuff,num_frames , threshold_reward , threshold_quiescence);
    end
    %     toc
    if mod(ii,10000)==0 | ii==1 | ii==2
        fprintf('%s\n' , [num2str(ii) , '/' , num2str(num_frames)])
    end
end
% toc

Fs = 30; % fps
duration_inMinutes = num_frames / (Fs*60);
disp(['num of rewards acquired =  ' , num2str(numRewardsAcquired) , ' rewards'])
reward_rate_per_min = numRewardsAcquired/duration_inMinutes;
disp(['num of rewards acquired =  ' , num2str(reward_rate_per_min) , ' rewards / min'])
disp(['total duration =  ' , num2str(duration_inMinutes) , ' min'])

%%
figure;
% figure; plot(logger.trial(:,[2])+0.1)
hold on; plot(logger.trial(:,[2,7,13,19]))
reward_times = find(diff(logger.trial(:,13))>0.5);
figure; plot(logger.decoder_outputs(1:num_frames,1))
hold on; plot([1,length(logger.trial(:,1))] , [threshold_reward , threshold_reward])
hold on; plot(reward_times, ones(size(reward_times))*0.2 , '.' , 'MarkerSize' , 20)

%%
% figure; plot((1:num_frames) / 30 , logger.decoder.outputs(1:num_frames,1))
figure; plot(weights_day0.regression_output.regression_reconstruction-0.3)
hold on
% plot(weights_day0.regression_output.regression_goalSignal-0.55)
plot(logger.decoder(1:num_frames,1))
%% save
save_dir = '\\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Rich\data\res2p\scanimage data\round 4 experiments\mouse 11.5\20210122';
save([save_dir , '\' ,'numRewardsAcquired'] , 'numRewardsAcquired');
save([save_dir , '\' , 'reward_rate_per_min'] , 'reward_rate_per_min');
