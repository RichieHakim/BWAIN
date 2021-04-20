%% BMI Ensemble selection script
% based on program_EnsembleSelection9_manyCells

% directory_refImOld = 'D:\RH_local\data\scanimage data\round 4 experiments\mouse 6.28\20201010';
% % fileName_refImOld = 'refImOld.mat';
% fileName_refImOld = 'baselineStuff.mat';
% % refImOld variable should be named: 'refImOld'

directory_Fall = '\\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Rich\data\res2p\scanimage data\round 4 experiments\mouse 6.28\20201215\suite2p\plane0';
fileName_Fall = 'Fall.mat';
% Fall variable should be named: 'Fall' (from S2p; contains stat file)
load([directory_Fall '\' fileName_Fall]);

% directory_spatialFootprints = directory_refImOld;
% fileName_spatialFootprints = 'spatial_footprints_reg.mat';
% % spatialFootprints variable should be named: 'spatial_footprints'

% directory_cellsToUse = directory_refImOld;
% fileName_cellsToUse = 'cellsToUse.mat';
% % cellsToUse variable should be named: 'cellsToUse'

directory_today = 'D:\RH_local\data\scanimage data\round 4 experiments\mouse 6.28\20201222\baseline';
fileName_movie = 'baseline_00001';

%% Import and downsample movie
frames_totalExpected = 3000;
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

ImportWarning_Waiting_Shown = 0;
ImportWarning_OpenAccess_Shown = 0;
while lastImportedFileIdx < filesExpected
    if ImportWarning_Waiting_Shown == 0
        disp('Looking for files to import')
        ImportWarning_Waiting_Shown = 1;
    end
    
    dirProps = dir([directory_today , '\', fileName_movie, '*.tif']);
    
    if size(dirProps,1) > 0
        fileNames = str2mat(dirProps.name);
        fileNames_temp = fileNames;
        fileNames_temp(:,[1:numel(fileName_movie), end-3:end]) = [];
        fileNums = str2num(fileNames_temp);
        
        if size(fileNames,1) > lastImportedFileIdx
            if fopen([directory_today, '\', fileNames(lastImportedFileIdx+1,:)]) ~= -1
                
                disp(['===== Importing:    ', fileNames(lastImportedFileIdx+1,:), '====='])
                movie_chunk = bigread5([directory_today, '\', fileNames(lastImportedFileIdx+1,:)]);
                
                if ds_preference
                    clear movie_chunk_ds
                    movie_chunk_ds = imresize3(movie_chunk, [size(movie_chunk,1), size(movie_chunk,2), round(size(movie_chunk,3)/ds_factor)]);
                    saveastiff(movie_chunk_ds, [directory_today, '\downsampled\ds_', fileNames(lastImportedFileIdx+1,:)]);
                    if ~exist('movie_all_ds')
                        movie_all_ds = movie_chunk_ds;
                    else
                        movie_all_ds = cat(3, movie_all_ds, movie_chunk_ds);
                    end
                end
                
                if ~exist('movie_all')
                    movie_all = movie_chunk;
                else
                    movie_all = cat(3, movie_all, movie_chunk);
                end

                
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
%%
Fs_frameRate = 30; % in Hz
duration_trace = size(movie_all,3) / Fs_frameRate;
duration_trial = 30; % in seconds
baseline_pctile = 30;

frame_height = size(movie_all,1);
frame_width = size(movie_all,2);
% frame_height = 512;
% frame_width = 1024;

paddingForMCRef = 50;
%% Make mean image (meanImForMC)
% basically it just makes a mean image from the end of a movie. It first
% makes a mean image from a bunch of frames from the end of the video. It
% does motion correction on a part of the image. If there is an error its
% probably in the indexing of the subimage used for motion correction. It
% then looks at the plots of the motion correction and throws away any
% frames with motion in it. Then it recalculates a mean image. This results
% in a crisper image for motion correction.

num_framesToUseFromEnd = 3000;
framesToUseFromEnd = [0:num_framesToUseFromEnd-1]; % Index of frames to use starting from the end of the video

framesForMeanImForMC = movie_all(:,:,end-framesToUseFromEnd);
meanIm = mean(framesForMeanImForMC,3);

clear xShifts yShifts
cc = 1;
for ii = 1:size(framesForMeanImForMC,3)
    [xShifts(cc) , yShifts(cc)] = motionCorrection_ROI(framesForMeanImForMC(150:350,400:600,ii) , meanIm(150:350,400:600)); % note the indexing of the sub-region of the image. Speeds up calculation
    cc = cc+1;
end
figure; plot(xShifts)
hold on; plot(yShifts)
figure; imagesc(meanIm)
% set(gca,'CLim', [100 600])
smoothedMotion = smooth(abs(xShifts + 1i*yShifts),10);
figure; plot(smoothedMotion)

% == second iteration of above code with still frames only ==
stillFrames = find((smoothedMotion == mode(smoothedMotion)) .* ([diff(smoothedMotion) ; 1] ==0));

framesForMeanImForMC = movie_all(:,:,end-stillFrames);
meanIm = mean(framesForMeanImForMC,3);

clear xShifts yShifts
cc = 1;
for ii = 1:size(framesForMeanImForMC,3)
    [xShifts(cc) , yShifts(cc)] = motionCorrection_ROI(framesForMeanImForMC(150:350,400:600,ii) , meanIm(150:350,400:600));
    cc = cc+1;
end
figure; plot(xShifts)
hold on; plot(yShifts)
figure; imagesc(meanIm)
% set(gca,'CLim', [100 600])
smoothedMotion = smooth(abs(xShifts + 1i*yShifts),10);
figure; plot(smoothedMotion)

disp(['Number of frames used in mean image:  ' , num2str(numel(smoothedMotion))]) % Make sure you're using enough frames for the calculation
%%

% meanIm = ops.meanImg; % make the reference image the Suite2p output reference image
%% Import and align refIm , spatial_footprints (or Fall.mat) , and cellNumsToUse + cellWeightings

% load([directory_refImOld '\' fileName_refImOld]);
% refImOld = baselineStuff.MC.meanImForMC;
refImOld = ops.meanImg; % make the reference image the Suite2p output reference image


% directory_spatialFootprints = directory_refImOld;
% fileName_spatialFootprints = 'spatial_footprints_reg.mat';
% load([directory_spatialFootprints '\' fileName_spatialFootprints]);

% directory_cellsToUse = directory_refImOld;
% fileName_cellsToUse = 'cellsToUse.mat';
% load([directory_cellsToUse '\' fileName_cellsToUse]);
%% Make fake cellNumsToUse + cellWeightings
% % choosing 100 cells at random (just the first 100 cells from s2p)
cellNumsToUse =     1:1000;
cellWeightings =    rand(1000 , 1);
% cellNumsToUse = output.day2;
% cellWeightings = output.weight;

numCells = length(cellNumsToUse); % in 1-indexed (matlab) indices
%%
spatial_footprints = zeros(numCells , frame_height , frame_width);
spatial_footprints_weighted = zeros(numCells , frame_height , frame_width);
for ii = cellNumsToUse
%     spatial_footprints(ii,:,:) = zeros(size(movie_all,1) , size(movie_all,2));
    for jj = 1:stat{ii}.npix
        spatial_footprints(ii , stat{ii}.ypix(jj)+1 , stat{ii}.xpix(jj)+1) = stat{ii}.lam(jj);
        spatial_footprints_weighted(ii , stat{ii}.ypix(jj)+1 , stat{ii}.xpix(jj)+1) = stat{ii}.lam(jj) .* cellWeightings(ii);
    end
end

figure;
imshowpair(squeeze(max(spatial_footprints , [],1))  ,  ...
    squeeze(max(spatial_footprints_weighted , [],1)), 'montage')

%% Non-rigid registration
tmp_im_fixed = refImOld;
% im_fixed = ops.meanImg;
tmp_im_moving = meanIm;

sigma = 20;

tmp_im_fixed = localnormalize(tmp_im_fixed, sigma , sigma);
tmp_im_moving = localnormalize(tmp_im_moving, sigma , sigma);

%
im_moving_gpu = tmp_im_moving;
im_fixed_gpu = tmp_im_fixed;
[D_field, movingReg] = imregdemons(im_moving_gpu ,im_fixed_gpu , 500 , 'PyramidLevels', 4,...
    'AccumulatedFieldSmoothing',2);
%
figure;
imshowpair(tmp_im_fixed , tmp_im_moving)

figure;
imshowpair(tmp_im_fixed , imwarp(tmp_im_moving , D_field))
%
figure;
imshowpair(D_field(:,:,1) , D_field(:,:,2))

%
% spatial_footprints_all1 = squeeze(max(baselineStuff1.spatial_footprints,[],1));
% spatial_footprints_all = squeeze(max(spatial_footprints,[],1));
% spatial_footprints_all2 = squeeze(max(baselineStuff2.spatial_footprints,[],1));
% figure;
% imagesc(spatial_footprints_all2)

spatial_footprints_warped = spatial_footprints;
spatial_footprints_tall = [];
spatial_footprints_tall_warped = [];
for ii = 1:size(spatial_footprints,1)
    tmp_spatial_footprint= imwarp(squeeze(spatial_footprints(ii,:,:)) , D_field);
    spatial_footprints_warped(ii,:,:) = tmp_spatial_footprint;
    
    nonzeros_ind = find(tmp_spatial_footprint ~= 0);
    [nonzeros_subY , nonzeros_subX] = ind2sub([frame_height, frame_width] , nonzeros_ind);
    nonzeros_lam = tmp_spatial_footprint(nonzeros_ind);
    tmp_tallStuff = [ones(length(nonzeros_ind), 1)*(cellNumsToUse(ii)) , nonzeros_subX , nonzeros_subY , nonzeros_lam];
    spatial_footprints_tall_warped = cat(1, spatial_footprints_tall_warped , tmp_tallStuff);
    
    
    
    tmp_spatial_footprint= squeeze(spatial_footprints(ii,:,:));

    nonzeros_ind = find(tmp_spatial_footprint ~= 0);
    [nonzeros_subY , nonzeros_subX] = ind2sub([frame_height, frame_width] , nonzeros_ind);
    nonzeros_lam = tmp_spatial_footprint(nonzeros_ind);
    tmp_tallStuff = [ones(length(nonzeros_ind), 1)*(cellNumsToUse(ii)) , nonzeros_subX , nonzeros_subY , nonzeros_lam];
    spatial_footprints_tall = cat(1, spatial_footprints_tall , tmp_tallStuff);
        
    if mod(ii,100) == 1
        disp(['warping ROI number:  ' , num2str(ii)])
    end
end

%%
cellWeightings_tall = zeros(size(spatial_footprints_tall,1) , 1);
for ii = 1:numCells
    cellWeightings_tall(spatial_footprints_tall(:,1) == ii) = cellWeightings(ii);
end
cellWeightings_tall_warped = zeros(size(spatial_footprints_tall_warped,1) , 1);
for ii = 1:numCells
    cellWeightings_tall_warped(spatial_footprints_tall_warped(:,1) == ii) = cellWeightings(ii);
end


spatial_footprints_tall_weighted = spatial_footprints_tall;
spatial_footprints_tall_weighted(:,4) = spatial_footprints_tall(:,4) .* cellWeightings_tall;

spatial_footprints_tall_warped_weighted = spatial_footprints_tall_warped;
spatial_footprints_tall_warped_weighted(:,4) = spatial_footprints_tall_warped(:,4) .* cellWeightings_tall_warped;

spatial_footprints_all = zeros(frame_height , frame_width);
spatial_footprints_all(sub2ind([frame_height,frame_width] , spatial_footprints_tall(:,3) , spatial_footprints_tall(:,2))) = spatial_footprints_tall(:,4);

spatial_footprints_all_weighted = zeros(frame_height , frame_width);
spatial_footprints_all_weighted(sub2ind([frame_height,frame_width] , spatial_footprints_tall_weighted(:,3) , spatial_footprints_tall_weighted(:,2))) = spatial_footprints_tall_weighted(:,4);

spatial_footprints_warped_all = zeros(frame_height , frame_width);
spatial_footprints_warped_all(sub2ind([frame_height,frame_width] , spatial_footprints_tall_warped(:,3) , spatial_footprints_tall_warped(:,2))) = spatial_footprints_tall_warped(:,4);

spatial_footprints_warped_weighted_all = zeros(frame_height , frame_width);
spatial_footprints_warped_weighted_all(sub2ind([frame_height,frame_width] , spatial_footprints_tall_warped_weighted(:,3) , spatial_footprints_tall_warped_weighted(:,2))) = spatial_footprints_tall_warped_weighted(:,4);


figure;
imshowpair(spatial_footprints_all .^0.3 , spatial_footprints_warped_weighted_all .^0.3)
figure;
imshow(spatial_footprints_warped_weighted_all ,[])

%% Transofrm coordinate indices
for ii = 1:numel(cellNumsToUse)
    idxBounds_ROI{ii}(1,1) = min(stat{cellNumsToUse(ii)}.xpix)+1; % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
    idxBounds_ROI{ii}(2,1) = max(stat{cellNumsToUse(ii)}.xpix)+1;
    idxBounds_ROI{ii}(1,2) = min(stat{cellNumsToUse(ii)}.ypix)+1;
    idxBounds_ROI{ii}(2,2) = max(stat{cellNumsToUse(ii)}.ypix)+1;
    
    mask_center{ii} = [ mean([idxBounds_ROI{ii}(1,1) , idxBounds_ROI{ii}(2,1)])  ,  mean([idxBounds_ROI{ii}(1,2) , idxBounds_ROI{ii}(2,2)]) ];

    spatial_footprints_cropped{ii} = squeeze(spatial_footprints(ii , idxBounds_ROI{ii}(1,2):idxBounds_ROI{ii}(2,2) , idxBounds_ROI{ii}(1,1):idxBounds_ROI{ii}(2,1)));
end

 %% Simulation
% % figure; plot(logger_output(:,28), 'LineWidth', 1.5)
% 
% % threshVals = [2.0:.3:3.5];
% threshVals = [2.5];
% % scale_factors = 1./std(dFoF_roi(:,cellNumsToUse));
% 
% duration_trial = 240; % I'm making the trials long to avoid overfitting to the ITIs.
% duration_total = size(F_roi,1);
% clear total_rewards_per30s currentCursor currentCSThresh currentCSQuiescence currentCETrial currentCEReward currentCETimeout
% cc = 1;
% for ii = threshVals
%     tic
%     disp(['testing threshold value:  ', num2str(ii)])
%     
%     for jj = 1:size(F_roi,1)
%         if jj == 1
%             startSession_pref = 1;
%         else
%             startSession_pref = 0;
%         end
%         
%         %         [logger_output(jj,:)] = BMI_trial_simulation(F_roi(jj,cellNumsToUse),startSession_pref, scale_factors, ii, duration_total, duration_trial);
%         [ currentCursor(jj) , currentCSThresh(jj), currentCSQuiescence(jj), currentCETrial(jj),  currentCEReward(jj), currentCETimeout(jj) ]...
%             = BMI_trial_simulation2(F_roi(jj,:),startSession_pref, scale_factors, ii, duration_total, duration_trial, Ensemble_group);
%     end
%     %         total_rewards_perMin(cc) = sum(all_rewards)/(duration_total/30);
%     %     disp(['total rewards per 30s at threshold = ' , num2str(ii) , ' is:   ', num2str(total_rewards_per30s(cc))]);
%     %     figure; plot([all_rewards.*1.5, all_cursor, baselineState.*1.1, baselineHoldState*0.8, thresholdState.*0.6])
%     toc
%     cc = cc+1;
%     
%     % figure;
%     % plot(threshVals, total_rewards_per30s)
%     % figure; plot(logger_output(:,33:36) .* repmat(scale_factors,size(logger_output,1),1))
%     figure; plot(currentCursor) %cursor
%     hold on; plot(currentCSThresh) %thresh
%     % hold on; plot(currentCSQuiescence) %wait for quiescence
%     hold on; plot(currentCETrial) %trial
%     hold on; plot(currentCEReward, 'LineWidth',2) % rewards
%     hold on; plot([0 numel(currentCEReward)] , [threshVals(end) threshVals(end)])
%     rewardToneHold_diff = diff(currentCEReward);
%     % timeout_diff = diff(logger_output(:,23));
%     numRewards = sum(rewardToneHold_diff(rewardToneHold_diff > 0.5));
%     numRewardsPerMin = numRewards / (size(F_roi,1) / (Fs_frameRate * 60));
%     disp(['For Threshold  =  ', num2str(ii) , '  total rewards: ', num2str(numRewards) , ' , rewards per min: ' , num2str(numRewardsPerMin) ])
%     % numTimeouts = sum(timeout_diff(timeout_diff > 0.5))
%     % numRewards/(numRewards+numTimeouts)
% end

%%
refIm = meanIm;
refIm = single(refIm);
crop_factor = 5;

length_x = size(refIm,2);
length_y = size(refIm,1);
middle_x = size(refIm,2)/2;
middle_y = size(refIm,1)/2;

indRange_y_crop = [round(middle_y - length_y/crop_factor) , round(middle_y + length_y/crop_factor) ];
indRange_x_crop = [round(middle_x - length_y/crop_factor) , round(middle_x + length_y/crop_factor) ];

refIm_crop = refIm(indRange_y_crop(1) : indRange_y_crop(2) , indRange_x_crop(1) : indRange_x_crop(2)) ;

refIm_crop_conjFFT = conj(fft2(refIm_crop));
refIm_crop_conjFFT_shift = fftshift(refIm_crop_conjFFT);

size(refIm_crop_conjFFT_shift,1);
if mod(size(refIm_crop_conjFFT_shift,1) , 2) == 0
    print('RH WARNING: y length of refIm_crop_conjFFT_shift is even. Something is very wrong')
end
if mod(size(refIm_crop_conjFFT_shift,2) , 2) == 0
    print('RH WARNING: x length of refIm_crop_conjFFT_shift is even. Something is very wrong')
end

refIm_crop_conjFFT_shift_centerIdx = ceil(size(refIm_crop_conjFFT_shift)/2);

figure;
imagesc(refIm)
figure;
imagesc(refIm_crop)
figure;
imagesc(log(abs(refIm_crop_conjFFT_shift)))

% baselineStuff.motionCorrectionRefImages.refIm_conjFFT_padded = refIm_conjFFT_padded;
%%
clear baselineStuff
% baselineStuff.threshVals = threshVals;
% baselineStuff.numRewards = numRewards;
% baselineStuff.numRewardsPerMin = numRewardsPerMin;

% baselineStuff.F_roi = F_roi;
% baselineStuff.dFoF_roi = dFoF_roi;
baselineStuff.cellNumsToUse = cellNumsToUse;
baselineStuff.directory = directory_today;
baselineStuff.file_baseName = fileName_movie;
baselineStuff.frames_totalExpected = frames_totalExpected;
baselineStuff.frames_perFile = frames_perFile;
baselineStuff.Fs_frameRate = Fs_frameRate;
% baselineStuff.duration_trace = duration_trace;
% baselineStuff.duration_trial = duration_trial;
% baselineStuff.baseline_pctile = baseline_pctile;
% baselineStuff.scale_factors = scale_factors;
% % baselineStuff.ensemble_assignments = Ensemble_group;


% baselineStuff.MC.framesForMeanImForMC = framesForMeanImForMC;
baselineStuff.MC.meanIm = meanIm;
baselineStuff.MC.meanImForMC_crop = refIm_crop;
baselineStuff.MC.meanImForMC_crop_conjFFT_shift = refIm_crop_conjFFT_shift;
baselineStuff.MC.refIm_crop_conjFFT_shift_centerIdx = refIm_crop_conjFFT_shift_centerIdx;
baselineStuff.MC.indRange_y_crop = indRange_y_crop;
baselineStuff.MC.indRange_x_crop = indRange_x_crop;

baselineStuff.ROIs.idxBounds_ROI = idxBounds_ROI;
baselineStuff.ROIs.mask_center = mask_center;

% baselineStuff.ROIs.spatial_footprints = spatial_footprints; % very slow and big file
% baselineStuff.ROIs.spatial_footprints_warped = spatial_footprints_warped; % very slow and big file
% baselineStuff.ROIs.spatial_footprints_cropped = spatial_footprints_cropped;

baselineStuff.ROIs.spatial_footprints_tall = spatial_footprints_tall;
baselineStuff.ROIs.spatial_footprints_tall_warped = spatial_footprints_tall_warped;
baselineStuff.ROIs.spatial_footprints_tall_weighted = spatial_footprints_tall_weighted;
baselineStuff.ROIs.spatial_footprints_tall_warped_weighted = spatial_footprints_tall_warped_weighted;

baselineStuff.ROIs.spatial_footprints_all = spatial_footprints_all;
baselineStuff.ROIs.spatial_footprints_warped_all = spatial_footprints_warped_all;
baselineStuff.ROIs.spatial_footprints_warped_weighted_all = spatial_footprints_warped_weighted_all;
baselineStuff.ROIs.spatial_footprints_all = spatial_footprints_all;

baselineStuff.ROIs.cellWeightings = cellWeightings;
baselineStuff.ROIs.cellWeightings_tall = cellWeightings_tall;
baselineStuff.ROIs.cellWeightings_tall_warped = cellWeightings_tall_warped;

% baselineStuff.paddingForMCRef = paddingForMCRef;
% % baselineStuff.mask_ROI = mask_ROI;
% % baselineStuff.mask_ROI_cropped = mask_ROI_cropped;
% % baselineStuff.mask_ROI_directCells = mask_ROI_directCells;

%%
% [baselineStuff , motionCorrectionRefImages] = make_motionCorrectionRefImages2(baselineStuff, paddingForMCRef); % second arg is num of pixels to pad sides of ROI with
% baselineStuff.motionCorrectionRefImages = motionCorrectionRefImages;
%%
baselineStuff.framesForMeanImForMC = [];
save([directory_today, '\baselineStuff'], 'baselineStuff','-v7.3')
% save(['F:\RH_Local\Rich data\scanimage data\mouse 1.31\baselineStuff'], 'baselineStuff')
% save([directory, '\motionCorrectionRefImages'], 'motionCorrectionRefImages')
