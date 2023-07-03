%% BMI Ensemble selection script

%% Load weights and baseline Movies
% 01/29/2023

directory_today = 'D:\RH_local\data\cage_0403\mouse_0403L\20230702\scanimage_data\baseline';
% directory_today = 'D:\RH_local\data\cage_0403\mouse_0403R\20230702\scanimage_data\baseline';
% fileName_movie = 'exp_00';
fileName_movie = 'baseline_00';

% Load Weights
% Should be in day N-1 or day 0 folder
directory_weights = 'D:\RH_local\data\cage_0403\mouse_0403L\20230702\analysis_data\day0_analysis';
% directory_weights = 'D:\RH_local\data\cage_0403\mouse_0403R\20230702\analysis_data\day0_analysis';
fileName_weights = 'weights_day0.mat';
load([directory_weights '\' fileName_weights]);
%% Transpose factors
% 20230312 factors: (n_neurons, n_components)
disp(['Loaded Factors shape :    ', num2str(size(factors))])
if size(factors, 2) > size(factors, 1)
    disp(['Transposing factors. Factors shape should be n_neurons * n_components'])
    factors = factors.';
end
%% Important Parameters
% ONLINE BMI PARAMETER SETTINGS: BLOCK SEQUENCE. HARDCODED (Sorry!)
block_sequence = struct();
block_sequence.blockNum_cap = 1; % MAX number of blocks in exp

block_sequence.blocks(1).cursor_to_use = 1; % call baselineStuff.cursors()
block_sequence.blocks(1).block_timecap = 60*60; % In seconds. Shift decoder factor after THIS much time spent
block_sequence.blocks(1).block_rewardcap = 250; % ADJUSTABLE: Shift decoder factor after THIS number of rewards

% block_sequence.blocks(2).cursor_to_use = 2; % call baselineStuff.cursors()
% block_sequence.blocks(2).block_timecap = 60*45; % In seconds. Shift decoder factor after THIS much time spent
% block_sequence.blocks(2).block_rewardcap = 100; % ADJUSTABLE: Shift decoder factor after THIS number of rewards
% 
% block_sequence.blocks(3).cursor_to_use = 1; % call baselineStuff.cursors()
% block_sequence.blocks(3).block_timecap = 60*30; % In seconds. Shift decoder factor after THIS much time spent
% block_sequence.blocks(3).block_rewardcap = 50; % ADJUSTABLE: Shift decoder factor after THIS number of rewards
% 
% block_sequence.blocks(4).cursor_to_use = 1; % call baselineStuff.cursors()
% block_sequence.blocks(4).block_timecap = 60*30; % In seconds. Shift decoder factor after THIS much time spent
% block_sequence.blocks(4).block_rewardcap = 50; % ADJUSTABLE: Shift decoder factor after THIS number of rewards


% ONLINE BMI PARAMETER SETTINGS: DECODER DEFINITION. HARDCODED (Sorry!)
cursors = struct();

% mouse_0322R_2ndFactorSpace
cursors(1).factor_to_use = 2;
cursors(1).angle_power = 2;
cursors(1).threshold_reward     = 1.5;
cursors(1).thresh_quiescence_cursorDecoder = 0.15;
cursors(1).thresh_quiescence_cursorMag = 0;  
cursors(1).win_smooth_cursor    = 1; % smoothing window (in frames)
cursors(1).bounds_cursor        = [-cursors(1).threshold_reward , cursors(1).threshold_reward *1.5];
cursors(1).range_freqOutput     = [1000 18000]; % this is set in the teensy code (only here for logging purposes)
cursors(1).voltage_at_threshold = 3.1; % this will be the maximum output voltage ([0:voltage_at_threshold])

% % mouse_0315N
% cursors(1).factor_to_use = 1;
% cursors(1).angle_power = 2;
% cursors(1).threshold_reward     = 1.5;
% cursors(1).thresh_quiescence_cursorDecoder = 0.15;
% cursors(1).thresh_quiescence_cursorMag = 0;  
% cursors(1).win_smooth_cursor    = 1; % smoothing window (in frames)
% cursors(1).bounds_cursor        = [-cursors(1).threshold_reward , cursors(1).threshold_reward *1.5];
% cursors(1).range_freqOutput     = [1000 18000]; % this is set in the teensy code (only here for logging purposes)
% cursors(1).voltage_at_threshold = 3.1; % this will be the maximum output voltage ([0:voltage_at_threshold])

% Choose PCn as decoder dimension: visualization purpose
factor_to_visualize = cursors(1).factor_to_use; % 1-indexed

% idx_zeroOut = [[215, 339];[798, 899]];  %% [[y1, y2];, [x1, x2]] OR NaN
idx_zeroOut = NaN;  %% [[y1, y2];, [x1, x2]] OR NaN

%% Load calcium trace

directory_zstack = 'D:\RH_local\data\cage_0322\mouse_0322R\20230502\analysis_data';
% directory_zstack = 'D:\RH_local\data\cage_0315\mouse_0315N\20230423\analysis_data\';


% stack_beforeWarp = load([directory_zstack , '\stack.mat']);
stack_beforeWarp = load([directory_zstack , '\stack_sparse.mat']);

%% Import and downsample movie
frames_totalExpected = 4000;
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
%                 movie_chunk = bigread5([directory_today, '\', fileNames(lastImportedFileIdx+1,:)]);
                disp([directory_today, '\', fileNames(lastImportedFileIdx+1,:)])
                reader = ScanImageTiffReader([directory_today, '\', fileNames(lastImportedFileIdx+1,:)]);
                movie_chunk = permute(reader.data(),[2,1,3]);
                
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

py.importlib.import_module('bph.motion_correction');

num_framesToUseFromEnd = 4000;
framesToUseFromEnd = [0:num_framesToUseFromEnd-1]; % Index of frames to use starting from the end of the video

framesForMeanImForMC = movie_all(:,:,end-framesToUseFromEnd);
meanIm = mean(framesForMeanImForMC,3);

% Initialize the shifter class
shifter = py.bph.motion_correction.Shifter_rigid('cpu');
% shifter.make_mask(py.tuple(int64([frame_height, frame_width])), py.tuple([1/64, 1/4]), 3);
% shifter.make_mask(py.tuple(int64([512, 512])), py.tuple([1/64, 1/4]), 3);
shifter.make_mask(py.tuple(int64([256, 256])), py.tuple([1/64, 1/4]), 3);
shifter.preprocess_template_images(gather(single(meanIm(129:256+128, 385:512+128))), py.int(0));

out = shifter.find_translation_shifts(gather(permute(framesForMeanImForMC(129:256+128, 385:512+128, :), [3,1,2])), py.int(0));  %% 0-indexed
shifts_yx          = single(int32(out{1}.numpy()));
% shifts_yx          = single(int16(out{1}.numpy()));
yShifts     = shifts_yx(:,1);
xShifts     = shifts_yx(:,2);
cc = single(out{2}.numpy());

figure; plot(xShifts)
hold on; plot(yShifts)
figure; imagesc(meanIm)
smoothedMotion = smooth(abs(xShifts + 1i*yShifts),10);
figure; plot(smoothedMotion)

% == second iteration of above code with still frames only ==
stillFrames = find((smoothedMotion == mode(smoothedMotion)) .* ([diff(smoothedMotion) ; 1] ==0));

framesForMeanImForMC = movie_all(:,:,end-stillFrames);
meanIm = mean(framesForMeanImForMC,3);

out = shifter.find_translation_shifts(gather(permute(framesForMeanImForMC(129:256+128, 385:512+128, :), [3,1,2])), py.int(0));  %% 0-indexed
shifts_yx          = single(int32(out{1}.numpy()));
% shifts_yx          = single(int16(out{1}.numpy()));
yShifts     = shifts_yx(:,1);
xShifts     = shifts_yx(:,2);
cc = single(out{2}.numpy());

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

% % load([directory_refImOld '\' fileName_refImOld]);
% % refImOld = baselineStuff.MC.meanImForMC;
% refImOld = single(ops.refImg); % make the reference image the Suite2p output reference image
% 
% 
% % directory_spatialFootprints = directory_refImOld;
% % fileName_spatialFootprints = 'spatial_footprints_reg.mat';
% % load([directory_spatialFootprints '\' fileName_spatialFootprints]);
% 
% % directory_cellsToUse = directory_refImOld;
% % fileName_cellsToUse = 'cellsToUse.mat';
% % load([directory_cellsToUse '\' fileName_cellsToUse]);

refImOld = single(refImg);
%% Make cellNumsToUse + cellWeightings using PC1 and iscell
% % make dFoF
% 
% % iscell_new = logical(iscell(:,2)>0.02);
% iscell_new = logical(iscell(:,1));
% 
% % F_new = F(logical(iscell_new) , :) - 0.7*Fneu(logical(iscell_new) , :);
% F_new = F(logical(iscell_new) , :);
% F_baseline = prctile(F_new,30,2);
% % F_baseline(F_baseline<1) = inf;
% 
% dFoF = (F_new - F_baseline) ./ F_baseline;
% dFoF_meanSub = dFoF - mean(dFoF,2);
% dFoF_zscore = zscore(dFoF,[],2);
% 
% pca_input = dFoF_zscore;
% pca_input(isnan(pca_input)) = 0;
% [coeff,score,latent,tsquared,explained,mu] = pca(dFoF_zscore);
% % [coeff,score,latent,tsquared,explained,mu] = pca(dFoF_meanSub);
% 
% cellNumsToUse =     find(iscell_new);
% cellWeightings =    score(:,1);
% cellWeightings(isnan(cellWeightings)) = 0;
% 
% cell_size_max = 165; % in pixels
% numCells = length(cellNumsToUse); % in 1-indexed (matlab) indices
% disp(['Using ' num2str(numCells) ' ROIs'])
% figure;
% plot(coeff(:,1))
% 
% % y = coeff(:,1);
% % X = dFoF_meanSub';
% % X(isnan(X))=0;
% % % theta = inv(X' * X) * X' * y;
% % theta = mvregress(X,y,'algorithm','cwls');
% % % [Mdl,FitInfo] = fitrlinear(X,smooth(y,10), 'IterationLimit' , 10000 , 'BetaTolerance' , 1e-7 , 'DeltaGradientTolerance' , 0.001...
% % %     , 'NumCheckConvergence' , 1000 , 'PassLimit' , 1000);
% % % y_output = X * Mdl.Beta;
% % y_output = X * theta;
% % 
% % hold on;
% % plot(y_output)
% % R = corrcoef(coeff(:,1) , y_output);
% % title(['R = ' , num2str(R(1,2))])
% %%
% iscell_new2 = logical(iscell(:,2)>0.003);
% 
% % F_new = F(logical(iscell_new) , :) - 0.7*Fneu(logical(iscell_new) , :);
% F_new2 = F(logical(iscell_new2) , :);
% F_baseline2 = prctile(F_new2,30,2);
% % F_baseline(F_baseline<1) = inf;
% 
% dFoF2 = (F_new2 - F_baseline2) ./ F_baseline2;
% dFoF_meanSub2 = dFoF2 - mean(dFoF2,2);
% dFoF_zscore2 = zscore(dFoF2,[],2);
% 
% score_new = coeff(:,1)' * dFoF_zscore2';
% coeff2 = score_new * dFoF_zscore2;
% 
% cellNumsToUse =     find(iscell_new2);
% cellWeightings =    score_new';
% cellWeightings(isnan(cellWeightings)) = 0;
% cell_size_max = 165; % in pixels
% numCells = length(cellNumsToUse); % in 1-indexed (matlab) indices
% disp(['Using ' num2str(numCells) ' ROIs'])
% 
% figure;
% plot(coeff2)

%% Make fake cellNumsToUse + cellWeightings
% % % choosing 100 cells at random (just the first 100 cells from s2p)
% cellNumsToUse =     1:1000;
% cellWeightings =    rand(1000 , 1);
% % cellNumsToUse = output.day2;
% % cellWeightings = output.weight;
% 
% cell_size_max = 165; % in pixels
% 
% numCells = length(cellNumsToUse); % in 1-indexed (matlab) indices
%% cellNumsToUse + cellWeightings
% cellNumsToUse   = find(iscell_custom);
% % cellNumsToUse   = 1:100;
% 
% % cellWeightings  = weights(iscell_custom);
% % cellWeightings  = rand(length(cellNumsToUse),1);
% 
% % numCells = length(cellNumsToUse); % in 1-indexed (matlab) indices
% % numCells = length(factors(:, factor_to_use)'); % in 1-indexed (matlab) indices

% Normalize factor space, just to make cursor calculation easy
factor_space = factors ./ vecnorm(factors);

cellWeightings = squeeze(factor_space(:, cursors(1).factor_to_use));

numCells = size(factors, 1);

cell_size_max = 300; % in pixels
neuropil_size_max = 3000; % in pixels

%% make weighted footprints
spatial_footprints_dense = zeros(numCells , frame_height , frame_width);
spatial_footprints_dense_weighted = zeros(numCells , frame_height , frame_width);
Fneu_masks_dense = zeros(numCells , frame_height, frame_width );
% Fneu_masks_dense_weighted = zeros(numCells , frame_height, frame_width);

tmpcoo = spatialFootprints_COO_1idx;
spatialFootprints = sparse(double(tmpcoo.row), double(tmpcoo.col), double(tmpcoo.data), double(tmpcoo.shape(1)), double(tmpcoo.shape(2)));
tmpcoo = neuropilMasks_COO_1idx;
neuropilMasks = sparse(double(tmpcoo.row), double(tmpcoo.col), double(tmpcoo.data), double(tmpcoo.shape(1)), double(tmpcoo.shape(2)));

for ii = 1:size(spatialFootprints, 1)

    sf_tmp = spatialFootprints(ii,:);
    scale_factor = 1/sum(nonzeros(sf_tmp));
    
    spatial_footprints_dense(ii, :, :) = reshape(spatialFootprints(ii,:), [frame_width, frame_height])' * scale_factor;
    spatial_footprints_dense_weighted(ii, :, :) = reshape(spatialFootprints(ii,:), [frame_width, frame_height])' * scale_factor * cellWeightings(ii);
    Fneu_masks_dense(ii, :, :) = reshape(neuropilMasks(ii,:), [frame_width, frame_height])';
%     disp(ii);
end

% %%
% for ii = 1:length(cellNumsToUse)
%     lam = stat{cellNumsToUse(ii)}.lam;
%     lam = lam / sum(lam);
%     for jj = 1:stat{cellNumsToUse(ii)}.npix
%         spatial_footprints(ii , stat{cellNumsToUse(ii)}.ypix(jj)+1 , stat{cellNumsToUse(ii)}.xpix(jj)+1) = lam(jj);
%     end
%     Fneu_masks(ii, stat{1,cellNumsToUse(ii)}.neuropil_mask) = 1;
% end
% Fneu_masks = permute(Fneu_masks, [1 3 2]);

% only used if idx_zeroOut is specified above
if ~isnan(idx_zeroOut)
    mask_zeroOut = zeros(frame_height, frame_width);
    mask_zeroOut(idx_zeroOut(1,1): idx_zeroOut(1,2), idx_zeroOut(2,1): idx_zeroOut(2,2)) = 1;

    ROIs_toZeroOut = squeeze(squeeze(max(max(spatial_footprints_dense .* reshape(mask_zeroOut, 1, frame_height, frame_width), [],2), [],3))) > 0;
    cellWeightings = factors(:, factor_to_visualize)' .* ~ROIs_toZeroOut';
else
    cellWeightings = factors(:, factor_to_visualize)';
end

% % just for visualization purposes
% for ii = 1:length(cellNumsToUse)
%     lam = stat{cellNumsToUse(ii)}.lam;
%     lam = lam / sum(lam);
%     for jj = 1:stat{cellNumsToUse(ii)}.npix
%         spatial_footprints_dense_weighted(ii , stat{cellNumsToUse(ii)}.ypix(jj)+1 , stat{cellNumsToUse(ii)}.xpix(jj)+1) = lam(jj) .* cellWeightings(ii);
%     end
% end


% % just for visualization purposes
% for ii = 1:length(cellNumsToUse)
%     lam = stat{cellNumsToUse(ii)}.lam;
%     lam = lam / sum(lam);
%     for jj = 1:stat{cellNumsToUse(ii)}.npix
%         spatial_footprints_dense_weighted(ii , stat{cellNumsToUse(ii)}.ypix(jj)+1 , stat{cellNumsToUse(ii)}.xpix(jj)+1) = lam(jj) .* cellWeightings(ii);
%     end
% end

%%
figure;
imagesc(squeeze(max(spatial_footprints_dense , [],1)))

%%
figure;
imshowpair(squeeze(max(spatial_footprints_dense , [],1))  ,  ...
    squeeze(max(spatial_footprints_dense_weighted , [],1)), 'montage')
%%
figure;
imagesc(squeeze(max(spatial_footprints_dense_weighted , [],1)))
%% make weighted footprints
% spatial_footprints = permute(h5read(path_spatialFootprints, '/spatial_footprints'), [3,2,1]);
% 
% dims_sf = size(spatial_footprints);
% dims_sf_frame = dims_sf(2:end);
% spatial_footprints_weighted = spatial_footprints .* repmat(reshape(cellWeightings, [size(cellWeightings,2), 1,1]), [1, dims_sf_frame]);
% 
% figure;
% imshowpair(squeeze(sum(spatial_footprints, 1))  ,  ...
%     squeeze(sum(spatial_footprints_weighted, 1)), 'montage')
%%
% 
% spatial_footprints = zeros(numCells , frame_height , frame_width);
% spatial_footprints_weighted = zeros(numCells , frame_height , frame_width);
% for ii = 1:length(cellNumsToUse)
%     %     spatial_footprints(ii,:,:) = zeros(size(movie_all,1) , size(movie_all,2));
%     for jj = 1:stat{cellNumsToUse(ii)}.npix
%         spatial_footprints(ii , stat{cellNumsToUse(ii)}.ypix(jj)+1 , stat{cellNumsToUse(ii)}.xpix(jj)+1) = stat{cellNumsToUse(ii)}.lam(jj);
%         spatial_footprints_weighted(ii , stat{cellNumsToUse(ii)}.ypix(jj)+1 , stat{cellNumsToUse(ii)}.xpix(jj)+1) = stat{cellNumsToUse(ii)}.lam(jj) .* cellWeightings(ii);
%     end
% end
% figure;
% imshowpair(squeeze(max(spatial_footprints , [],1))  ,  ...
%     squeeze(max(spatial_footprints_weighted , [],1)), 'montage')
%% Non-rigid registration
% tmp_im_fixed = refImOld;
% % im_fixed = ops.meanImg;
% tmp_im_moving = meanIm;

tmp_im_fixed = meanIm;
% im_fixed = ops.meanImg;
tmp_im_moving = refImOld;

sigma = 20;

tmp_im_fixed = localnormalize(tmp_im_fixed, sigma , sigma);
tmp_im_moving = localnormalize(tmp_im_moving, sigma , sigma);

%
[D_field, movingReg] = imregdemons(tmp_im_moving ,tmp_im_fixed , 500 , 'PyramidLevels', 4,...
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

%% Warp the zstack images to the current session
tmp_im_fixed = meanIm;
% im_fixed = ops.meanImg;
% tmp_im_moving = squeeze(stack_beforeWarp.stack.stack_avg(3,:,:));
tmp_im_moving = squeeze(stack_beforeWarp.stack_sparse.stack_avg(3,:,:));


sigma = 20;

tmp_im_fixed = localnormalize(tmp_im_fixed, sigma , sigma);
tmp_im_moving = localnormalize(tmp_im_moving, sigma , sigma);

[D_field_zstack, movingReg] = imregdemons(tmp_im_moving , tmp_im_fixed , 500 , 'PyramidLevels', 4,...
    'AccumulatedFieldSmoothing',2);

% stack_warped = stack_beforeWarp.stack;
% for ii = 1:size(stack_beforeWarp.stack.stack_avg, 1)
%     stack_warped.stack_avg(ii,:,:)= imwarp(squeeze(stack_beforeWarp.stack.stack_avg(ii,:,:)) , D_field_zstack);
% end

stack_warped = stack_beforeWarp.stack_sparse;
for ii = 1:size(stack_beforeWarp.stack_sparse.stack_avg, 1)
    stack_warped.stack_avg(ii,:,:)= imwarp(squeeze(stack_beforeWarp.stack_sparse.stack_avg(ii,:,:)) , D_field_zstack);
end

%
figure;
imshowpair(tmp_im_fixed , tmp_im_moving)

figure;
imshowpair(tmp_im_fixed , imwarp(tmp_im_moving , D_field_zstack))
%
figure;
imshowpair(D_field_zstack(:,:,1) , D_field_zstack(:,:,2))

figure;
% imshowpair(squeeze(stack_beforeWarp.stack.stack_avg(3,:,:)), squeeze(stack_warped.stack_avg(3,:,:)))
imshowpair(squeeze(stack_beforeWarp.stack_sparse.stack_avg(3,:,:)), squeeze(stack_warped.stack_avg(3,:,:)))

%% new 'tall' stuff (short and fat now)
cell_sizes = nan(size(spatial_footprints_dense,1),1);
Fneu_mask_sizes = nan(size(Fneu_masks_dense,1),1);
spatial_footprints_warped = spatial_footprints_dense;
Fneu_masks_warped = Fneu_masks_dense;
spatial_footprints_tall = [];
spatial_footprints_tall_warped = [];
Fneu_masks_tall = [];
Fneu_masks_tall_warped = [];
cellNumsToUse = [1:size(spatial_footprints_dense, 1)];
for ii = 1:size(spatial_footprints_dense,1)
    % Warp F / Fneu masks
    tmp_spatial_footprint= imwarp(squeeze(spatial_footprints_dense(ii,:,:)) , D_field);
    tmp_spatial_footprint = tmp_spatial_footprint ./ sum(sum(tmp_spatial_footprint));
    tmp_Fneu_mask = imwarp(squeeze(Fneu_masks_dense(ii,:,:)), D_field);
    tmp_Fneu_mask = tmp_Fneu_mask ./ sum(sum(tmp_Fneu_mask ));
    
    spatial_footprints_warped(ii,:,:) = tmp_spatial_footprint;
    Fneu_masks_warped(ii,:,:) = tmp_Fneu_mask;
    
    % Find nonzeros idx to create tall array: F masks
    nonzeros_ind = find(tmp_spatial_footprint ~= 0);
    cell_sizes(ii) = length(nonzeros_ind);
    [nonzeros_subY , nonzeros_subX] = ind2sub([frame_height, frame_width] , nonzeros_ind);
    nonzeros_lam = tmp_spatial_footprint(nonzeros_ind);
    tmp_tallStuff = [ones(length(nonzeros_ind), 1)*(cellNumsToUse(ii)) , nonzeros_subX , nonzeros_subY , nonzeros_lam];
    tmp_tallStuff = pad_or_crop_tmp_tallStuff(tmp_tallStuff , cell_size_max);
    
    spatial_footprints_tall_warped = cat(1, spatial_footprints_tall_warped , tmp_tallStuff);
    
    % Find nonzeros idx to create tall array: Fneu masks
    Fneu_nonzeros_ind = find(tmp_Fneu_mask ~= 0);
    % remove a random subset of pixels to reduce size of neuropil mask
    Fneu_nonzeros_ind = Fneu_nonzeros_ind(randperm(length(Fneu_nonzeros_ind)));
    Fneu_nonzeros_ind = Fneu_nonzeros_ind(1:min(neuropil_size_max, length(Fneu_nonzeros_ind)));
    % finish with the indexing (same as spatial_footprints)
    Fneu_mask_sizes(ii) = length(Fneu_nonzeros_ind);
    [Fneu_nonzeros_subY , Fneu_nonzeros_subX] = ind2sub([frame_height, frame_width] , Fneu_nonzeros_ind);
    Fneu_nonzeros_lam = ones(size(Fneu_nonzeros_ind));
    tmp_Fneu_tallStuff = [ones(length(Fneu_nonzeros_ind), 1)*(cellNumsToUse(ii)) , Fneu_nonzeros_subX , Fneu_nonzeros_subY , Fneu_nonzeros_lam];
    tmp_Fneu_tallStuff = pad_or_crop_tmp_tallStuff(tmp_Fneu_tallStuff , neuropil_size_max);
    
    Fneu_masks_tall_warped = cat(1, Fneu_masks_tall_warped , tmp_Fneu_tallStuff);
    
    
    % Just Visualization
    tmp_spatial_footprint= squeeze(spatial_footprints_dense(ii,:,:));
    
    nonzeros_ind = find(tmp_spatial_footprint ~= 0);
    [nonzeros_subY , nonzeros_subX] = ind2sub([frame_height, frame_width] , nonzeros_ind);
    nonzeros_lam = tmp_spatial_footprint(nonzeros_ind);
    tmp_tallStuff = [ones(length(nonzeros_ind), 1)*(cellNumsToUse(ii)) , nonzeros_subX , nonzeros_subY , nonzeros_lam];
    tmp_tallStuff = pad_or_crop_tmp_tallStuff(tmp_tallStuff , cell_size_max);
    spatial_footprints_tall = cat(1, spatial_footprints_tall , tmp_tallStuff);
    
    if mod(ii,100) == 1
        disp(['warping ROI number:  ' , num2str(ii)])
    end
end

SPT_idxNaN = isnan(spatial_footprints_tall(:,1));
SPT_warped_idxNaN = isnan(spatial_footprints_tall_warped(:,1));
Fneu_warped_idxNaN = isnan(Fneu_masks_tall_warped(:,1));
%%
figure;
histogram(cell_sizes)
xlabel('cell sizes (pixels)')
disp(['99th percentile of cell sizes is: ' num2str(ceil(prctile(cell_sizes,99))) ' pixels' ])

figure;
histogram(Fneu_mask_sizes)
xlabel('neuropil mask sizes (pixels)')
disp(['99th percentile of neuropil mask sizes is: ' num2str(ceil(prctile(Fneu_mask_sizes,99))) ' pixels' ])

%% Visualization only; variables here are not for online BMI
cellWeightings_tall = zeros(size(spatial_footprints_tall,1) , 1);
for ii = 1:numCells
    cellWeightings_tall(spatial_footprints_tall(:,1) == cellNumsToUse(ii)) = cellWeightings(ii);
end
cellWeightings_tall_warped = zeros(size(spatial_footprints_tall_warped,1) , 1);
for ii = 1:numCells
    cellWeightings_tall_warped(spatial_footprints_tall_warped(:,1) == cellNumsToUse(ii)) = cellWeightings(ii);
end


spatial_footprints_tall_weighted = spatial_footprints_tall;
spatial_footprints_tall_weighted(:,4) = spatial_footprints_tall(:,4) .* cellWeightings_tall;

spatial_footprints_tall_warped_weighted = spatial_footprints_tall_warped;
spatial_footprints_tall_warped_weighted(:,4) = spatial_footprints_tall_warped(:,4) .* cellWeightings_tall_warped;

spatial_footprints_all = zeros(frame_height , frame_width);
spatial_footprints_all(sub2ind([frame_height,frame_width] , spatial_footprints_tall(~SPT_idxNaN,3) , spatial_footprints_tall(~SPT_idxNaN,2))) = spatial_footprints_tall(~SPT_idxNaN,4);

spatial_footprints_all_weighted = zeros(frame_height , frame_width);
spatial_footprints_all_weighted(sub2ind([frame_height,frame_width] , spatial_footprints_tall_weighted(~SPT_idxNaN,3) , spatial_footprints_tall_weighted(~SPT_idxNaN,2))) = spatial_footprints_tall_weighted(~SPT_idxNaN,4);

spatial_footprints_warped_all = zeros(frame_height , frame_width);
spatial_footprints_warped_all(sub2ind([frame_height,frame_width] , spatial_footprints_tall_warped(~SPT_warped_idxNaN,3) , spatial_footprints_tall_warped(~SPT_warped_idxNaN,2))) = spatial_footprints_tall_warped(~SPT_warped_idxNaN,4);

Fneu_masks_warped_all = zeros(frame_height , frame_width);
Fneu_masks_warped_all(sub2ind([frame_height,frame_width] , Fneu_masks_tall_warped(~Fneu_warped_idxNaN,3) , Fneu_masks_tall_warped(~Fneu_warped_idxNaN,2))) = Fneu_masks_tall_warped(~Fneu_warped_idxNaN,4);

spatial_footprints_warped_weighted_all = zeros(frame_height , frame_width);
spatial_footprints_warped_weighted_all(sub2ind([frame_height,frame_width] , spatial_footprints_tall_warped_weighted(~SPT_warped_idxNaN,3) , spatial_footprints_tall_warped_weighted(~SPT_warped_idxNaN,2))) = spatial_footprints_tall_warped_weighted(~SPT_warped_idxNaN,4);

%%
figure;
imshowpair(real(spatial_footprints_all .^1) , real(spatial_footprints_warped_all .^1))
% figure;
% imshow(spatial_footprints_warped_weighted_all ,[])
%%
figure;
imshowpair(real(spatial_footprints_warped_all .^0.1) , real(Fneu_masks_warped_all .^1))
% figure;
% imshow(spatial_footprints_warped_weighted_all ,[])
%% Transform coordinate indices
% for ii = 1:numel(cellNumsToUse)
%     idxBounds_ROI{ii}(1,1) = min(stat{cellNumsToUse(ii)}.xpix)+1; % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
%     idxBounds_ROI{ii}(2,1) = max(stat{cellNumsToUse(ii)}.xpix)+1;
%     idxBounds_ROI{ii}(1,2) = min(stat{cellNumsToUse(ii)}.ypix)+1;
%     idxBounds_ROI{ii}(2,2) = max(stat{cellNumsToUse(ii)}.ypix)+1;
%     
%     mask_center{ii} = [ mean([idxBounds_ROI{ii}(1,1) , idxBounds_ROI{ii}(2,1)])  ,  mean([idxBounds_ROI{ii}(1,2) , idxBounds_ROI{ii}(2,2)]) ];
%     
%     spatial_footprints_cropped{ii} = squeeze(spatial_footprints(ii , idxBounds_ROI{ii}(1,2):idxBounds_ROI{ii}(2,2) , idxBounds_ROI{ii}(1,1):idxBounds_ROI{ii}(2,1)));
% end

%%
refIm = meanIm;
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

refIm_crop_conjFFT_shift_centerIdx = ceil(size(refIm_crop_conjFFT_shift)/2);

figure;
imagesc(refIm)
figure;
imagesc(refIm_crop)
figure;
imagesc(log(abs(refIm_crop_conjFFT_shift)))

% baselineStuff.motionCorrectionRefImages.refIm_conjFFT_padded = refIm_conjFFT_padded;
%% IMPORTANT CHECK
% make sure these are aligned!!!!!

figure; imshowpair(meanIm , spatial_footprints_warped_all.^1  )
figure; imshowpair(meanIm , spatial_footprints_warped_all.^0.3)

disp(['Mean intensity of meanIm: ', num2str(mean(mean(meanIm)))])
%%
figure; imagesc(spatial_footprints_all_weighted*3)
figure; imshowpair(meanIm, spatial_footprints_all_weighted*3)


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
% baselineStuff.MC.meanImForMC_crop = refIm_crop;
% baselineStuff.MC.meanImForMC_crop_conjFFT_shift = refIm_crop_conjFFT_shift;
% baselineStuff.MC.refIm_crop_conjFFT_shift_centerIdx = refIm_crop_conjFFT_shift_centerIdx;
baselineStuff.MC.indRange_y_crop = indRange_y_crop;
baselineStuff.MC.indRange_x_crop = indRange_x_crop;

% baselineStuff.ROIs.idxBounds_ROI = idxBounds_ROI;
% baselineStuff.ROIs.mask_center = mask_center;

% baselineStuff.ROIs.spatial_footprints = spatial_footprints; % very slow and big file
% baselineStuff.ROIs.spatial_footprints_warped = spatial_footprints_warped; % very slow and big file
% baselineStuff.ROIs.spatial_footprints_cropped = spatial_footprints_cropped;

% baselineStuff.ROIs.spatial_footprints_tall = spatial_footprints_tall;
baselineStuff.ROIs.spatial_footprints_tall_warped = spatial_footprints_tall_warped;
% baselineStuff.ROIs.spatial_footprints_tall_weighted = spatial_footprints_tall_weighted;
% baselineStuff.ROIs.spatial_footprints_tall_warped_weighted = spatial_footprints_tall_warped_weighted;

baselineStuff.ROIs.Fneu_masks_tall_warped = Fneu_masks_tall_warped;

baselineStuff.ROIs.spatialFootprints = spatialFootprints;
baselineStuff.ROIs.neuropilMasks = neuropilMasks;

% baselineStuff.ROIs.spatial_footprints_all = spatial_footprints_all;
% baselineStuff.ROIs.spatial_footprints_warped_all = spatial_footprints_warped_all;
% % baselineStuff.ROIs.spatial_footprints_warped_weighted_all = spatial_footprints_warped_weighted_all;
% baselineStuff.ROIs.spatial_footprints_all = spatial_footprints_all;

% baselineStuff.ROIs.SPT_idxNaN = SPT_idxNaN;
% baselineStuff.ROIs.SPT_warped_idxNaN = SPT_warped_idxNaN;

baselineStuff.ROIs.cell_size_max = cell_size_max;
baselineStuff.ROIs.neuropil_size_max = neuropil_size_max;
baselineStuff.ROIs.num_cells = numCells;

baselineStuff.ROIs.factors = factors;
% baselineStuff.ROIs.cellWeightings = cellWeightings;
% baselineStuff.ROIs.cellWeightings_tall = cellWeightings_tall;
% baselineStuff.ROIs.cellWeightings_tall_warped = cellWeightings_tall_warped;

% baselineStuff.paddingForMCRef = paddingForMCRef;
% % baselineStuff.mask_ROI = mask_ROI;
% % baselineStuff.mask_ROI_cropped = mask_ROI_cropped;
% % baselineStuff.mask_ROI_directCells = mask_ROI_directCells;

% baselineStuff.factor_to_use = 4;
baselineStuff.factor_to_use = factor_to_visualize;
baselineStuff.factor_space = factor_space; % (n_neurons, n_components)

% 03/27/2023 Block-trial params structure in baselineStuff
baselineStuff.cursors = cursors;
baselineStuff.block_sequence = block_sequence;

%%
baselineStuff.framesForMeanImForMC = [];
% path_save = [directory_weights, '\baselineStuff_day0'];
path_save = [directory_today, '\baselineStuff.mat'];
% path_save = [directory_today, '\baselineStuff_PC3.mat'];
save(path_save, 'baselineStuff','-v7.3')
disp(['Saved baselineStuff to:  ' ,path_save]) 

%%
path_stack_warped =  [directory_today, '\stack_warped.mat'];
save(path_stack_warped, 'stack_warped');
disp(['Saved warped stack to:  ' ,path_stack_warped]) 
% save(['F:\RH_Local\Rich data\scanimage data\mouse 1.31\baselineStuff'], 'baselineStuff')
% save([directory, '\motionCorrectionRefImages'], 'motionCorrectionRefImages')
%% FUNCTIONS
function tmp_tallStuff = pad_or_crop_tmp_tallStuff(tmp_tallStuff , cell_size_max)
if size(tmp_tallStuff,1) > cell_size_max
    tmp_tallStuff = tmp_tallStuff(1:cell_size_max,:);
end
if size(tmp_tallStuff,1) < cell_size_max
    tmp_tallStuff = cat(1 , tmp_tallStuff , nan(cell_size_max - size(tmp_tallStuff,1) , size(tmp_tallStuff,2)));
end
end