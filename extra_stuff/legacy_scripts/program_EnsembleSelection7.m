%% BMI Ensemble selection script
%% import baseline movie
directory = 'F:\RH_Local\Rich data\scanimage data\mouse 1.31\20200226';
file_baseName = 'baseline_00001_';

frames_totalExpected = 27300;
frames_perFile = 1000;

ds_factor = 5; % downsampling
ds_preference = 1;

% lastImportedFileIdx = 0;
% clear movie_all

if exist('lastImportedFileIdx') == 0
    lastImportedFileIdx = 0;
end

% scanAngleMultiplier = [1.7, 1.1];
scanAngleMultiplier = [1.5, 0.8];
pixelsPerDegree = [26.2463 , 31.3449] .* scanAngleMultiplier; % expecting 1024 x 512

filesExpected = ceil(frames_totalExpected/frames_perFile);

ImportWarning_Waiting_Shown = 0;
ImportWarning_OpenAccess_Shown = 0;
while lastImportedFileIdx < filesExpected
    if ImportWarning_Waiting_Shown == 0
        disp('Looking for files to import')
        ImportWarning_Waiting_Shown = 1;
    end
    
    dirProps = dir([directory , '\', file_baseName, '*.tif']);
    
    if size(dirProps,1) > 0
        fileNames = str2mat(dirProps.name);
        fileNames_temp = fileNames;
        fileNames_temp(:,[1:numel(file_baseName), end-3:end]) = [];
        fileNums = str2num(fileNames_temp);
        
        if size(fileNames,1) > lastImportedFileIdx
            if fopen([directory, '\', fileNames(lastImportedFileIdx+1,:)]) ~= -1
                
                disp(['===== Importing:    ', fileNames(lastImportedFileIdx+1,:), '====='])
                movie_chunk = bigread5([directory, '\', fileNames(lastImportedFileIdx+1,:)]);
                
                if ds_preference
                    clear movie_chunk_ds
                    movie_chunk_ds = imresize3(movie_chunk, [size(movie_chunk,1), size(movie_chunk,2), round(size(movie_chunk,3)/ds_factor)]);
%                     saveastiff(movie_chunk_ds, [directory, '\downsampled\ds_', fileNames(lastImportedFileIdx+1,:)]);
                end
                
                if ~exist('movie_all')
                    movie_all = movie_chunk;
                else
                    movie_all = cat(3, movie_all, movie_chunk);
                end
                if ~exist('movie_all_ds')
                    movie_all_ds = movie_chunk_ds;
                else
                    movie_all_ds = cat(3, movie_all_ds, movie_chunk_ds);
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

paddingForMCRef = 30;
%% Make mean image (meanImForMC)
% basically it just makes a mean image from the end of a movie. It first
% makes a mean image from a bunch of frames from the end of the video. It
% does motion correction on a part of the image. If there is an error its
% probably in the indexing of the subimage used for motion correction. It
% then looks at the plots of the motion correction and throws away any
% frames with motion in it. Then it recalculates a mean image. This results
% in a crisper image for motion correction.

framesToUseFromEnd = [0:3000]; % Index of frames to use starting from the end of the video

framesForMeanImForMC = movie_all(:,:,end-framesToUseFromEnd);
meanImForMC = mean(framesForMeanImForMC,3);

clear xShifts yShifts
cc = 1;
for ii = 1:size(framesForMeanImForMC,3)
    [xShifts(cc) , yShifts(cc)] = motionCorrection_ROI(framesForMeanImForMC(150:350,400:600,ii) , meanImForMC(150:350,400:600)); % note the indexing of the sub-region of the image. Speeds up calculation
    cc = cc+1;
end
figure; plot(xShifts)
hold on; plot(yShifts)
figure; imagesc(meanImForMC)
% set(gca,'CLim', [100 600])
smoothedMotion = smooth(abs(xShifts + 1i*yShifts),10);
figure; plot(smoothedMotion)

% == second iteration of above code with still frames only ==
stillFrames = find((smoothedMotion == mode(smoothedMotion)) .* ([diff(smoothedMotion) ; 1] ==0));

framesForMeanImForMC = movie_all(:,:,end-stillFrames);
meanImForMC = mean(framesForMeanImForMC,3);

clear xShifts yShifts
cc = 1;
for ii = 1:size(framesForMeanImForMC,3)
    [xShifts(cc) , yShifts(cc)] = motionCorrection_ROI(framesForMeanImForMC(150:350,400:600,ii) , meanImForMC(150:350,400:600));
    cc = cc+1;
end
figure; plot(xShifts)
hold on; plot(yShifts)
figure; imagesc(meanImForMC)
% set(gca,'CLim', [100 600])
smoothedMotion = smooth(abs(xShifts + 1i*yShifts),10);
figure; plot(smoothedMotion)

disp(['Number of frames used in mean image:  ' , num2str(numel(smoothedMotion))]) % Make sure you're using enough frames for the calculation

%% Make SD image
movie_std = std(single(movie_all_ds),[],3);
movie_fano = movie_std ./ mean(movie_all_ds,3);

figure; imagesc(movie_fano)
%% Choose cells
% Import the Fall.mat file from suite2p's output

    criteria1_spikeRate = 5;
    criteria1_Freq = 5;
    criteria2_spikeRate = 20;
    criteria2_Freq = 1;
    
    look_at_nonCells_too_pref = 0;

    cc = 1;
clear F_screen dFoF_screen mask_ROI mask_ROI_cropped scale_factors_screen...
    dFoF_roi_scaled_screen dFoF_roi_scaled_screen_collapsed cellNum_collapsed...
    F_screen_ds dFoF_screen_ds scale_factors_screen_ds dFoF_roi_scaled_screen_ds dFoF_roi_scaled_screen_collapsed_ds
for ii = fliplr(1:size(F,1))
% for ii = fliplr(1:15)
    tmpMinY = min(stat{ii}.ypix)+1;
    tmpMaxY = max(stat{ii}.ypix)+1;
    tmpMinX = min(stat{ii}.xpix)+1;
    tmpMaxX = max(stat{ii}.xpix)+1;
    
    halfWidthOfCell = ceil((tmpMaxX - tmpMinX)/2);
    halfHeightOfCell = ceil((tmpMaxY - tmpMinY)/2);

    YMin_wPadding = tmpMinY - (paddingForMCRef + halfHeightOfCell);
    YMax_wPadding = tmpMaxY + (paddingForMCRef + halfHeightOfCell);
    XMin_wPadding = tmpMinX - (paddingForMCRef + halfWidthOfCell);
    XMax_wPadding = tmpMaxX + (paddingForMCRef + halfWidthOfCell);
    
%     if iscell(ii,1) == 1 || look_at_nonCells_too_pref == 1
    if iscell(ii,1) == 1 || look_at_nonCells_too_pref == 1
        if  sum(spks(ii,1:1000) > criteria1_spikeRate) > criteria1_Freq...
                && sum(spks(ii,1001:end) > criteria1_spikeRate) > criteria1_Freq...
%                 && sum(spks(ii,2001:end) > criteria1_spikeRate) > criteria1_Freq ...
%                 && sum(spks(ii,3001:end) > criteria1_spikeRate) > criteria1_Freq
            if  sum(spks(ii,1:1000) > criteria2_spikeRate) > criteria2_Freq ...
                    && sum(spks(ii,1001:end) > criteria2_spikeRate) > criteria2_Freq ...
%                     && sum(spks(ii,2001:end) > criteria2_spikeRate) > criteria2_Freq ...
%                     && sum(spks(ii,3001:end) > criteria2_spikeRate) > criteria2_Freq
                
                if sum([YMin_wPadding YMax_wPadding XMin_wPadding XMax_wPadding] < 1) == 0    ...
                        &&    sum([XMin_wPadding XMax_wPadding] > size(movie_all,2)) == 0    ...
                        &&    sum([YMin_wPadding YMax_wPadding] > size(movie_all,1)) == 0
                
                mask_ROI{ii} = zeros(size(movie_all,1) , size(movie_all,2));
                for jj = 1:stat{ii}.npix
                    mask_ROI{ii}(stat{ii}.ypix(jj)+1 , stat{ii}.xpix(jj)+1) = stat{ii}.lam(jj)*100;
                end
                mask_ROI_cropped{ii} = mask_ROI{ii}(tmpMinY:tmpMaxY , tmpMinX:tmpMaxX);
                mask_ROI_cropped_int16{ii} = int16(mask_ROI_cropped{ii});

                F_screen(ii,:) = squeeze(sum(sum( double(movie_all(tmpMinY: tmpMaxY , tmpMinX: tmpMaxX, :)) .* repmat(mask_ROI{ii}(tmpMinY: tmpMaxY , tmpMinX: tmpMaxX),1,1,size(movie_all,3)) ,1),2));
                dFoF_screen(ii,:) = (F_screen(ii,:)-prctile(F_screen(ii,:),30)) ./ prctile(F_screen(ii,:),30);
                scale_factors_screen(ii) = 1./std(dFoF_screen(ii,:));
                dFoF_roi_scaled_screen(ii,:) = dFoF_screen(ii,:) .* scale_factors_screen(ii);
                dFoF_roi_scaled_screen_collapsed(:,cc) = dFoF_roi_scaled_screen(ii,:);
                
 % Again but now for the downsampled movie                
                F_screen_ds(ii,:) = squeeze(sum(sum( double(movie_all_ds(tmpMinY: tmpMaxY , tmpMinX: tmpMaxX, :)) .* repmat(mask_ROI{ii}(tmpMinY: tmpMaxY , tmpMinX: tmpMaxX),1,1,size(movie_all_ds,3)) ,1),2));
                dFoF_screen_ds(ii,:) = (F_screen_ds(ii,:)-prctile(F_screen_ds(ii,:),30)) ./ prctile(F_screen_ds(ii,:),30);
                scale_factors_screen_ds(ii) = 1./std(dFoF_screen_ds(ii,:));
                dFoF_roi_scaled_screen_ds(ii,:) = dFoF_screen_ds(ii,:) .* scale_factors_screen_ds(ii);
                dFoF_roi_scaled_screen_collapsed_ds(:,cc) = dFoF_roi_scaled_screen_ds(ii,:);

                cellNum_collapsed(cc) = ii;

                tmpIm = double(ops.meanImg);
                tmpIm2 = double(ops.sdmov);
                tmpIm3 = double(ops.meanImg);
                tmpIm4 = double(ops.max_proj);
                tmpIm5 = cat(3, ops.meanImg/max(ops.meanImg(:)), ops.meanImg/max(ops.meanImg(:)), (ops.meanImg.* mask_ROI{ii})/1);

                fig = figure;
                fig.Position = [10 50 1200 800];
                fig.Name = ['cell #: ' , num2str(ii-1)];
                
                subplot(3,6,1)
                imagesc(mask_ROI_cropped{ii})
                title(['mask cell# ' , num2str(ii-1)])
                
                subplot(3,6,2)
                imagesc(mask_ROI_cropped_int16{ii})
                title(['mask int16'])
                
                subplot(3,6,3)
                imagesc(tmpIm2(tmpMinY:tmpMaxY , tmpMinX:tmpMaxX))
                title(num2str(stat{ii}.aspect_ratio))
                title('sdmov')
                
                subplot(3,6,4)
                imagesc(tmpIm3(tmpMinY+1:tmpMaxY , tmpMinX+1:tmpMaxX))
                title('meanImg')
                
                subplot(3,6,5)
                imagesc(tmpIm4(tmpMinY+1:tmpMaxY , tmpMinX+1:tmpMaxX))
                title('max_proj')
                
                subplot(3,6,6)
                histogram(dFoF_screen(ii,:),1000)
                xlim([-0.2 2])
                title(['std = ' num2str(std(dFoF_screen(ii,:)))])


                subplot(3,6,7:12)
                plot(dFoF_screen(ii,:))
                ylim([-0.2 5])
                
                subplot(3,6,13:15)
                imshow(tmpIm5)
                title('mean Image & ROI')

                subplot(3,6,16:18)
                plot(dFoF_screen(ii,:))
                xlim([1 1000])
                ylim([-0.2 0.5])
                
                cc = cc+1;
                end
            end
        end
    end
% tightfig;
end

fig2=figure;
fig2.Name = ['mean F of all cells'];
plot(mean(F,1))

%% Choose cells
traces_to_use_E1 = input('Input numbers for E1 trace to use:  ');
traces_to_use_E2 = input('Input numbers for E2 trace to use:  ');
cellNumsToUse = [traces_to_use_E1 , traces_to_use_E2]+1;

%% Update ROIs
figure; hold on;
clear F_roi_new dFoF_roi_new
for ii = 1:numel(cellNumsToUse)
    F_roi_new(:,ii) = F_screen(cellNumsToUse(ii),:);
    dFoF_roi_new(:,ii) = dFoF_screen(cellNumsToUse(ii),:);
%     plot((1:size(dFoF_roi_new,1)) / Fs_frameRate,dFoF_roi_new(:,ii))
end
F_roi = F_roi_new; dFoF_roi = dFoF_roi_new;
scale_factors = 1./std(dFoF_roi);
dFoF_roi_scaled = dFoF_roi .* repmat(scale_factors,size(dFoF_roi,1),1);

%% Plot direct cell ROIs
figure; plot(F_roi)
title('F')
figure; plot(dFoF_roi .* repmat(scale_factors,size(dFoF_roi,1),1))
title('Scaled dFoF')
% figure; plot(dFoF_roi )
E1 = mean(dFoF_roi(:,(1:2)) .* repmat(scale_factors(1:2),size(dFoF_roi,1),1),2);
E2 = mean(dFoF_roi(:,(3:4)) .* repmat(scale_factors(3:4),size(dFoF_roi,1),1),2);
cursor = E1-E2;
% figure; plot(E2)
figure; plot(cursor)
% %%
% corr_roi_screen = cov(dFoF_roi_scaled_screen_collapsed) .* (1-eye(size(dFoF_roi_scaled_screen_collapsed,2)));
% 
% test = linkage(dFoF_roi_scaled_screen_collapsed,'complete', 'correlation');
% 
% figure;
% imagesc(test)
% 
% figure; 
% imagesc(corr_roi_screen)
% colorbar
% %%
% cov_roi = cov(dFoF_roi_scaled) .* (1-eye(size(dFoF_roi_scaled,2)));
% figure; 
% imagesc(cov_roi)
% colorbar
% 
% %%
% smoothingFactor = 1000;
% clear dFoF_roi_scaled_screen_collapsed_ds_smooth_meanSub
% for ii = 1:size(dFoF_roi_scaled_screen_collapsed_ds,2)
%     dFoF_roi_scaled_screen_collapsed_ds_smooth_meanSub(:,ii) = ...
%         (dFoF_roi_scaled_screen_collapsed_ds(:,ii) - smooth(dFoF_roi_scaled_screen_collapsed_ds(:,ii), smoothingFactor)) ./ mean(dFoF_roi_scaled_screen_collapsed_ds(:,ii));
% end
% figure; plot(dFoF_roi_scaled_screen_collapsed_ds_smooth_meanSub(:,1:3))
% 
% [coeff,score,latent,tsquared,explained,mu] = pca(dFoF_roi_scaled_screen_collapsed_ds_smooth_meanSub');
% 
% test2 = corr([coeff(:,1:3) , dFoF_roi_scaled_screen_collapsed_ds]);
% 
% figure; imagesc(test2 .* (eye(size(test2,1))==0))
% colorbar
% 
% figure; imagesc(score)
% cellsToUseInE1_collapsed = find(score(:,2) > 20);
% cellsToUseInE1 = cellNum_collapsed(find(score(:,2) > 20));
% 
% dFoF_roi_screen_collapsed_PCScaled_E1 = ...
%     dFoF_screen(cellsToUseInE1,:)' .* score(cellsToUseInE1_collapsed,2)';
% dFoF_roi_screen_collapsed_PCScaled_E1_scaled = dFoF_roi_screen_collapsed_PCScaled_E1 ./ std(dFoF_roi_screen_collapsed_PCScaled_E1);
% 
% E1_tmp = mean(dFoF_roi_screen_collapsed_PCScaled_E1_scaled,2);
% figure; plot(smooth(E1_tmp,5))
% 
% figure; plot(explained)
% figure; 
% 
% PCsToPlot = 1:5;
% for ii = 1:5
%     subplot(numel(PCsToPlot),1,ii)
%  plot(coeff(:,ii))
%  linkaxes
% end
% 
% figure; plot(dFoF_roi_scaled_screen(cellsToUseInE1,:)')
% figure; plot(smooth(mean(dFoF_roi_scaled_screen(cellNum_collapsed(find(score(:,2) > 20)),:)',2),10))

%% Transofrm coordinate indices

for ii = 1:numel(cellNumsToUse)
    idxBounds_ROI{ii}(1,1) = min(stat{cellNumsToUse(ii)}.xpix)+1; % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
    idxBounds_ROI{ii}(2,1) = max(stat{cellNumsToUse(ii)}.xpix)+1;
    idxBounds_ROI{ii}(1,2) = min(stat{cellNumsToUse(ii)}.ypix)+1;
    idxBounds_ROI{ii}(2,2) = max(stat{cellNumsToUse(ii)}.ypix)+1;

    mask_center{ii} = [ mean([idxBounds_ROI{ii}(1,1) , idxBounds_ROI{ii}(2,1)])  ,  mean([idxBounds_ROI{ii}(1,2) , idxBounds_ROI{ii}(2,2)]) ];
    
    mask_ROI_directCells{ii} = mask_ROI_cropped_int16{cellNumsToUse(ii)};
end
%% Check that PCs have a cell in the middle
figure;
% axPCA = subplot(1,4,1);
clear temp temp2 temp3 coeff scores
numPCsToUse = 3;
for jj = 1:numPCsToUse % PCs to use
    for ii = 1:4
        temp{ii} = movie_all(idxBounds_ROI{ii}(1,2):idxBounds_ROI{ii}(2,2)  ,  idxBounds_ROI{ii}(1,1):idxBounds_ROI{ii}(2,1),:);
        temp2{ii} = permute(temp{ii},[3 1 2]);
        temp3{ii} = reshape(temp2{ii}, size(temp2{ii},1) , size(temp2{ii},2) * size(temp2{ii},3));
        
        [coeff{ii} scores{ii}] = pca(double(temp3{ii}'));
        justPC1Scores{ii} = scores{ii}(:,jj);
        scores2{ii} = reshape(justPC1Scores{ii},size(temp{ii},1) , size(temp{ii},2));
        
        subplot(numPCsToUse,4,ii + 4*(jj-1));
        imagesc(scores2{ii})
    end
end
%% Simulation
% figure; plot(logger_output(:,28), 'LineWidth', 1.5)

% threshVals = [2.0:.3:3.5];
threshVals = [2.2];
% scale_factors = 1./std(dFoF_roi(:,cellNumsToUse));

duration_trial = 240; % I'm making the trials long to avoid overfitting to the ITIs.
duration_total = size(F_roi,1);
clear total_rewards_per30s currentCursor currentCSThresh currentCSQuiescence currentCETrial currentCEReward currentCETimeout
cc = 1;
for ii = threshVals
    tic
    disp(['testing threshold value:  ', num2str(ii)])
    
    for jj = 1:size(F_roi,1)
        if jj == 1
            startSession_pref = 1;
        else
            startSession_pref = 0;
        end
        
        %         [logger_output(jj,:)] = BMI_trial_simulation(F_roi(jj,cellNumsToUse),startSession_pref, scale_factors, ii, duration_total, duration_trial);
        [ currentCursor(jj) , currentCSThresh(jj), currentCSQuiescence(jj), currentCETrial(jj),  currentCEReward(jj), currentCETimeout(jj) ]...
            = BMI_trial_simulation2(F_roi(jj,:),startSession_pref, scale_factors, ii, duration_total, duration_trial);
    end
%         total_rewards_perMin(cc) = sum(all_rewards)/(duration_total/30);
    %     disp(['total rewards per 30s at threshold = ' , num2str(ii) , ' is:   ', num2str(total_rewards_per30s(cc))]);
    %     figure; plot([all_rewards.*1.5, all_cursor, baselineState.*1.1, baselineHoldState*0.8, thresholdState.*0.6])
    toc
    cc = cc+1;

% figure;
% plot(threshVals, total_rewards_per30s)
% figure; plot(logger_output(:,33:36) .* repmat(scale_factors,size(logger_output,1),1))
figure; plot(currentCursor) %cursor
hold on; plot(currentCSThresh) %thresh
% hold on; plot(currentCSQuiescence) %wait for quiescence
hold on; plot(currentCETrial) %trial
hold on; plot(currentCEReward, 'LineWidth',2) % rewards
hold on; plot([0 numel(currentCEReward)] , [threshVals(end) threshVals(end)])
rewardToneHold_diff = diff(currentCEReward);
% timeout_diff = diff(logger_output(:,23));
numRewards = sum(rewardToneHold_diff(rewardToneHold_diff > 0.5));
numRewardsPerMin = numRewards / (size(F_roi,1) / (Fs_frameRate * 60));
disp(['For Threshold  =  ', num2str(ii) , '  total rewards: ', num2str(numRewards) , ' , rewards per min: ' , num2str(numRewardsPerMin) ])
% numTimeouts = sum(timeout_diff(timeout_diff > 0.5))
% numRewards/(numRewards+numTimeouts)
end
%%
baselineStuff.threshVals = threshVals;
baselineStuff.numRewards = numRewards;
baselineStuff.numRewardsPerMin = numRewardsPerMin;

baselineStuff.traces_to_use_E1 = traces_to_use_E1;
baselineStuff.traces_to_use_E2 = traces_to_use_E2;
baselineStuff.F_roi = F_roi;
baselineStuff.dFoF_roi = dFoF_roi;
baselineStuff.cellNumsToUse = cellNumsToUse;
baselineStuff.mask_center = mask_center;
baselineStuff.directory = directory;
baselineStuff.file_baseName = file_baseName;
baselineStuff.frames_totalExpected = frames_totalExpected;
baselineStuff.frames_perFile = frames_perFile;
baselineStuff.Fs_frameRate = Fs_frameRate;
baselineStuff.duration_trace = duration_trace;
baselineStuff.duration_trial = duration_trial;
baselineStuff.baseline_pctile = baseline_pctile;
baselineStuff.scale_factors = scale_factors;

baselineStuff.idxBounds_ROI = idxBounds_ROI;

baselineStuff.framesForMeanImForMC = framesForMeanImForMC;
baselineStuff.meanImForMC = meanImForMC;

baselineStuff.paddingForMCRef = paddingForMCRef;
baselineStuff.mask_ROI = mask_ROI;
baselineStuff.mask_ROI_cropped = mask_ROI_cropped;
baselineStuff.mask_ROI_directCells = mask_ROI_directCells;


%%
[baselineStuff , motionCorrectionRefImages] = make_motionCorrectionRefImages2(baselineStuff, paddingForMCRef); % second arg is num of pixels to pad sides of ROI with
baselineStuff.motionCorrectionRefImages = motionCorrectionRefImages;

%%
baselineStuff.framesForMeanImForMC = [];
save([directory, '\baselineStuff'], 'baselineStuff')
% save([directory, '\motionCorrectionRefImages'], 'motionCorrectionRefImages')
