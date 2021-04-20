%% BMI Ensemble selection script
% function best_decoder_value = BMI_ensembleSelection(directory)
%% import baseline movie
directory = 'F:\RH_Local\Rich data\scanimage data\mouse 1.31\test 20200215';
file_baseName = 'test_00003_';

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
                    movie_chunk_ds = imresize3(movie_chunk, [size(movie_chunk,1), size(movie_chunk,2), round(size(movie_chunk,3)/ds_factor)]);
                    saveastiff(movie_chunk_ds, [directory, '\downsampled\ds_', fileNames(lastImportedFileIdx+1,:)]);
                    clear movie_chunk_ds
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
baseline_pctile = 20;
% %% Make Standard Deviation Image
% tic
% chunk_size = 25; % number of columns to process at once. vary this value to maximize speed. There is a sweet spot for memory usage around 25 columns of size 512 each.
% movie_std = nan(size(movie_all,1), size(movie_all,2));
% for ii = 1:chunk_size:size(movie_all,2)
%     if ii + chunk_size > size(movie_all,2)
%         movie_std(:,ii:size(movie_all,2)) = std(single(movie_all(:,ii:size(movie_all,2),:)),[],3);
%     else
%         movie_std(:,ii:ii+chunk_size) = std(single(movie_all(:,ii:ii+chunk_size,:)),[],3);
%     end
% end
% toc
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
%% Choose cells
% Import the Fall.mat file from suite2p's output
clear dFoF
clear F_new
% for ii = 1:size(F,1)
% %     dFoF(ii,:) = F(ii,:) ./ prctile(F(ii,:),20);
%     
%     F_new(ii,:) = squeeze(mean(mean(movie_all(stat{ii}.ypix+1 , stat{ii}.xpix+1,1:5:end),1),2));
%     dFoF(ii,:) = F_new(ii,:) ./ prctile(F_new(ii,:),20);
% 
% end

for ii = fliplr(1:size(F,1))
% for ii = 3
% for ii = (1:size(F,1))
    if iscell(ii,1) == 1
%         if  sum(spks(ii,:) > 10) > 50
%             if  sum(spks(ii,:) > 50) > 6
        if  sum(spks(ii,1:1000) > 5) > 5 && sum(spks(ii,1001:2000) > 5) > 5 && sum(spks(ii,2001:3000) > 5) > 5 && sum(spks(ii,3001:end) > 5) > 5
            if  sum(spks(ii,1:1000) > 30) > 1 && sum(spks(ii,1001:2000) > 30) > 1 && sum(spks(ii,2001:3000) > 30) > 1 && sum(spks(ii,3001:end) > 30) > 1
%                 if stat{ii}.std < 40
%                 ii
                
                tmpMinY = min(stat{ii}.ypix)+1;
                tmpMaxY = max(stat{ii}.ypix)+1;
                tmpMinX = min(stat{ii}.xpix)+1;
                tmpMaxX = max(stat{ii}.xpix)+1;
                
                Im_mask{ii} = zeros(size(movie_all,1) , size(movie_all,2));
                for jj = 1:stat{ii}.npix
                    Im_mask{ii}(stat{ii}.ypix(jj)+1 , stat{ii}.xpix(jj)+1) = stat{ii}.lam(jj)*100;
                end

                F_new(ii,:) = squeeze(sum(sum( double(movie_all(tmpMinY: tmpMaxY , tmpMinX: tmpMaxX, :)) .* repmat(Im_mask{ii}(tmpMinY: tmpMaxY , tmpMinX: tmpMaxX),1,1,size(movie_all,3)) ,1),2));
                dFoF(ii,:) = F_new(ii,:) ./ prctile(F_new(ii,:),30);
        
%                 dFoF(ii,:) = F(ii,:) ./ prctile(F(ii,:),30);


                tmpIm = double(ops.meanImg);
                tmpIm2 = double(ops.sdmov);
                tmpIm3 = double(ops.meanImg);
                tmpIm4 = double(ops.meanImgE);
                tmpIm5 = double(ops.max_proj);
                tmpIm6 = double(ops.Vcorr);
                               

                fig = figure;
%                 fig.Position = [100 50 150 900];
%                 fig.Position = [10 50 1200 800];
                fig.Position = [10 400 1200 800];
                fig.Name = ['cell #: ' , num2str(ii-1)];
                
                subplot(3,6,1)
                imagesc(Im_mask{ii}(tmpMinY:tmpMaxY , tmpMinX:tmpMaxX))
                title(['mask cell# ' , num2str(ii-1)])
                
                subplot(3,6,2)
                imagesc(tmpIm2(tmpMinY:tmpMaxY , tmpMinX:tmpMaxX))
                title(num2str(stat{ii}.aspect_ratio))
                title('sdmov')
                
                subplot(3,6,3)
                imagesc(tmpIm3(tmpMinY+1:tmpMaxY , tmpMinX+1:tmpMaxX))
                title('meanImg')
                
                subplot(3,6,4)
                imagesc(tmpIm4(tmpMinY+1:tmpMaxY , tmpMinX+1:tmpMaxX))
                title('meanImgE')
                
                subplot(3,6,5)
                imagesc(tmpIm5(tmpMinY+1:tmpMaxY , tmpMinX+1:tmpMaxX))
                title('max_proj')
                
                subplot(3,6,6)
                imagesc(tmpIm6(tmpMinY+1:tmpMaxY , tmpMinX+1:tmpMaxX))
                title('Vcorr')
                
%                 fig2 = figure;
                subplot(3,6,7:12)
                plot(dFoF(ii,:))
%                 plot(smooth(dFoF(ii,:),3))
                ylim([0.8 5])
%                 fig2.Name = num2str(ii);
                
                subplot(3,6,13:15)
%                 plot(smooth(dFoF(ii,:),5))
%                 plot(dFoF(ii,:))
                histogram(dFoF(ii,:),1000)
                xlim([0.8 2])
                title(num2str(std(dFoF(ii,:))))
%                 ylim([0.9 1.2])

                subplot(3,6,16:18)
                plot(dFoF(ii,:))
                xlim([1 1000])
                ylim([0.8 1.5])
                
                %                 figure; imagesc(ops.meanImg(stat{ii}.ypix , stat{ii}.xpix))
%                 end
            end
        end
    end
end

fig2=figure;
fig2.Name = ['mean F of all cells'];
plot(mean(F,1))

%% User enters cells to look at
% cellNums_s2p = [25 27 12 38  28 ]+1;
% cellNums_s2p = [25 27 12 38]+1;
cellNums_s2p = [11 23 13 568]+1;

%%
% movie_mean = mean(movie_all,3);
% movie_mean = mean(movie_all(:,:,2.3e4:2.33e4),3);
% movie_mean = mean(movie_all,3);
movie_mean = meanImForMC;
% movie_mean = mean(movie_all(:,:,end-5000:end-300),3);
% movie_fano = movie_std ./ movie_mean;
%
% h1 = figure;
% imagesc(movie_mean)
% set(gca,'CLim',[0 1e4])

% h1 = figure;
% % imagesc(movie_mean)
% imagesc(movie_fano)
% ax1 = subplot(1,1,1);
% set(ax1, 'CLim',[0.1 1.3]);

% h4 = figure;
% imagesc(max(movie_all,[],3))

movie_clahe = adapthisteq(movie_mean./max(movie_mean(:)),'NumTiles',[20 20],'clipLimit',0.2,'Distribution','rayleigh');

h1 = figure;
ax1 = subplot(1,1,1);
imagesc(movie_clahe);
set(gca,'CLim', [0.26 1])

h2 = figure;
ax2 = subplot(1,1,1);
% imagesc(movie_fano)
im_1 = imagesc(movie_mean);

% set(ax2, 'CLim',[0.1 1.3]);
set(ax2, 'CLim',[100 900]);
%%
load([directory, '\downsampled\suite2p\plane0\Fall.mat']);
clear masks_s2p
for ii = 1:numel(cellNums_s2p)
    masks_s2p( stat{cellNums_s2p(ii)}.ypix +1, stat{cellNums_s2p(ii)}.xpix +1) = 1;
end
alphamask(masks_s2p, [0.9, 0.3, 0.3] , 0.4, ax2);
% alphamask(masks_s2p, [0.9, 0.3, 0.3] , 0.4, ax1);
% tightfig;
%%

h3 = figure;
ax3 = subplot(2,1,1);
ax4 = subplot(2,1,2);
% set(h3,'Position',[100 20 900 1800]);

% ROI selection
cmap = distinguishable_colors(50,['b']);
pref_addROI = 1;
makeNewROI = 1;

cc = 1;

clear mask userShape mask_position ROI_coords F_roi dFoF_roi
while pref_addROI == 1
    %     set(h2,'CurrentAxes',ax2);
    axes(ax2)
    if makeNewROI == 1
        userShape{cc} = imrect;
        
        %         mask{cc} = userShape{cc}.createMask;
        mask{cc} = createMask(userShape{cc}, im_1);
        %         hey = images.roi.Rectangle(gca,'StripeColor','r');
        
        mask_position{cc} = userShape{cc}.getPosition; % first two values are x and y of top left corner (assuming ROI made by dragging top left corner first), second two values are dX and dY
        ROI_coords{cc} = round([mask_position{cc}(1) , mask_position{cc}(1) + mask_position{cc}(3) , mask_position{cc}(2) , mask_position{cc}(2) + mask_position{cc}(4)]); % x1, x2 , y1, y2
        
        %% ROI extraction and dFoF calculation
        baselineWin = 90 * Fs_frameRate;
        F_roi(:,cc) = squeeze( mean( mean( movie_all([ROI_coords{cc}(3):ROI_coords{cc}(4)], [ROI_coords{cc}(1):ROI_coords{cc}(2)], : ) ,1) ,2) );
        dFoF_roi(:,cc) = (F_roi(:,cc) - medfilt1(F_roi(:,cc), baselineWin)) ./  medfilt1(F_roi(:,cc), baselineWin);
        numTraces = size(F_roi,2);
        
        %% plotting
        ROI_patchX = [ROI_coords{cc}(1) , ROI_coords{cc}(2) , ROI_coords{cc}(2) , ROI_coords{cc}(1)];
        ROI_patchY = [ROI_coords{cc}(3) , ROI_coords{cc}(3) , ROI_coords{cc}(4) , ROI_coords{cc}(4)];
        patch(ROI_patchX, ROI_patchY,cmap(cc,:),'EdgeColor',cmap(cc,:),'FaceColor','none','LineWidth',2);
        
        axes(ax3); hold on
        plot((1:size(movie_all,3)) / Fs_frameRate, F_roi(:,cc),'Color',cmap(cc,:))
        
        axes(ax4); hold on
        plot((1:size(movie_all,3)) / Fs_frameRate,dFoF_roi(:,cc),'Color',cmap(cc,:))
        %             plot((1:size(movie_all,3)) / Fs_frameRate, smooth(dFoF_roi(:,cc),15,'sgolay'),'Color',cmap(cc,:))
        
        clear figLegend
        for ii = 1:numTraces
            figLegend{ii} = num2str(ii);
        end
        legend(figLegend);
        
        cc = cc+1;
        makeNewROI = 0;
    end
    button = waitforbuttonpress;
    if button == 1
        keyPressOutput = get(gcf,'CurrentKey');
        if strcmp(keyPressOutput, 'return') == 1
            pref_addROI = 0;
        end
        
        if strcmp(keyPressOutput, 'space') ~= 0
            makeNewROI = 1;
        else makeNewROI = 0;
        end
    end
end



%% Choose cells
traces_to_use_E1 = input('Input numbers for E1 trace to use:  ');
traces_to_use_E2 = input('Input numbers for E2 trace to use:  ');
cellNumsToUse = [traces_to_use_E1 , traces_to_use_E2];

%% Update ROIs
figure; hold on;
clear F_roi_new dFoF_roi_new
for cc = 1:size(F_roi,2)
    F_roi_new(:,cc) = squeeze( mean( mean( movie_all([ROI_coords{cc}(3):ROI_coords{cc}(4)], [ROI_coords{cc}(1):ROI_coords{cc}(2)], : ) ,1) ,2) );
    dFoF_roi_new(:,cc) = (F_roi_new(:,cc) - medfilt1(F_roi_new(:,cc), baselineWin)) ./  medfilt1(F_roi_new(:,cc), baselineWin);
    plot((1:size(dFoF_roi_new,1)) / Fs_frameRate,dFoF_roi_new(:,cc),'Color',cmap(cc,:))
end
F_roi = F_roi_new; dFoF_roi = dFoF_roi_new;
scale_factors = 1./std(dFoF_roi(:,cellNumsToUse));

%% Plot direct cell ROIs
figure; plot(F_roi(:,cellNumsToUse))
figure; plot(dFoF_roi(:,cellNumsToUse) .* repmat(scale_factors,size(dFoF_roi,1),1))
E1 = mean(dFoF_roi(:,cellNumsToUse(1:2)) .* repmat(scale_factors(1:2),size(dFoF_roi,1),1),2);
E2 = mean(dFoF_roi(:,cellNumsToUse(3:4)) .* repmat(scale_factors(3:4),size(dFoF_roi,1),1),2);
cursor = E1-E2;
figure; plot(cursor)

%% Transofrm coordinate indices
% mask_position{cc} = userShape{cc}.getPosition; % first two values are x and y of top left corner (assuming ROI made by dragging top left corner first), second two values are dX and dY

clear mask_center mask_width
for ii = 1:numel(mask_position)
    mask_centerX_temp = mask_position{ii}(1) + mask_position{ii}(3)/2;
    mask_centerY_temp = mask_position{ii}(2) + mask_position{ii}(4)/2;
    
    mask_center(ii,:) = [mask_centerX_temp , mask_centerY_temp];
    mask_width(ii,:) = [mask_position{ii}(3) , mask_position{ii}(4)];
end
mask_center_cellsToUse = mask_center(cellNumsToUse,:);
mask_width_cellsToUse = mask_width(cellNumsToUse,:);

image_center = [size(movie_all,2) , size(movie_all,1)]/2;
mask_center_SI_angle = (mask_center - repmat(image_center, size(mask_center,1), 1)) ./ repmat(pixelsPerDegree, size(mask_center,1), 1);
mask_center_SI_angle_cellsToUse = mask_center_SI_angle(cellNumsToUse,:);

mask_width_SI_angle = mask_width ./ repmat(pixelsPerDegree, size(mask_width,1), 1);
mask_width_SI_angle_cellsToUse = mask_width_SI_angle(cellNumsToUse,:);

for ii = 1:size(mask_center_cellsToUse,1)
    idxBounds_ROI{ii}(1,1) = round( mask_center_cellsToUse(ii,1) - (mask_width_cellsToUse(ii,1)/2) ); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
    idxBounds_ROI{ii}(2,1) = round( mask_center_cellsToUse(ii,1) + (mask_width_cellsToUse(ii,1)/2) );
    idxBounds_ROI{ii}(1,2) = round( mask_center_cellsToUse(ii,2) - (mask_width_cellsToUse(ii,2)/2) );
    idxBounds_ROI{ii}(2,2) = round( mask_center_cellsToUse(ii,2) + (mask_width_cellsToUse(ii,2)/2) );
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

% threshVals = [0 2.4:.1:2.5];
threshVals = [2.4];
scale_factors = 1./std(dFoF_roi(:,cellNumsToUse));

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
            = BMI_trial_simulation2(F_roi(jj,cellNumsToUse),startSession_pref, scale_factors, ii, duration_total, duration_trial);
    end
    %     total_rewards_per30s(cc) = sum(all_rewards)/(duration_total/30);
    %     disp(['total rewards per 30s at threshold = ' , num2str(ii) , ' is:   ', num2str(total_rewards_per30s(cc))]);
    %     figure; plot([all_rewards.*1.5, all_cursor, baselineState.*1.1, baselineHoldState*0.8, thresholdState.*0.6])
    toc
    cc = cc+1;
end
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
numRewards = sum(rewardToneHold_diff(rewardToneHold_diff > 0.5))
% numTimeouts = sum(timeout_diff(timeout_diff > 0.5))
% numRewards/(numRewards+numTimeouts)

%%
baselineStuff.threshVals = threshVals;
% baselineStuff.total_rewards_per30s = total_rewards_per30s;
% baselineStuff.movie_std = movie_std;
% baselineStuff.movie_fano = movie_fano;

baselineStuff.movie_mean = movie_mean;
baselineStuff.traces_to_use_E1 = traces_to_use_E1;
baselineStuff.traces_to_use_E2 = traces_to_use_E2;
baselineStuff.F_roi = F_roi;
baselineStuff.dFoF_roi = dFoF_roi;
baselineStuff.cellNumsToUse = cellNumsToUse;
baselineStuff.mask_center = mask_center;
baselineStuff.mask_width = mask_width;
baselineStuff.mask_center_cellsToUse = mask_center_cellsToUse;
baselineStuff.mask_width_cellsToUse = mask_width_cellsToUse;
baselineStuff.image_center = image_center;
baselineStuff.mask_center_SI_angle = mask_center_SI_angle;
baselineStuff.mask_center_SI_angle_cellsToUse = mask_center_SI_angle_cellsToUse;
baselineStuff.directory = directory;
baselineStuff.file_baseName = file_baseName;
baselineStuff.frames_totalExpected = frames_totalExpected;
baselineStuff.frames_perFile = frames_perFile;
baselineStuff.scanAngleMultiplier = scanAngleMultiplier;
baselineStuff.pixelsPerDegree = pixelsPerDegree;
baselineStuff.Fs_frameRate = Fs_frameRate;
baselineStuff.duration_trace = duration_trace;
baselineStuff.duration_trial = duration_trial;
baselineStuff.baseline_pctile = baseline_pctile;
baselineStuff.scale_factors = scale_factors;

baselineStuff.ROI_patchX = ROI_patchX;
baselineStuff.ROI_patchY = ROI_patchY;
baselineStuff.mask_width_SI_angle = mask_width_SI_angle;
baselineStuff.mask_width_SI_angle_cellsToUse = mask_width_SI_angle_cellsToUse;

baselineStuff.idxBounds_ROI = idxBounds_ROI;

baselineStuff.framesForMeanImForMC = framesForMeanImForMC;
baselineStuff.meanImForMC = meanImForMC;


%%
[baselineStuff , motionCorrectionRefImages] = make_motionCorrectionRefImages(baselineStuff);
baselineStuff.motionCorrectionRefImages = motionCorrectionRefImages;

%%
baselineStuff.framesForMeanImForMC = [];
save([directory, '\baselineStuff'], 'baselineStuff')
% save([directory, '\motionCorrectionRefImages'], 'motionCorrectionRefImages')
