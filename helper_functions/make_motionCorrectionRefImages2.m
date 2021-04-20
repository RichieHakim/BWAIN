function [baselineStuff , motionCorrectionRefImages] = make_motionCorrectionRefImages2(baselineStuff, padding)
clear idxBounds_ROI idxBounds_imgMC
for ii = 1:numel(baselineStuff.cellNumsToUse)
    idxBounds_imgMC{ii}(1,1) = baselineStuff.idxBounds_ROI{ii}(1,1) - padding;
    idxBounds_imgMC{ii}(2,1) = baselineStuff.idxBounds_ROI{ii}(2,1) + padding;
    idxBounds_imgMC{ii}(1,2) = baselineStuff.idxBounds_ROI{ii}(1,2) - padding;
    idxBounds_imgMC{ii}(2,2) = baselineStuff.idxBounds_ROI{ii}(2,2) + padding;
end
baselineStuff.idxBounds_imgMC = idxBounds_imgMC;

clear directory file_baseName ii idxBounds_ROI idxBounds_imgMC scaleFactor_forMotionCorrectionImage sFMCI
%%
% Upload/update ROI images for motion correction
% numCells = numel(baselineStuff.cellNumsToUse);
ImageData = baselineStuff.meanImForMC;
numCells = numel(baselineStuff.cellNumsToUse);
clear BMIstuff
% global motionCorrectionRefImages
for ii = 1:numCells
    motionCorrectionRefImages.img_ROI_reference{ii} = ImageData(baselineStuff.idxBounds_ROI{ii}(1,2):baselineStuff.idxBounds_ROI{ii}(2,2)  ,  baselineStuff.idxBounds_ROI{ii}(1,1):baselineStuff.idxBounds_ROI{ii}(2,1)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
    motionCorrectionRefImages.img_MC_reference{ii} = ImageData(baselineStuff.idxBounds_imgMC{ii}(1,2):baselineStuff.idxBounds_imgMC{ii}(2,2)  ,  baselineStuff.idxBounds_imgMC{ii}(1,1):baselineStuff.idxBounds_imgMC{ii}(2,1)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
end
% save('motionCorrectionRefImages','motionCorrectionRefImages')
% save('baselineStuff','baselineStuff')
%%
figure;
subplot(2, numCells, 1)
for ii = 1:numCells
    subplot(2,numCells, ii);
    imagesc(motionCorrectionRefImages.img_MC_reference{ii})
    subplot(2,numCells, ii+numCells)
    imagesc(motionCorrectionRefImages.img_ROI_reference{ii})
end

clear numCells ii
end