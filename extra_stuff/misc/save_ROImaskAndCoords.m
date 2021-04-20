scaleFactor_forMotionCorrectionImage = 4;
sFMCI = scaleFactor_forMotionCorrectionImage/2;

clear idxBounds_ROI idxBounds_imgMC
for ii = 1:size(baselineStuff.mask_center_cellsToUse,1)
    idxBounds_ROI{ii}(1,1) = round( baselineStuff.mask_center_cellsToUse(ii,1) - (baselineStuff.mask_width_cellsToUse(ii,1)/2) ); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
    idxBounds_ROI{ii}(2,1) = round( baselineStuff.mask_center_cellsToUse(ii,1) + (baselineStuff.mask_width_cellsToUse(ii,1)/2) );
    idxBounds_ROI{ii}(1,2) = round( baselineStuff.mask_center_cellsToUse(ii,2) - (baselineStuff.mask_width_cellsToUse(ii,2)/2) );
    idxBounds_ROI{ii}(2,2) = round( baselineStuff.mask_center_cellsToUse(ii,2) + (baselineStuff.mask_width_cellsToUse(ii,2)/2) );
    
    idxBounds_imgMC{ii}(1,1) = round( baselineStuff.mask_center_cellsToUse(ii,1) - (baselineStuff.mask_width_cellsToUse(ii,1)*sFMCI) ); % note that idx will be [[x1;x2] , [y1;y2]]
    idxBounds_imgMC{ii}(2,1) = round( baselineStuff.mask_center_cellsToUse(ii,1) + (baselineStuff.mask_width_cellsToUse(ii,1)*sFMCI) ); % for now, the motion correction (MC) image is just doubling the width of the ROI image
    idxBounds_imgMC{ii}(1,2) = round( baselineStuff.mask_center_cellsToUse(ii,2) - (baselineStuff.mask_width_cellsToUse(ii,2)*sFMCI) );
    idxBounds_imgMC{ii}(2,2) = round( baselineStuff.mask_center_cellsToUse(ii,2) + (baselineStuff.mask_width_cellsToUse(ii,2)*sFMCI) );
end
baselineStuff.idxBounds_ROI = idxBounds_ROI;
baselineStuff.idxBounds_imgMC = idxBounds_imgMC;

clear directory file_baseName ii idxBounds_ROI idxBounds_imgMC scaleFactor_forMotionCorrectionImage sFMCI

% end

%%
% Upload/update ROI images for motion correction
% numCells = numel(baselineStuff.cellNumsToUse);
numCells = 4;
% clear BMIstuff
% global motionCorrectionRefImages
for ii = 1:numCells
    baselineStuff.motionCorrectionRefImages.img_ROI_reference{ii} = ImageData(baselineStuff.idxBounds_ROI{ii}(1,2):baselineStuff.idxBounds_ROI{ii}(2,2)  ,  baselineStuff.idxBounds_ROI{ii}(1,1):baselineStuff.idxBounds_ROI{ii}(2,1)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
    baselineStuff.motionCorrectionRefImages.img_MC_reference{ii} = ImageData(baselineStuff.idxBounds_imgMC{ii}(1,2):baselineStuff.idxBounds_imgMC{ii}(2,2)  ,  baselineStuff.idxBounds_imgMC{ii}(1,1):baselineStuff.idxBounds_imgMC{ii}(2,1)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
end
% save('motionCorrectionRefImages','motionCorrectionRefImages')
save('baselineStuff','baselineStuff')
%%
figure;
subplot(2, numCells, 1)
for ii = 1:numCells
    subplot(2,numCells, ii);
    imagesc(baselineStuff.motionCorrectionRefImages.img_MC_reference{ii})
    subplot(2,numCells, ii+numCells)
    imagesc(baselineStuff.motionCorrectionRefImages.img_ROI_reference{ii})
end

clear numCells ii