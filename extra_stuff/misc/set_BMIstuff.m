% Upload/update ROI images for motion correction
numCells = numel(baselineStuff.cellNumsToUse);
clear BMIstuff.img_ROI_reference
global BMIstuff
for ii = 1:numCells
    BMIstuff.img_ROI_reference{ii} = ImageData(baselineStuff.idxBounds_ROI{ii}(1,2):baselineStuff.idxBounds_ROI{ii}(2,2)  ,  baselineStuff.idxBounds_ROI{ii}(1,1):baselineStuff.idxBounds_ROI{ii}(2,1)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
    BMIstuff.img_MC_reference{ii} = ImageData(baselineStuff.idxBounds_imgMC{ii}(1,2):baselineStuff.idxBounds_imgMC{ii}(2,2)  ,  baselineStuff.idxBounds_imgMC{ii}(1,1):baselineStuff.idxBounds_imgMC{ii}(2,1)); % note that idxBounds_ROI will be [[x1;x2] , [y1;y2]]
end
