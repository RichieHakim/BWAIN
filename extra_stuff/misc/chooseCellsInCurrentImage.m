figure; 
imagesc(ImageData);

cellNumsToChoose = 1:4;
clear mask_position
for ii = cellNumsToChoose
userShape = imrect;
mask = createMask(userShape);
mask_position{ii} = userShape.getPosition;

baselineStuff.mask_center_cellsToUse(ii,1:2) = mask_position{ii}(1:2) + mask_position{ii}(3:4)./2;
baselineStuff.mask_width_cellsToUse(ii,1:2) = mask_position{ii}(3:4);
end
clear ii userShape mask mask_position cellNumsToChoose

%%
save_ROImaskAndCoords