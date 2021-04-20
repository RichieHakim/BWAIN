directory = 'F:\RH_Local\Rich data\scanimage data\mouse 10.30\20191203';
file_baseName = '\EnsembleSelectionWorkspace.mat';
load([directory, file_baseName])

%%
for ii = 1:numel(baselineStuff.cellNumsToUse)
    
hSf = scanimage.mroi.scanfield.fields.IntegrationField();  % create an Integration Scanfield
hSf.channel = 1;                                    % assign IntegrationField to channel 1
hSf.centerXY = baselineStuff.mask_center_SI_angle_cellsToUse(ii,:);                           % set center [x,y] of IntegrationField to center of Reference space
hSf.sizeXY = baselineStuff.mask_width_SI_angle_cellsToUse(ii,:);                           % set size [x,y] of IntegrationField in Reference space
% hSf.centerXY = baselineStuff.mask_center_cellsToUse;                           % set center [x,y] of IntegrationField to center of Reference space
% hSf.sizeXY = baselineStuff.mask_width_cellsToUse;                           % set size [x,y] of IntegrationField in Reference space
% hSf.rotationDegrees = 0;                            % set rotation of IntegrationField in Reference space
% hSf.mask = rand(10);                                % set a mask with weights for underlying pixels
% hSf.mask = [];                                % set a mask with weights for underlying pixels
 
hRoi = scanimage.mroi.Roi();   % create empty Roi
z = 0;
hRoi.add(z,hSf);               % add IntegrationField at z = 0
 
hSI.hIntegrationRoiManager.roiGroup.add(hRoi);  % add IntegrationRoi to IntegrationRoiManager
end