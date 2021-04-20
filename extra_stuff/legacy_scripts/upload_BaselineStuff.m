% function upload_BaselineStuff
global baselineStuff

directory = 'F:\RH_Local\Rich data\scanimage data\mouse 10.30\20191203';
file_baseName = '\EnsembleSelectionWorkspace.mat';
load([directory, file_baseName])

%%
save_ROImaskAndCoords