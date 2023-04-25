function PMTtest(path_imLight, path_imDark, path_save)
%%
% Function for finding the gain factor and converting to absolute photons
% Inputs should be images of a fluorescent slide or something homogeneous
% Gyu Heo 2023
% Args:
%     path_imLight (str):
%         path to scanimage .tif file containing multiple images
%          obtained with a constant about of light on
%         shape: (height, width, n_images)
%     path_imLight (str):
%         path to scanimage .tif file containing multiple images
%          obtained with zero light
%         shape: (height, width, n_images)
%     path_save (str):
%         path, with a '.mat' suffix for where to save the file
    

%% Load Light- / Dark-current Sample Image
import ScanImageTiffReader.ScanImageTiffReader
reader_Light = ScanImageTiffReader(path_imLight);
slices_Light = double(permute(reader_Light.data(),[3,2,1])); % Frames * Vert * Horiz

reader_Dark = ScanImageTiffReader(path_imDark);
slices_Dark = double(permute(reader_Dark.data(),[3,2,1])); % Frames * Vert * Horiz

%% Assume Poisson Dist., get mean / var values
mean_Light_current = squeeze(mean(slices_Light, 1));
var_Light_current = squeeze(var(slices_Light, 0, 1));

mean_Dark_current = squeeze(mean(slices_Dark, 1));
var_Dark_current = squeeze(var(slices_Dark, 0, 1));

% (H, W)
mean_current = mean_Light_current - mean_Dark_current;
var_current =  var_Light_current - var_Dark_current;

% Assumes I = g*N, where N is the photon count governed by Poisson distribution.
gain = var_current./mean_current;

% (H,W)
N_photon = mean_Light_current./gain;

%% Plotting
figure();
imagesc(mean_Light_current);
title("Light Current meanImage");

figure();
imagesc(mean_Dark_current);
title("Dark Current meanImage");

figure();
imagesc(mean_current);
title("Light - Dark Current meanImage");

figure();
plot(mean(gain,1));
title("average Gain");

figure();
plot(mean(N_photon,1));
title("average number of photon");

%% Save?
outputs = struct();
outputs.mean_Light_current = mean_Light_current;
outputs.var_Light_current = var_Light_current;
outputs.mean_Dark_current = mean_Dark_current;
outputs.var_Dark_current = var_Dark_current;

outputs.mean_current = mean_current;
outputs.var_current = var_current;

outputs.gain = gain;
outputs.N_photon = N_photon;

outputs.path_imLight = path_imLight;
outputs.path_imDark = path_imDark;

if nargin >= 3
    save(path_save, 'outputs');
end

