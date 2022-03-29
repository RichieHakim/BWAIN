%% Online Automatic Z-Correction
% Akshay Jaggi and Richard Hakim, 2022
%
% Code to automatically and actively align a stream of 2p microscopy images
% by comparing them to a reference z-stack. 
% This code will interface with scanimage to intake recorded images, 
% determine the z-correction necessary, and then send that command
% to the fast-z controller in scanimage. 
% Since this code will run online, it will be called on a certain interval.
% To keep the computer from slowing down, we'll run this only during
% ITI periods or during rewards. 

% Input: reference z-stack and recorded images 
% Does: 
% 1. Records the rolling median over the last n images
% 2. Subtracts the rolling median from the image
% 3. Computes the fft2 of the de-medianed image and takes the phase image
% 4. Find the phase correlation of the image with each of the reference
%    images in the z-stack
% 5. Given the z-stack height between the reference images, return the
%    distance between the central reference image and the reference image 
%    that the current image is most similar to. 
% 6. Send distance in um to the z-stack controller
% Output: distance from most similar reference image to the central
%         reference image