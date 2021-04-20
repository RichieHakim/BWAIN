%%
% This script aligns the logger to the tifs that come out of scanimage. It
% assumes that the logger used the current simple_image_hash.m to compute the hash
% 
% Dependencies:
% -  simple_image_hash.m
% -  compute_hashSequence.m
% -  compare_hashSequence.m
% -  indexWithNaNs.m
%% Import movie
% Should be in day N-1 or day 0 folder
directory_movie = 'D:\RH_local\data\scanimage data\round 5 experiments\mouse 2_6\20210417\exp';
fileName_movie = 'file_';

frames_totalExpected = 108000;
frames_perFile = 1000;

% lastImportedFileIdx = 0;
% clear movie_all

if exist('lastImportedFileIdx') == 0
    lastImportedFileIdx = 0;
end
if exist('ind_current') == 0
    ind_current = 0;
end

filesExpected = ceil(frames_totalExpected/frames_perFile);

import ScanImageTiffReader.ScanImageTiffReader
ImportWarning_Waiting_Shown = 0;
ImportWarning_OpenAccess_Shown = 0;
clear movie_chunk;
% clear movie_all;
% while lastImportedFileIdx < filesExpected
for iter_file = 1:filesExpected
    if ImportWarning_Waiting_Shown == 0
        disp('Looking for files to import')
        ImportWarning_Waiting_Shown = 1;
    end
    
    dirProps = dir([directory_movie , '\', fileName_movie, '*.tif']);
    
    if size(dirProps,1) > 0
        fileNames = str2mat(dirProps.name);
        fileNames_temp = fileNames;
        fileNames_temp(:,[1:numel(fileName_movie), end-3:end]) = [];
        fileNums = str2num(fileNames_temp);
        
        if size(fileNames,1) > lastImportedFileIdx
            if fopen([directory_movie, '\', fileNames(lastImportedFileIdx+1,:)]) ~= -1
                
                disp(['===== Importing:    ', fileNames(lastImportedFileIdx+1,:), '====='])
                %                 movie_chunk = bigread5([directory_movie, '\', fileNames(lastImportedFileIdx+1,:)]);
                reader = ScanImageTiffReader([directory_movie, '\', fileNames(lastImportedFileIdx+1,:)]);
                movie_chunk = permute(reader.data(),[2,1,3]);
                
                if ~exist('movie_all')
                    movie_all = zeros(size(movie_chunk,1), size(movie_chunk,2), frames_totalExpected , 'int16');
                    movie_all(:,:, ind_current+1:ind_current+size(movie_chunk,3)) = movie_chunk;
                else
                    movie_all(:,:, ind_current+1:ind_current+size(movie_chunk,3)) = movie_chunk;
                end
                ind_current = ind_current+size(movie_chunk,3);
                
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
movie_hash = compute_hashSequence(movie_all,1);
%%
% Import logger
dir_logger = 'D:\RH_local\data\scanimage data\round 5 experiments\mouse 2_6\20210417';

fileName_logger = 'logger.mat';
load([dir_logger '\' fileName_logger]);

fileName_logger_valsROIs = 'logger_valsROIs.mat';
load([dir_logger '\' fileName_logger_valsROIs]);

%%
hash1 = logger.timeSeries(:,28);
hash2 = movie_hash;
%% this part is a demo. Comment out
% % hash1 = logger.timeSeries(:,28);
% hash1 = movie_hash;
% inds_to_nix_1 = sort(randi(length(movie_hash),5,1));
% hash1(1:100) = [];
% hash1(800:8000) = [];
% hash1(inds_to_nix_1) = [];
% hash1(100000:end) = NaN;
% hash1(10001) = hash1(30000);
% 
% hash2 = movie_hash;
% inds_to_nix_2 = sort(randi(length(movie_hash),10,1));
% hash2(inds_to_nix_2) = [];
% hash2(20001) = hash2(40000);

%%
% syntax:   ia_1to2 = indices of sequence 1 to align sequence 1 onto
%           sequence 2 (ie x-axis is sequence 1 indices, and y-values are 
%           sequence 2 indices

[   ia_1to2 , ia_2to1,...
    vals_uniqueToSet1 , inds_uniqueToSet1 ,...
    vals_uniqueToSet2 , inds_uniqueToSet2 ,...
    inds_duplicateSet1Values, inds_duplicateSet2Values]...
= compare_hashSequences(hash1 , hash2);

logger_alignment_vectors.inds_align_logger2movie = ia_1to2;
logger_alignment_vectors.inds_align_movie_to_logger = ia_2to1;
logger_alignment_vectors.vals_unique_toLogger = vals_uniqueToSet1;
logger_alignment_vectors.inds_unique_toLogger = inds_uniqueToSet1;
logger_alignment_vectors.vals_unique_toMovie = vals_uniqueToSet2;
logger_alignment_vectors.inds_unique_toMovie = inds_uniqueToSet2;
logger_alignment_vectors.inds_duplicate_inLogger = inds_duplicateSet1Values;
logger_alignment_vectors.inds_duplicate_inMovie = inds_duplicateSet2Values;
%%
% logger_aligned = 
% logger_valsROI_aligned = 
array1_aligned = indexWithNaNs(hash1  , ia_2to1);
%%
figure; plot(hash2(:,1))
hold on; plot(array1_aligned(:,1))
%%
logger_aligned.timeSeries = indexWithNaNs(logger.timeSeries , ia_2to1);
logger_aligned.timers = indexWithNaNs(logger.timers , ia_2to1);
logger_aligned.decoder = indexWithNaNs(logger.decoder , ia_2to1);
logger_aligned.motionCorrection = indexWithNaNs(logger.motionCorrection , ia_2to1);

logger_valsROIs_aligned = indexWithNaNs(logger_valsROIs , ia_2to1);
%%
figure; plot(hash2(:,1))
hold on; plot(logger_aligned.timeSeries(:,28))
%% Save aligned logger
save_path = 'D:\RH_local\data\scanimage data\round 5 experiments\mouse 2_6\20210417';

save_name = 'logger_aligned.mat';
save([save_path , '\' , save_name] , 'logger_aligned');

save_name = 'logger_valsROIs_aligned.mat';
save([save_path , '\' , save_name] , 'logger_valsROIs_aligned');

save_name = 'logger_alignment_vectors.mat';
save([save_path , '\' , save_name ] , 'logger_alignment_vectors');


        
        
        
