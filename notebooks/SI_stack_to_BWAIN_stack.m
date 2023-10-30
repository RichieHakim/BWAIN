% path_mouse = 'D:\RH_local\data\cage_0916\mouse_0916N\20231021\';
% path_mouse = 'D:\RH_local\data\cage_0908\mouse_0908\20231021\'
% path_mouse = 'D:\RH_local\data\cage_0914\mouse_0914\20231021\'


path_stack = fullfile(path_mouse, 'scanimage_data\zstack\zstack_960_00001_00001.tif');
path_save = fullfile(path_mouse, 'analysis_data\stack_960nm_dense.mat');
path_save_sparse = fullfile(path_mouse, 'analysis_data\stack_sparse.mat');

num_frames_per_slice = 60;
num_slices = 41; % % 03262023 Increase stack range, 25 -> 41
num_volumes = 10;
step_size_um = 0.8;
centered = 1;
% FAST
% STEP
% # Frames/File 100000

% % For zstack_bulk:
% num_frames_per_slice = 60;
% num_slices = 101; % 200 um stack
% num_volumes = 3;
% step_size_um = 2;
% centered = 1;
% % FAST
% % STEP
% % # Frames/File 100000

frames_to_discard_per_slice = 30; % discards this many frames from beginning due to jitter from piezo

stack.num_frames_per_slice = num_frames_per_slice;
stack.step_size_um = step_size_um;
stack.centered = centered;

import ScanImageTiffReader.ScanImageTiffReader
reader = ScanImageTiffReader(path_stack);
slices_raw = permute(reader.data(),[3,2,1]);

slices_rs = reshape(slices_raw, num_frames_per_slice, num_slices, num_volumes, size(slices_raw,2), size(slices_raw,3));
slices_rs = slices_rs(frames_to_discard_per_slice+1:end,:,:,:,:);
% slices_rs = slices_rs(frames_to_discard_per_slice+1:end,:,1:5,:,:);
disp(size(slices_rs))

stack.stack_avg = squeeze(squeeze(mean(mean(slices_rs, 1), 3)));
% stack.stack_avg = squeeze(prctile(reshape(slices_raw, num_slices, num_frames_per_slice, size(slices_raw,2), size(slices_raw,3)), zCorrPtile, 2));

save(path_save, 'stack')
%%
range = ((num_slices - 1) * step_size_um)/2;
disp(['range of slices: ', num2str(range)])

%%
for ii = 1:size(stack.stack_avg, 1)
    figure; imagesc(squeeze(stack.stack_avg(ii,:,:)))
end
%%

[input,inputCropped, ~, ~] = make_fft_for_MC(squeeze(stack.stack_avg(2,:,:)));
figure; imagesc(inputCropped)
inputMasked = maskImage(input, 20,10);
figure; imagesc(abs(fft2(inputMasked)))

%%
stack_sparse = stack;
stack_sparse.step_size_um = 4;
stack_sparse.step_numIdx = ceil(stack_sparse.step_size_um  / step_size_um);
stack_sparse.idx_center = ceil(num_slices / 2);

idx = stack_sparse.idx_center;
n = stack_sparse.step_numIdx ;
idx_slices = uint16([idx-n*2, idx-n*1, idx-n*0, idx+n*1, idx+n*2]);
    
stack_sparse.stack_avg = stack_sparse.stack_avg(idx_slices,:,:);

stack_sparse.params.num_frames_per_slice = num_frames_per_slice;
stack_sparse.params.num_slices = num_slices;
stack_sparse.params.num_volumes = num_volumes;
stack_sparse.params.step_size_um = step_size_um;
stack_sparse.params.centered = centered;

%%
figure()
imagesc(squeeze(stack_sparse.stack_avg(3,:,:)))

%%
save(path_save_sparse, 'stack_sparse')
%%
function image = maskImage(image, border_outer, border_inner)
    border_outer = int64(border_outer);
    border_inner = int64(border_inner);
    imDim = size(image);
    mid = floor(imDim./2);

    image(1:border_outer,:) = 0;
    image(end-border_outer:end,:) = 0;
    image(:, 1:border_outer) = 0;
    image(:, end-border_outer:end) = 0;

    image(mid(1)-border_inner:mid(1)+border_inner, mid(2)-border_inner:mid(2)+border_inner) = 0;
end

function [refIm_crop_conjFFT_shift, refIm_crop, indRange_y_crop, indRange_x_crop] = make_fft_for_MC(refIm)

    refIm = single(refIm);
    % crop_factor = 5;
    crop_size = 256; % MAKE A POWER OF 2! eg 32,64,128,256,512

    length_x = size(refIm,2);
    length_y = size(refIm,1);
    middle_x = size(refIm,2)/2;
    middle_y = size(refIm,1)/2;

    % indRange_y_crop = [round(middle_y - length_y/crop_factor) , round(middle_y + length_y/crop_factor) ];
    % indRange_x_crop = [round(middle_x - length_y/crop_factor) , round(middle_x + length_y/crop_factor) ];

    indRange_y_crop = [round(middle_y - (crop_size/2-1)) , round(middle_y + (crop_size/2)) ];
    indRange_x_crop = [round(middle_x - (crop_size/2-1)) , round(middle_x + (crop_size/2)) ];

    refIm_crop = refIm(indRange_y_crop(1) : indRange_y_crop(2) , indRange_x_crop(1) : indRange_x_crop(2)) ;

    refIm_crop_conjFFT = conj(fft2(refIm_crop));
    refIm_crop_conjFFT_shift = fftshift(refIm_crop_conjFFT);

    % size(refIm_crop_conjFFT_shift,1);
    % if mod(size(refIm_crop_conjFFT_shift,1) , 2) == 0
    %     disp('RH WARNING: y length of refIm_crop_conjFFT_shift is even. Something is very wrong')
    % end
    % if mod(size(refIm_crop_conjFFT_shift,2) , 2) == 0
    %     disp('RH WARNING: x length of refIm_crop_conjFFT_shift is even. Something is very wrong')
    % end

%         refIm_crop_conjFFT_shift_centerIdx = ceil(size(refIm_crop_conjFFT_shift)/2);
end