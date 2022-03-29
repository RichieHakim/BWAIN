path_stack = 'D:\RH_local\data\round_6_experiments\mouse_1_19\scanimage_data\zstack\file_00003_00001.tif';
path_save = 'D:\RH_local\data\round_6_experiments\mouse_1_19\scanimage_data\zstack\stack.mat';

num_frames_per_slice = 60;
num_slices = 5;
num_volumes = 10;
step_size_um = 3;
centered = 1;

frames_to_discard_per_slice = 30; % discards this many frames from beginning due to jitter from piezo

% zCorrPtile = 10;

stack.num_frames_per_slice = num_frames_per_slice;
stack.step_size_um = step_size_um;
stack.centered = centered;

import ScanImageTiffReader.ScanImageTiffReader
reader = ScanImageTiffReader(path_stack);
slices_raw = permute(reader.data(),[3,2,1]);

slices_rs = reshape(slices_raw, num_frames_per_slice, num_slices, num_volumes, size(slices_raw,2), size(slices_raw,3));
slices_rs = slices_rs(frames_to_discard_per_slice+1:end,:,:,:,:);
stack.stack_avg = squeeze(squeeze(mean(mean(slices_rs, 1), 3)));
% stack.stack_avg = squeeze(prctile(reshape(slices_raw, num_slices, num_frames_per_slice, size(slices_raw,2), size(slices_raw,3)), zCorrPtile, 2));

save(path_save, 'stack')

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