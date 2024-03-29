%% Plot image with same shape as the channel 1 iamge
% stack = load('D:\RH_local\data\cage_0916\mouse_0916N\20231023\analysis_data\stack_sparse.mat');
% stack = load('D:\RH_local\data\cage_0908\mouse_0908\20231022\analysis_data\stack_sparse.mat');
stack = load('D:\RH_local\data\cage_0914\mouse_0914\20231022\analysis_data\stack_sparse.mat');


channel1_image = findall(groot,'Type','Figure','Name','Channel 1');
f = figure;
h = axes;
f.Position = channel1_image.Position;
f.InnerPosition = channel1_image.InnerPosition;
f.OuterPosition = channel1_image.OuterPosition;

% % 2022/10/10 Added stack_warped
% % 2022/11/11 Added stack_sparse
% type_stack = 'stack';
% type_stack = 'stack_warped';
type_stack = 'stack_sparse';

imSize = size(stack.(type_stack).stack_avg);
centerSlice = ceil(imSize(1)/2);
imagesc(squeeze(stack.(type_stack).stack_avg(centerSlice,:,:)));
axis off;
set(h,'position',[0.02 0.001 .96 1]);
set(f, 'MenuBar', 'none');
set(f, 'ToolBar', 'none');
%%
reader = ScanImageTiffReader.ScanImageTiffReader(['D:\RH_local\data\BMI_cage_g8Test\mouse_g8t', '\', 'testRun_00001_00050.tif']);
movie_chunk = permute(reader.data(),[2,1,3]);
im = squeeze(mean(movie_chunk, 3));

channel1_image = findall(groot,'Type','Figure','Name','Channel 1');
f = figure;
h = axes;
f.Position = channel1_image.Position;
f.InnerPosition = channel1_image.InnerPosition;
f.OuterPosition = channel1_image.OuterPosition;

imSize = size(im);
centerSlice = ceil(imSize(1)/2);
imagesc(im);
axis off;
set(h,'position',[0.02 0.001 .96 1]);
set(f, 'MenuBar', 'none');
set(f, 'ToolBar', 'none');
