%% Plot image with same shape as the channel 1 iamge
% stack = load('D:\RH_local\data\BMI_cage_g2F\mouse_g2FB\20221112\analysis_data\stack.mat');
stack = load('D:\RH_local\data\BMI_cage_g2F\mouse_g2FB\20221111\analysis_data\stack_sparse.mat');
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