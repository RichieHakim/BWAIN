%% Plot image with same shape as the channel 1 iamge
stack = load('D:\RH_local\data\round_6_experiments\mouse_1_19\scanimage_data\zstack\stack.mat');
channel1_image = findall(groot,'Type','Figure','Name','Channel 1');
f = figure;
h = axes;
f.Position = channel1_image.Position;
f.InnerPosition = channel1_image.InnerPosition;
f.OuterPosition = channel1_image.OuterPosition;
imSize = size(stack.stack.stack_avg);
centerSlice = ceil(imSize(1)/2);
imagesc(squeeze(stack.stack.stack_avg(centerSlice,:,:)));
axis off;
set(h,'position',[0.02 0.001 .96 1]);
set(f, 'MenuBar', 'none');
set(f, 'ToolBar', 'none');