function show_weights(currentImage,image_footprints_weighted,frameNum)
persistent imageplot imageax  imageh  image_window %currax currh curr_window
if isempty(imageplot) || ~isvalid(imageplot)
    imageplot = figure;
    
    imageax   = subplot(1,1,1);                                                                                     %initialize an axis for the weighted image
    imageh    = imagesc(bsxfun( @times , image_footprints_weighted , single(currentImage) ));                       %initialize an object object onto that axis
    image_window = nan([size(currentImage),5]);                                                                     %initialize a window to average values over (this parallels number of windows in smoothing)
    
%     currax    = subplot(1,2,2);
%     currh     = image(single(currentImage));
%     curr_window  = nan([size(currentImage),5]);
%     colormap(currax,'bone')
end

image_window(:,:,mod(frameNum,5)+1)  = bsxfun( @times , image_footprints_weighted , single(currentImage) );         %store the current frame in a rolling loop of 5 frames (window size)
%curr_window(:,:,mod(frameNum,5)+1)   = currentImage;

imageh.CData = mean(image_window,3);                                                                                %change the image data to current image
%currh.CData  = single(mean(curr_window,3));

colormap(imageax,bluewhitered(imageax))                                                                             %update the colormap with the current image so that zero stays white
