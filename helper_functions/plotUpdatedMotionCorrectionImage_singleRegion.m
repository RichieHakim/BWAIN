function plotUpdatedMotionCorrectionImage_singleRegion(Images_MC_ref , Images_MC_moving , Images_MC_corrected, figName)
% deadFrames is the number of frames that should be blanked out as the
% cursors scrolls back over the traces
% clear hPlot history historyPointer hFig hAx
persistent hPlot hFig hAx

if ~exist('hAx')
        hFig = figure('Name',figName);
        %         hAx = axes('Parent',hFig);
        for ii = 1:3
            hAx{ii} = subplot(3,1,ii);

            hPlot{ii} = imshow(Images_MC_ref);
            set(hAx{ii},'xlimmode','manual',...
                'ylimmode','manual',...
                'zlimmode','manual',...
                'climmode','manual',...
                'alimmode','manual');
%             set(hAx{ii}, 'CLim', [-15 4500]);
%             set(hAx{ii}, 'CLim', [-15 max(Images_MC_ref{mod(ii-1,4)+1}(:))]);
            set(hAx{ii}, 'CLim', [-15 , max(Images_MC_ref(:))]);
        end
    
    
    % else if isempty(hAx) || ~isvalid(hAx)
else if isempty(hAx) || ~isvalid(hFig)
        hFig = figure('Name',figName);
        %         hAx = axes('Parent',hFig);
        for ii = 1:3
            hAx{ii} = subplot(3,1,ii);

            hPlot{ii} = imshow(Images_MC_ref);
            set(hAx{ii},'xlimmode','manual',...
                'ylimmode','manual',...
                'zlimmode','manual',...
                'climmode','manual',...
                'alimmode','manual');
%             set(hAx{ii}, 'CLim', [-15 4500]);
%             set(hAx{ii}, 'CLim', [-15 max(Images_MC_ref{mod(ii-1,4)+1}(:))]);
            set(hAx{ii}, 'CLim', [-15 , max(Images_MC_ref(:))]);
        end
    end
end

% Images_All = zeros(size(Images_MC_ref{1},1) , size(Images_MC_ref{1},2) , 12);
Images_All = {};
Images_All{1} = Images_MC_ref;

Images_All{2} = Images_MC_moving;

Images_All{3} = Images_MC_corrected;

for ii = 1:3
    %     subplot(4,3,ii,hAx{ii});
    
    %   hPlot = imagesc(hAx,Image);
    %     set(hPlot{ii},'CData',uint8(Images_All(:,:,ii)));
    set(hPlot{ii},'CData',(Images_All{ii}));
end
%   hPlot(1).LineWidth = 2;
% figure; imagesc(Images_All{1})

end