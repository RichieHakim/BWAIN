function plotUpdatedMotionCorrectionImages(Images_MC_ref , Images_MC_moving , Images_MC_corrected, figName)
% deadFrames is the number of frames that should be blanked out as the
% cursors scrolls back over the traces
% clear hPlot history historyPointer hFig hAx
persistent hPlot hFig hAx

if ~exist('hAx')
        hFig = figure('Name',figName);
        %         hAx = axes('Parent',hFig);
        for ii = 1:12
            hAx{ii} = subplot(3,4,ii);

            hPlot{ii} = imshow(Images_MC_ref{1});
            set(hAx{ii},'xlimmode','manual',...
                'ylimmode','manual',...
                'zlimmode','manual',...
                'climmode','manual',...
                'alimmode','manual');
            set(hAx{ii}, 'CLim', [-15 4500]);
            set(hAx{ii}, 'CLim', [-15 max(Images_MC_ref{mod(ii-1,4)+1}(:))]);
        end
    
    
    % else if isempty(hAx) || ~isvalid(hAx)
else if isempty(hAx) || ~isvalid(hFig)
        hFig = figure('Name',figName);
        %         hAx = axes('Parent',hFig);
        for ii = 1:12
            hAx{ii} = subplot(3,4,ii);

            hPlot{ii} = imshow(Images_MC_ref{1});
            set(hAx{ii},'xlimmode','manual',...
                'ylimmode','manual',...
                'zlimmode','manual',...
                'climmode','manual',...
                'alimmode','manual');
            set(hAx{ii}, 'CLim', [-15 4500]);
            set(hAx{ii}, 'CLim', [-15 max(Images_MC_ref{mod(ii-1,4)+1}(:))]);
        end
    end
end

% Images_All = zeros(size(Images_MC_ref{1},1) , size(Images_MC_ref{1},2) , 12);
Images_All = {};
for ii = 1:4
    %         Images_All(:,:,ii + 0) = Images_MC_ref{ii};
    Images_All{ii + 0} = Images_MC_ref{ii};
end
for ii = 1:4
    %         Images_All(:,:,ii + 4) = Images_MC_corrected{ii};
    Images_All{ii + 4} = Images_MC_moving{ii};
end
for ii = 1:4
    %         Images_All(:,:,ii + 8) = Images_MC_moving{ii};
    Images_All{ii + 8} = Images_MC_corrected{ii};
end

for ii = 1:12
    %     subplot(4,3,ii,hAx{ii});
    
    %   hPlot = imagesc(hAx,Image);
    %     set(hPlot{ii},'CData',uint8(Images_All(:,:,ii)));
    set(hPlot{ii},'CData',(Images_All{ii}));
end
%   hPlot(1).LineWidth = 2;
% figure; imagesc(Images_All{1})

end