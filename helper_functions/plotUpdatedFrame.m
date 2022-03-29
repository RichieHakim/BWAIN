function plotUpdatedFrame(Image , figName)
% deadFrames is the number of frames that should be blanked out as the
% cursors scrolls back over the traces
% clear hPlot history historyPointer hFig hAx
persistent hPlot hFig hAx

% size(Image)
% clear hAx
if ~exist('hAx')
        hFig = figure('Name',figName);
        %         hAx = axes('Parent',hFig);
        for ii = 1:1
            hAx{ii} = subplot(1,1,ii);

            hPlot{ii} = imagesc(Image);
            set(hAx{ii},'xlimmode','manual',...
                'ylimmode','manual',...
                'zlimmode','manual',...
                'climmode','manual',...
                'alimmode','manual');
%             set(hAx{ii}, 'CLim', [-15 4500]);
%             set(hAx{ii}, 'CLim', [-15 max(Images_MC_ref{mod(ii-1,4)+1}(:))]);
            set(hAx{ii}, 'CLim', [-15 , max(Image(:))]);
        end
    
    
    % else if isempty(hAx) || ~isvalid(hAx)
else if isempty(hAx) || ~isvalid(hFig)
        hFig = figure('Name',figName);
        %         hAx = axes('Parent',hFig);
        for ii = 1:1
            hAx{ii} = subplot(1,1,ii);

            hPlot{ii} = imagesc(Image);
            set(hAx{ii},'xlimmode','manual',...
                'ylimmode','manual',...
                'zlimmode','manual',...
                'climmode','manual',...
                'alimmode','manual');
%             set(hAx{ii}, 'CLim', [-15 4500]);
%             set(hAx{ii}, 'CLim', [-15 max(Images_MC_ref{mod(ii-1,4)+1}(:))]);
            set(hAx{ii}, 'CLim', [-15 , max(Image(:))]);
        end
    end
end


for ii = 1:1
    %     subplot(4,3,ii,hAx{ii});
    
    %   hPlot = imagesc(hAx,Image);
    %     set(hPlot{ii},'CData',uint8(Images_All(:,:,ii)));
%     size(hPlot)
    set(hPlot{ii},'CData',Image);
%     figure; imagesc(Image)
end
%   hPlot(1).LineWidth = 2;
% figure; imagesc(Images_All{1})

end