function plotUpdatedImagesc(image , clim, figName)
% cursors scrolls back over the traces
% clear hPlot history historyPointer hFig hAx
persistent hPlot hFig hAx

if ~exist('hAx')
        hFig = figure('Name',figName);
        hAx = subplot(1,1,1);

        hPlot = imshow(image);
        set(hAx,'xlimmode','manual',...
            'ylimmode','manual',...
            'zlimmode','manual',...
            'climmode','manual',...
            'alimmode','manual');
        set(hAx, 'CLim', clim);
    
    
    % else if isempty(hAx) || ~isvalid(hAx)
else if isempty(hAx) || ~isvalid(hFig)
        hFig = figure('Name',figName);
        %         hAx = axes('Parent',hFig);
        hAx = subplot(1,1,1);

        hPlot = imshow(image);
        set(hAx,'xlimmode','manual',...
            'ylimmode','manual',...
            'zlimmode','manual',...
            'climmode','manual',...
            'alimmode','manual');
        set(hAx, 'CLim', clim);

    end
end


set(hPlot,'CData',(image));
set(hAx, 'CLim', clim);
drawnow

end