function plotUpdatingPlot(outputVals, refresh_period, figName, figTitle)
persistent hPlot hFig hAx counter_refresh

if ~exist('refresh_period')
    refresh_period = 0;
end
if ~exist('figName')
    figName = '';
end

if ~exist('hFig')
    hFig = figure('Name',figName);
    for ii = 1:numel(outputVals)
        hPlot(ii) = plot(outputVals{ii});
        hold on
    end
    counter_refresh = 0;
else
    if isempty(hFig) || ~isvalid(hFig)
        hFig = figure('Name',figName);
        for ii = 1:numel(outputVals)
            hPlot(ii) = plot(outputVals{ii});
            hold on
        end
        counter_refresh = 0;
    end
end

if counter_refresh >= refresh_period
    counter_refresh = 0;
    for ii = 1:numel(outputVals)
        set(hPlot(ii) , 'YData', outputVals{ii})
    end
else
    counter_refresh = counter_refresh + 1;
end

if exist('figTitle')
    figTitle = title(hAx,figTitle);
end

end