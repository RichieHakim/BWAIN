function plotUpdatedOutput4(outputVals, length_history, xAxisScaleFactor, figName, deadFrames, refresh_period, figTitle)
% deadFrames is the number of frames that should be blanked out as the
% cursors scrolls back over the traces
% clear hPlot history historyPointer hFig hAx
persistent hPlot history historyPointer hFig hAx
% clear history
if ~exist('hAx')
    hFig = figure('Name',figName);
    hAx = axes('Parent',hFig,'XLim',[1,length_history]);
    hPlot = plot(rand(numel(outputVals)),rand(numel(outputVals)));
else if isempty(hAx) || ~isvalid(hAx)
        hFig = figure('Name',figName);
        hAx = axes('Parent',hFig,'XLim',[1,length_history]);
        hPlot = plot(rand(numel(outputVals)),rand(numel(outputVals)));
    end
end

if isempty(history) || length(history) ~= length_history
    history = nan(length_history, numel(outputVals));
end

if isempty(historyPointer) || historyPointer > length_history
    historyPointer = 1;
end

if size(history,2) ~= numel(outputVals)
    history = outputVals;
else
    history(historyPointer,:) = outputVals;
end

historyPointer = historyPointer + 1;
if historyPointer > length_history
    historyPointer = 1;
end

history(mod(historyPointer:(historyPointer + deadFrames), length_history) + 1,:) = NaN;

if mod(historyPointer,refresh_period) ==0

xaxis = (1:length_history)/xAxisScaleFactor;
    for ii = 1:numel(outputVals)
        set(hPlot(ii), 'XData', xaxis , 'YData', history(:,ii))
    end
end

% if exist('figTitle')
%     figTitle = title(hAx,figTitle);
% end

end