function plotLEDs(cursor_vel,cursor_pos,target_pos,npos,thresh,plot_window)
persistent ledplot s1 s2 cursor_window hCursor hVel

if isempty(ledplot) || ~isvalid(ledplot)
    ledplot = figure;
    
    s1 = subplot(2,1,1);
    xlim([1,npos])
    ylim([-0.5,1.5])
    xticks([])
    hold on
    hCursor = scatter(s1,[cursor_pos-thresh+1:cursor_pos],zeros(thresh,1),20,'b','filled');  %plot  current position and target position as dots
    hTarget = scatter(s1,[target_pos-thresh+1:target_pos], ones(thresh,1),20,'r','filled');
    
    s2 = subplot(2,1,2);
    hold on
    xlim([1,plot_window])
    ylim([-5e-2,5e-2])
    xticks([])
    ylabel('cursor velocity')
    cursor_window = nan(plot_window,1);
    hVel = plot(s2,1:plot_window,cursor_window);
end
    

i                 = find(isnan(cursor_window),1);
cursor_window(i)  = cursor_vel;
cursor_window(mod(i+1:i+10,plot_window)+1) = nan;


set(hVel   ,'YData',cursor_window)
set(hCursor,'XData',[cursor_pos-thresh+1 : cursor_pos])


