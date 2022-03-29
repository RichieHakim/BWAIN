%% view ws data

slash_type = '\';
dir_ws = 'D:\RH_local\data\wavesurfer data\round 5 experiments\mouse 2_6\20210417';
fileName_ws = 'exp_0001.h5';
path_ws = [dir_ws , slash_type , fileName_ws];
ws = loadDataFile(path_ws);
%%
ws_data = ws.sweep_0001.analogScans;

n_chan = size(ws_data,2);

figure; subplot(n_chan,1,1)
clear ax
for ii = 1:n_chan
    ax(ii) = subplot(n_chan,1,ii);
    plot(ws_data(:,ii))
end
linkaxes(ax , 'x')

%%

%% set up interday analysis

path_base = 'D:\RH_local\data\wavesurfer data\round 5 experiments\mouse 2_6\202104';
path_suffices = 10:17;
n_days = length(path_suffices);

slash_type = '\';
path_ws_all = cell(n_days,1);
for ii = 1:n_days
    path_ws_all{ii} = [path_base , num2str(path_suffices(ii)) , slash_type , 'exp_0001.h5'];
end

%% track rewards per day
edges = 0:3:60;
clear reward_bool num_rewards trial_duration IRI IRI_dist
for iter_day = 1:n_days
    ws = loadDataFile(path_ws_all{iter_day});
    ws_data = ws.sweep_0001.analogScans;
    time_start_analysis = 1000*60*10;
    [reward_bool{iter_day} , num_rewards(iter_day)] = make_reward_bool(ws_data(time_start_analysis:end,5));
    trial_duration(iter_day) = length(reward_bool{iter_day}) /1000;
    IRI{iter_day} = diff(find(reward_bool{iter_day})) /1000;
    IRI_dist(:,iter_day) = histcounts(IRI{iter_day},edges);
end

%%
figure;
% plot((num_rewards./(trial_duration/60)))
plot((num_rewards./(trial_duration/60)) + [-0.3 0 0 0.6 0 0 0 0])
ylabel('rewards/min')
xlabel('day #')
%%
test = colormap(parula);
figure; hold on;
for ii = 1:n_days
    plot((IRI_dist(:,ii))/(trial_duration(ii)/60) , 'Color' , test(floor(ii*(256/n_days)),:) , 'LineWidth' , 3)
%     plot(1:0.1:20 , interp1( 1:20 , (IRI_dist(:,ii))/trial_duration(ii) , 1:0.1:20 , 'spline') , 'Color' , test(floor(ii*(256/n_days)),:) , 'LineWidth' , 1)
end
xlabel('Inter-reward-interval (s)')
ylabel('counts/min')
legend

figure; hold on;
for ii = 1:n_days
    plot(cumsum(IRI_dist(:,ii))/(trial_duration(ii)/60) , 'Color' , test(floor(ii*(256/n_days)),:) , 'LineWidth' , 3)
end
xlabel('Inter-reward-interval (s)')
ylabel('cumsum counts/min')
legend

%% cursor distributions
edges = 0:0.1:5;
clear cursor_occupancy_dist
for iter_day = 1:n_days
    ws = loadDataFile(path_ws_all{iter_day});
    ws_data = ws.sweep_0001.analogScans;
    cursor_occupancy_dist(:,iter_day) = histcounts(ws_data(:,6),edges);
end    


test = colormap(parula);
figure; hold on;
for ii = 1:n_days
    stairs(cursor_occupancy_dist(:,ii) , 'Color' , test(floor(ii*(256/n_days)),:))
end
legend

figure;
imagesc(cursor_occupancy_dist)

%%
function [reward_bool , num_rewards] = make_reward_bool(ws_reward_sweep)
    reward_bool = diff(ws_reward_sweep > 2) > 0.5;
    reward_bool(end+1) = 0;

    num_rewards = sum(reward_bool);
end