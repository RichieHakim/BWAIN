%% Make expSettings file
% There are two options:
% 1. Pull from params.json file. Python simulations can output this.
% 2. Pull from expParams file. BMI experiments can output this.

% The online experiment script is expected to fill in all parameters
% required, then THIS expSettings file will OVERWRITE all the parameters
% with matching fields.

%% Pull from params.json file
% only pulls specific fields
filepath_json = 'D:\RH_local\data\cage_0315\mouse_0315N\20230423\analysis_data\day0_analysis\params_mouse0315N_2.json';
json_params = json_load(filepath_json);

pp = struct(); %% 'params pulled'

pp.dFoF = struct();
pp.dFoF.duration_rolling = json_params.dFoF.win_rolling_percentile;
pp.dFoF.ptile            = json_params.dFoF.percentile_baseline;
pp.dFoF.interval_update  = json_params.dFoF.roll_stride;
pp.dFoF.frac_neuropil    = json_params.dFoF.neuropil_fraction;
pp.dFoF.additive_offset  = json_params.dFoF.channelOffset_correction;

pp.cursor = struct();
pp.cursor.factor_to_use                    = json_params.simulation.idx_factor;
pp.cursor.angle_power                      = json_params.simulation.power;
pp.cursor.thresh_quiescence_cursorDecoder  = json_params.simulation.thresh_quiescence_cursorDecoder;
pp.cursor.thresh_quiescence_cursorMag      = json_params.simulation.thresh_quiescence_cursorMag;
pp.cursor.angle_power                      = json_params.simulation.power;
pp.cursor.win_smooth_cursor                = json_params.simulation.win_smooth_cursor;

pp.trial = struct();
pp.trial.duration_quiescence_hold = json_params.simulation.duration_quiescence_hold;
pp.trial.duration_threshold       = json_params.simulation.duration_threshold_hold;

%% Pull from expParams file
% pulls all fields, but then you can delete ones you want
filepath_expParams = 'D:\RH_local\data\cage_0322\mouse_0322R\20230425\analysis_data\expParams.mat';
clear expParams
load(filepath_expParams);
assert(exist('expParams') > 0)

pp = expParams.params;

pp = rmfield(pp, 'mode');
pp = rmfield(pp, 'directory');
pp = rmfield(pp, 'blocks');

pp.timing = rmfield(pp.timing, 'duration_session');

%%% Optionally overwrite some parameters
% pp.cursor.factor_to_use = 3;


%% Save expSettings file

expSettings = pp;

filepath_save = 'D:\RH_local\data\cage_0315\mouse_0315N\20230423\analysis_data\day0_analysis\expSettings.mat';
save(filepath_save, 'expSettings')

%%
% load('C:\Users\Rich Hakim\Downloads\expSettings.mat');

