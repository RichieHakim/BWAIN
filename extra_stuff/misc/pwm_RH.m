function output_signal = pwm_RH(input_signal, Fs_signal, Fs_carrier)
% input_signal = sin(1:0.0001:10);

% Fs_signal Frequency of Carrier Signal (Sawtooth)
% Fs_carrier Frequency of Message Signal (Sinusoidal)

a = 1; % Amplitude of Carrier Signal
duration = numel(input_signal);
t = 1/Fs_signal:1/Fs_signal:duration/Fs_signal;

carrier_signal = a.*sawtooth(2*pi*Fs_carrier*t);
carrier_signal = carrier_signal - min(carrier_signal);
carrier_signal = carrier_signal .* (1/max(carrier_signal));
pwm = input_signal > carrier_signal;

% figure; plot(input_signal); hold on; plot(carrier_signal); 
% hold on; plot(pwm)

output_signal = pwm;
% max(carrier_signal)