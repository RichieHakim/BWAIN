function outputTrace = makeLEDoutputPWMTrace(durationInSeconds, ramping_pref, ramp_duration)
% The buckpuck I'm using takes 5V in for OFF and 0V for ON.
% This uses pulse width modulation because I ran out of analog outputs.

global sesh_reward

Fs_outputSampleRate = sesh_reward.Rate; % in Hz
xaxis = [1/Fs_outputSampleRate : 1/Fs_outputSampleRate : durationInSeconds];
signal_max_amplitude = .2; % I'd leave it low. Problems arise if you don't. The ramp looks jumpy

signal = signal_max_amplitude*ones(numel(xaxis),1);

if ramping_pref
    signal(round(1:ramp_duration*Fs_outputSampleRate)) = (1/(ramp_duration*Fs_outputSampleRate) : 1/(ramp_duration*Fs_outputSampleRate) : 1) * signal_max_amplitude;
    signal(round(end - (ramp_duration*Fs_outputSampleRate) + 1 : end)) = (fliplr(1/(ramp_duration*Fs_outputSampleRate) : 1/(ramp_duration*Fs_outputSampleRate) : 1)) * signal_max_amplitude;
end

signal = 1-signal;

signal_pwm = pwm_RH(signal', Fs_outputSampleRate, 500);
outputTrace = single(signal_pwm)';

end