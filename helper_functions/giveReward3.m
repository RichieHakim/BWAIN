function giveReward3(source, solenoidONpref, LEDONpref , solenoid_duration_in_ms, solenoid_delay_in_ms, LED_duration_in_sec, LED_ramping_pref, LED_ramping_duration_in_sec)
% global sesh_reward

% solenoid_duration_in_ms = 50;
% LED_duration_in_sec = 1;
% LED_ramping_pref = 1;
% LED_ramping_duration_in_sec = 0.1;
if solenoidONpref && LEDONpref
    traceDuration = max([(solenoid_duration_in_ms*1.1 + solenoid_delay_in_ms)/1000 , LED_duration_in_sec]) * source.hSI.sesh_reward.Rate;
end
if solenoidONpref && ~LEDONpref
    traceDuration = ((solenoid_duration_in_ms*1.1 + solenoid_delay_in_ms)/1000) * source.hSI.sesh_reward.Rate;
end
if ~solenoidONpref && LEDONpref
    traceDuration = LED_duration_in_sec * source.hSI.sesh_reward.Rate;
end
traceDuration = ceil(traceDuration);

outputTrace_LED = ones(traceDuration,1);
if LEDONpref
    trace_LED_temp = makeLEDoutputPWMTrace(LED_duration_in_sec, LED_ramping_pref, LED_ramping_duration_in_sec);
    outputTrace_LED(1:numel(trace_LED_temp)) = trace_LED_temp;
end

outputTrace_solenoid = zeros(traceDuration,1);
if solenoidONpref
    delayFrames = round((solenoid_delay_in_ms/1000) * source.hSI.sesh_reward.Rate);
    trace_solenoid_temp = zeros(round(solenoid_duration_in_ms/1000 * source.hSI.sesh_reward.Rate * 1.1),1); % make more zeros than there will be ones
    trace_solenoid_temp(2   :    round((solenoid_duration_in_ms/1000) * source.hSI.sesh_reward.Rate) + 1) = 1; % make a bunch of ones, the length of
    outputTrace_solenoid(delayFrames+1: delayFrames + numel(trace_solenoid_temp)) = trace_solenoid_temp;
end

outputTrace_5VForLickDetection = ones(traceDuration,1); % constant output for lick detection
% figure; plot(trace_solenoid_temp)
% size(outputTrace_solenoid)
% figure; plot( outputTrace_solenoid )
% size(outputTrace_5VForLickDetection)
% figure; plot(outputTrace_5VForLickDetection)
% size(outputTrace_LED)

% queueOutputData(source.hSI.sesh_reward, [ outputTrace_solenoid     outputTrace_5VForLickDetection   outputTrace_LED ]); % [ (reward solenoid) (lick detection voltage)]
queueOutputData(source.hSI.sesh_reward, [ outputTrace_solenoid ]); % [ (reward solenoid) (lick detection voltage)]

listener_in = addlistener(source.hSI.sesh_reward,'DataAvailable',@(sesh,event)nullFunction); % check my old function liveTF for how to use this type of function. Here it is necessary to use because we have an analog channel (being used purely as a clock for the digital output channel). This requires a listener for some reason.

startBackground(source.hSI.sesh_reward);

end