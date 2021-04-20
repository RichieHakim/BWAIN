%% close all ports
% instrreset
%% reward stuff
clear sesh_reward Channel_solenoid Channel_LEDAmplitudeModulation
global sesh_reward
sesh_reward = daq.createSession('ni');
sesh_reward.Rate = 100000;
% Channel_solenoid = addDigitalChannel(sesh, 'PXI1Slot6', 'Port0/Line1', 'OutputOnly');
% addAnalogInputChannel(sesh, 'PXI1Slot6',0,'Voltage'); % this channel is being used only to give a clock to the session. Nothing is being collected on it.
Channel_solenoid = addDigitalChannel(sesh_reward, 'PXI1Slot5', 'Port0/Line1', 'OutputOnly');
Channel_5VoutputForLickDetection = addDigitalChannel(sesh_reward, 'PXI1Slot5', 'Port0/Line2', 'OutputOnly');
Channel_LEDAmplitudeModulation = addDigitalChannel(sesh_reward, 'PXI1Slot5', 'Port0/Line3', 'OutputOnly');
addAnalogInputChannel(sesh_reward, 'PXI1Slot5',0,'Voltage'); % this channel is being used only to give a clock to the session. Nothing is being collected on it.
% Channel_SoundAmplitudeModulation = addAnalogOutputChannel(sesh_reward, 'PXI1Slot5',1,'Voltage'); % this channel is being used only to give a clock to the session. Nothing is being collected on it.

outputSingleScan(sesh_reward, [ 0 1 1])
%%
% %% sound stuff
% clear sesh_sound Channel_SoundAmplitudeModulation
% global sesh_sound
% sesh_sound = daq.createSession('ni');
% sesh_sound.Rate = 100000;
% Channel_FrequencyAmplitudeModulation = addAnalogOutputChannel(sesh_sound, 'PXI1Slot6',0,'Voltage'); 
% Channel_SoundAmplitudeModulation = addAnalogOutputChannel(sesh_sound, 'PXI1Slot6',1,'Voltage'); 
% 
% outputSingleScan(sesh_sound, [ .15 0 ])
% 
% % test sound amplitude modulation
% outputSingleScan(sesh_sound, [ .15 1 ])
% pause(0.5)
% outputSingleScan(sesh_sound, [ .15 2 ])
% pause(0.5)
% outputSingleScan(sesh_sound, [ .15 3.3 ])

% %% sound stuff
% clear task_FrequencyOutputVoltage 
% global task_FrequencyOutputVoltage
% task_FrequencyOutputVoltage = most.util.safeCreateTask('FrequencyOutputVoltage Task');
% task_FrequencyOutputVoltage.createAOVoltageChan('PXI1Slot6',0);
% task_FrequencyOutputVoltage.writeAnalogData(.2);
% 
% 
% % task_FrequencyOutputVoltage.delete()
% 
% %%
% clear task_AnalogOutputVoltage 
% global task_AnalogOutputVoltage
% task_AnalogOutputVoltage = most.util.safeCreateTask('AnalogOutputVoltage Task');
% task_AnalogOutputVoltage.createAOVoltageChan('PXI1Slot6',1);
% task_AnalogOutputVoltage.writeAnalogData(1);


% task_AnalogOutputVoltage.delete()






% %% LED stuff
% 
% clear sesh_LED Channel_LEDAmplitudeModulation
% global sesh_LED
% sesh_LED = daq.createSession('ni');
% sesh_LED.Rate = 100000;
% addAnalogInputChannel(sesh_LED, 'PXI1Slot5',1,'Voltage'); % this channel is being used only to give a clock to the session. Nothing is being collected on it.
% % Channel_SoundAmplitudeModulation = addAnalogOutputChannel(sesh_reward, 'PXI1Slot5',1,'Voltage'); % this channel is being used only to give a clock to the session. Nothing is being collected on it.
% Channel_LEDAmplitudeModulation = addDigitalChannel(sesh_LED, 'PXI1Slot5', 'Port0/Line3', 'OutputOnly');
% 
% outputSingleScan(sesh_LED, [ 1 ])


% setTeensyAnalogInputs(0,0)



% %% arduino stuff
% 
% % Simple Arduino connection example
% 
% % function arduinoExample()
% baud = 9600;
% ard = ArduinoConnection(@messageHandler, baud);
% 
% % alternative: ard = ArduinoConnection(@(m) disp(m), baud); % this prints the message
% % alternative: ard = ArduinoConnection(@(m), baud); % this doesn't
% 
% % 
% % while true
% % 	msgToSend = 'Foo';
% % 	ard.writeString(msgToSend);
% % 	fprintf('Sent message: %s\n', msgToSend)
% % 	pause(5)
% % end
% % end
% ard.writeString(['SetMaxFreq ' , num2str(1000)])
% ard.writeString(['SetMinFreq ' , num2str(30000)])
% 
% ard.writeString(['SetAmplitude ' , num2str(1000)])
% 
% for ii = 1:20
% ard.writeString(['SetFrequency ' , num2str(ii*1000)])
% pause(0.1)
% end
% ard.writeString(['SetFrequency ' , num2str(3000)])
% % ard.writeString(['GetFrequencyFromPin'])
% % ard.writeString(['GetAmplitudeFromPin'])
% 
% 
% 
% function messageHandler(msg)
% 	fprintf('Recived message: %s\n', msg)
% end

