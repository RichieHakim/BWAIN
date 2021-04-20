function setTeensyAnalogInputs(frequencyPin, amplitudePin)
% set amplitdue to between 0-3.3
if amplitudePin > 3.3
    amplitudePin = 3.3;
end
if amplitudePin < 0 
    amplitudePin = 0;
end

if frequencyPin > 3.3
    frequencyPin = 3.3;
end
if frequencyPin < 0 
    frequencyPin = 0;
end

global sesh_sound

outputSingleScan(sesh_sound, [ frequencyPin , amplitudePin ])

end