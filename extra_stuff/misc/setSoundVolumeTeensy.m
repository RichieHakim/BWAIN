function setSoundVolumeTeensy(amplitude)
% set amplitdue to between 0-3.3
if amplitude > 3.3
    amplitude = 3.3;
end
if amplitude < 0 
    amplitude = 0;
end

global sesh_sound

outputSingleScan(sesh_sound, [ amplitude])

end