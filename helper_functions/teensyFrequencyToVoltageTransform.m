function outputVoltage = teensyFrequencyToVoltageTransform(frequency, range_freqOutput, range_teensyInputVoltage)
range_freqOutput = [1000 18000]; % this is set in the teensy code (Ofer made it)
range_teensyInputVoltage = [0 3.3]; % using a teensy 3.5 currently

if frequency < range_freqOutput(1)
    frequency = range_freqOutput(1);
end
if frequency > range_freqOutput(2)
    frequency = range_freqOutput(2);
end

frequency_scaled = (frequency-range_freqOutput(1)) / (range_freqOutput(2)-range_freqOutput(1)); % a value between 0 and 1 showing where between the min and max of range_freqOutput the desired frequency is.
outputVoltage = range_teensyInputVoltage(1) + frequency_scaled*(range_teensyInputVoltage(2) - range_teensyInputVoltage(1));
    
end