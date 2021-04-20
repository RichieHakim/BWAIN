function outputFrequency = convert_voltage_to_frequency(voltage, voltage_max, range_frequency)
%% This function is meant to mimic (exactly) the function in the teensySineGenerator that is converting
%  voltage to a frequency, so that it can be logged
min_offset = log2(range_frequency(1));
max_offset = log2(range_frequency(2));
outputFrequency = 2.^((voltage/voltage_max) *(max_offset-min_offset)+min_offset);
end