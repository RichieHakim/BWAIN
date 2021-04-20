function outputVoltage = convert_cursor_to_voltage(cursor, range_cursor, voltage_at_threshold)

range_teensyInputVoltage = [0 voltage_at_threshold]; % using a teensy 3.5 currently

cursor(cursor < range_cursor(1)) = range_cursor(1);
cursor(cursor > range_cursor(2)) = range_cursor(2);

cursor_scaled = (cursor-range_cursor(1)) / (range_cursor(2)-range_cursor(1)); % a value between 0 and 1 showing where between the min and max of range_freqOutput the desired frequency is.
outputVoltage = range_teensyInputVoltage(1) + cursor_scaled*(range_teensyInputVoltage(2) - range_teensyInputVoltage(1));

end