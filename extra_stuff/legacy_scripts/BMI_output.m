function teensyOutput = BMI_output(vals, varargin)

global bmiCounter
bmiCounter = bmiCounter + 1;
if (bmiCounter == 1)
    disp('BMI COunter first pass');
    disp(varargin{1}{2});
end

global teensyOutput
% teensyOutput
% HARD CODED CONSTRAINTS ON OUTPUT VOLTAGE FOR TEENSY 3.5
if teensyOutput > 3.3
    teensyOutput = 3.3;
    %     warning('CURSOR IS TRYING TO GO ABOVE 3.3V')
end
if teensyOutput < 0
    teensyOutput = 0;
    %     warning('CURSOR IS TRYING TO GO BELOW 0V')
end
% 4
% teensyOutput

end