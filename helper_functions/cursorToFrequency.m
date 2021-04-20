function outputFrequency = cursorToFrequency(cursor, range_cursor, range_frequency)
%%
% range_cursor = [-1 1];
% range_frequency = [1000 20000];

principleNote = 440; % in hz
% octave_basis = 1:0.25:6;
octave_basis = 1:0.01:6;
freq_basis = principleNote*(2.^(octave_basis));
freq_basis(freq_basis < range_frequency(1)) = []; freq_basis(freq_basis > range_frequency(2)) = [];

num_freq = numel(freq_basis);

cursor_scaled = (cursor - range_cursor(1)) / (range_cursor(2) - range_cursor(1));  % 0 to 1
cursor_scaled = max(min(cursor_scaled,1) , 1/(num_freq+1));

idx_freqToUse = ceil(num_freq*cursor_scaled);

outputFrequency = freq_basis(idx_freqToUse);

end