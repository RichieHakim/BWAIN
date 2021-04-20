function array1_indexed1To2 = indexWithNaNs(array1 , ind_align_2to1)
% Rich Hakim 2021
% inputs:
% array1 =  An array. dim-1 = sequence to align, dim-2 = alignment performed similarly across this axis.
% ind_align_2to1 =  A sequence of indices. Each entry is a an index in the
%                   new space (2), and each value is the index corresponding to
%                   the original space (1). (ie. x-index is space 2 indices, y-values
%                   are space 1 indices).

% The function looks through each index in space 2 that doesn't contain a NaN, finds the
% corresponding index in space 1, and then populates the space 2 index with
% the appropriate value from space 1.

array1_indexed1To2 = nan(length(ind_align_2to1) , size(array1,2));
array1_indexed1To2(~isnan(ind_align_2to1),:) = array1(ind_align_2to1(~isnan(ind_align_2to1)),:);

end