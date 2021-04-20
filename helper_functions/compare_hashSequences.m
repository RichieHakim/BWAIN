function [  ind_align_1to2 , ind_align_2to1,...
            vals_uniqueToSet1 , inds_uniqueToSet1 ,...
            vals_uniqueToSet2 , inds_uniqueToSet2 ,...
            inds_duplicateSet1Values, inds_duplicateSet2Values  ]...
            = compare_hashSequences(hash1 , hash2)
% hash1 = output;
% inds_to_nix_1 = randi(length(output),5,1);
% hash1(inds_to_nix_1) = [];
% % output1(10001) = output1(30000);
% hash2 = output;
% inds_to_nix_2 = randi(length(output),10,1);
% hash2(inds_to_nix_2) = [];
% % output2(20001) = output2(40000);
%%
[vals_uniqueToSet1 , ~,inds_uniqueToSet1] = intersect(setxor(hash1,hash2) , hash1);
inds_uniqueToSet1 = sort(inds_uniqueToSet1);
[vals_uniqueToSet2 , ~,inds_uniqueToSet2] = intersect(setxor(hash1,hash2) , hash2);
inds_uniqueToSet2 = sort(inds_uniqueToSet2);
% [shared_vals , ia , ib] = intersect(output1 , output2, 'stable');
%%
hash2_temp = hash2;
ind_align_1to2 = zeros(length(hash1),1);
for ii = 1:length(hash1)
    ind = find(hash1(ii) == hash2_temp);
    if isempty(ind)
        ind_align_1to2(ii) = nan;
    elseif length(ind) == 2
        ind = ind(1);
        ind_align_1to2(ii) = ind;
    else
        ind_align_1to2(ii) = ind;
    end
    hash2_temp(ind) = nan;
end

% this could probably be done better with indexWithNaNs, but it's easier to
% understand with a for loop
ind_align_2to1 = nan(length(hash2),1);
for ii = 1:length(ind_align_1to2)
    if ~isnan(ind_align_1to2(ii))
        ind_align_2to1(ind_align_1to2(ii)) = ii;
    end
end

figure; plot(hash2)
hold on; plot(ind_align_1to2 , hash1)

fig_idx = figure; 
ax1 = subplot(3,1,1);
plot(diff(ind_align_1to2))
xlabel('x-index intervals of Sequence 1')
ylabel('diff(alignment sequence)')
ax2 = subplot(3,1,2);
plot(ind_align_1to2)
xlabel('indices of Sequence 1')
ylabel('corresponding indices of Sequence 2')
ax3 = subplot(3,1,3);
plot(ind_align_2to1)
xlabel('indices of Sequence 2')
ylabel('corresponding indices of Sequence 1')
linkaxes([ax1,ax2,ax3],'x')

if any(diff(ind_align_1to2)<=0)
    warning('RH WARNING: Sequence alignment is not monotonically increasing. Consider repeating with offending indices having their hash values set to NaN')
end

if length(unique(hash1)) < length(hash1)
    warning('RH WARNING: Sequence 1 is not unique. Check manually')
    [C, ia, ib] = unique(hash1,'first');
    inds_duplicateSet1Values = find(not(ismember(1:numel(C),ia)))
else
    inds_duplicateSet1Values = [];
end
if length(unique(hash2)) < length(hash2)
    warning('RH WARNING: Sequence 2 is not unique. Check manually')
    [C, ia, ib] = unique(hash2,'first');
    inds_duplicateSet2Values = find(not(ismember(1:numel(C),ia)))
else
    inds_duplicateSet2Values = [];
end

if length(inds_uniqueToSet1)>0
    disp(['There were ' num2str(length(inds_uniqueToSet1))...
        ' unique values in sequence 1'])
end
if length(inds_uniqueToSet2)>0
    disp(['There were ' num2str(length(inds_uniqueToSet2))...
        ' unique values in sequence 2'])
end



end