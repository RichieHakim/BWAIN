function quiescenceState = algorithm_baseline_exp_quiescence(D, baselineCursorThreshold)
% baselineCursorThreshold
% dFoF
if D <= baselineCursorThreshold
%     1
    quiescenceState = 1;
else
    quiescenceState = 0;
%     1
end

end