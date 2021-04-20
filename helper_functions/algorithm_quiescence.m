function quiescenceState = algorithm_quiescence(dFoF, baselineCursorThreshold)
% baselineCursorThreshold
% dFoF
if dFoF < baselineCursorThreshold
%     1
    quiescenceState = 1;
else quiescenceState = 0;
%     1
end

end