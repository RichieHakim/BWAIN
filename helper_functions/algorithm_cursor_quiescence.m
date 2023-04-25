function quiescenceState = algorithm_cursor_quiescence(cursor_positions, decoder_magnitudes, factor_to_use, thresh_quiescence_cursorDecoder, thresh_quiescence_cursorMag)
% baselineCursorThreshold

if (cursor_positions(factor_to_use) <= thresh_quiescence_cursorDecoder) && (decoder_magnitudes(factor_to_use) <= thresh_quiescence_cursorMag)
    quiescenceState = 1;
else
    quiescenceState = 0;
end

end