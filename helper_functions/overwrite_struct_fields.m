%% RH 2023
%%% Written to be used to over write fields, hierarchically, from one
%%% struct into another struct.
function struct_overwritten = overwrite_struct_fields(struct_overwritten, struct_overwriter)
    fn_ers = fieldnames(struct_overwriter);
    for idx = 1:length(fn_ers)
        fn_er = string(fn_ers(idx));
        field_er = struct_overwriter.(fn_er);
        if isfield(struct_overwritten, fn_er)
            field_en = struct_overwritten.(fn_er);
            if isstruct(field_er)
                struct_overwritten.(fn_er) = overwrite_struct_fields(field_en, field_er);
            else
                struct_overwritten.(fn_er) = field_er;
            end
        else
            struct_overwritten.(fn_er) = field_er;
        end
    end
end