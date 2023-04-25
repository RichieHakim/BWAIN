function [output] = json_load(path)
    fid = fopen(path); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    output = jsondecode(str);
end