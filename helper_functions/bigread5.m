function data = bigread5(path_to_file)
%reads tiff files in Matlab bigger than 4GB, allows reading from sframe to sframe+num2read-1 frames of the tiff - in other words, you can read page 200-300 without rading in from page 1.  
%based on a partial solution posted on Matlab Central (http://www.mathworks.com/matlabcentral/answers/108021-matlab-only-opens-first-frame-of-multi-page-tiff-stack)
%Darcy Peterka 2014, v1.0
%Darcy Peterka 2014, v1.1 (bugs to dp2403@columbia.edu)
%Simplified by Rich Hakim 2016 to work with files not from scanimage
%Hacked by Andrew Landau 2019 to find image info faster
%Updated again by Rich Hakim 2020 for it to work in Matlab 2019b

%Program checks for bit depth, whether int or float, and byte order.  Assumes uncompressed, non-negative (i.e. unsigned) data.

% Usage:  my_data=bigread('path_to_data_file, start frame, num to read);
% "my_data" will be your [M,N,frames] array.
%Will do mild error checking on the inputs - last two inputs are optional -
%if nargin == 2, then assumes second is number of frames to read, and
%starts at frame 1

%get image info
fid = fopen(path_to_file, 'r');
tifname = fopen(fid);
fclose(fid);
% info = matlab.io.internal.imagesci.tifftagsread(tifname,0,0,0);
info = imfinfo(path_to_file);
numFrames = size(info,1);

bd=info(1).BitDepth;
he=info(1).ByteOrder;
bo=strcmp(he,'big-endian');
if (bd==64)
	form='double';
elseif(bd==32)
    form='single';
elseif (bd==16)
    form='uint16';
elseif (bd==8)
    form='uint8';
end

% Use low-level File I/O to read the file
fp = fopen(path_to_file , 'rb');
% The StripOffsets field provides the offset to the first strip. Based on
% the INFO for this file, each image consists of 1 strip.

ofds=zeros(1,numFrames);
for i=1:numFrames
    ofds(i)=info(i).StripOffsets(1);
end

fseek(fp, ofds(1), 'bof'); %go to start of first strip

cols = info(1).Width;
rows = info(1).Height;
data = zeros(rows,cols,numFrames,form);

if strcmpi(form,'uint16') || strcmpi(form,'uint8')
    if(bo)
        for cnt = 1:numFrames
            fseek(fp,ofds(cnt),'bof');
            data(:,:,cnt) = fread(fp, [cols rows], form, 0, 'ieee-be')';
        end
    else
        for cnt = 1:numFrames
            fseek(fp,ofds(cnt),'bof');
            data(:,:,cnt) = fread(fp, [cols rows], 'int16', 0, 'ieee-le')';
        end
    end
elseif strcmpi(form,'single')
    for cnt = 1:numFrames
        fseek(fp,ofds(cnt),'bof');
        data(:,:,cnt) = fread(fp, [cols rows], form, 0, 'ieee-be');
    end
elseif strcmpi(form,'double')
    for cnt = 1:numFrames
        fseek(fp,ofds(cnt),'bof');
        data(:,:,cnt) = fread(fp, [cols rows], form, 0, 'ieee-le.l64');
    end
end

fclose(fp);

