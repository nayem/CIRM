function AMS_points = get_amsfeature_chan_fast(sub_gf,nChan,fs,hop_size)

if nargin < 3
    fs = 16e3;
end

if nargin < 4
    hop_size = 160;
end

% nChan = 64; 
nFrame = floor(length(sub_gf)/hop_size)-1;
ns_ams = extract_AMS_perChan(sub_gf,nChan,fs);

AMS_points = ns_ams';

AMS_points = AMS_points(1:nFrame,:);
