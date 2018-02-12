function AMS_points = get_amsfeature_realimag(sig,nChan,fs,winlen,overlaplen,domain)

if nargin < 3
    fs = 16e3;
end

if nargin < 4
    winlen = 320;
end
if nargin < 5
    overlaplen = 160;
end

% nChan = 64; 
% nFrame = floor(length(sub_gf)/hop_size)-1;
nFrame = single(fix((length(sig)-overlaplen)/(winlen-overlaplen))); % Number of time frames
ns_ams = extract_AMS_realimag(sig,nChan,fs,nFrame,winlen,overlaplen,domain);

AMS_points = ns_ams';

AMS_points = AMS_points(1:nFrame,:);
