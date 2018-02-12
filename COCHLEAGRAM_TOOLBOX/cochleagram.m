function a = cochleagram(r, winLength,overlaplen)
% Generate a cochleagram from responses of a Gammatone filterbank.
% It gives the log energy of T-F units
% The first variable is required.
% winLength: window (frame) length in samples
% Written by ZZ Jin, and adapted by DLW in Jan'07

if nargin < 2
    winLength = 320;      % default window length in sample points which is 20 ms for 16 KHz sampling frequency
end
if nargin < 3
    overlaplen = 160;
end

[numChan,sigLength] = size(r);     % number of channels and input signal length

winShift = winLength - overlaplen; %winLength/2;            % frame shift (default is half frame)
increment = winLength/winShift;    % special treatment for first increment-1 frames
M = single(fix((sigLength-overlaplen)/(winLength-overlaplen))); % Number of time frames

% calculate energy for each frame in each channel
a = zeros(numChan,M);
for m = 1:M      
    for i = 1:numChan
        if m < increment        % shorter frame lengths for beginning frames
            a(i,m) = r(i,1:m*winShift)*r(i,1:m*winShift)';
        else
            startpoint = floor((m-increment)*winShift);
            a(i,m) = r(i,startpoint+1:startpoint+winLength)*r(i,startpoint+1:startpoint+winLength)';
        end
    end
end
