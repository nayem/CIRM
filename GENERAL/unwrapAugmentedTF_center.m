function [ unwrap_mag ] = unwrapAugmentedTF_center(impt_mag,num_per_side)
% Description: Unwrap the augmented time-frequency (T-F) representation.
% Only retains center frame for each augmented vector
%
% Input:
%   impt_mag: wrapped T-F representation with dimensions (2*T+1)*d x m
%   T: number of frames to left and right of each frame used to augmented
%      T-F representation
%
% Output:
%   unwrap_mag: unwrapped T-F representation with dimensions d x (2*T+1) x
%               m

sliding_window_len = 2*num_per_side + 1;
[numWrapFreqs]     = size(impt_mag,1);
numFreqs           = numWrapFreqs/sliding_window_len;

start_cent_freq = num_per_side*numFreqs + 1;
stop_cent_freq  = start_cent_freq + numFreqs - 1;

unwrap_mag = impt_mag(start_cent_freq:stop_cent_freq,:);

end

