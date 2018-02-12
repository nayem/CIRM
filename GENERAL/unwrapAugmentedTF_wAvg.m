function [ unwrap_avgmag,unwrap_mag ] = unwrapAugmentedTF_wAvg(impt_mag,num_per_side,unwrap_mag)
% Description: Unwrap the augmented time-frequency (T-F) representation and compute the average.
%
% Input:
%   impt_mag: wrapped T-F representation with dimensions (2*T+1)*d x m
%   T: number of frames to left and right of each frame used to augmented
%      T-F representation
%
% Output:
%   unwrap_avgmag: unwrapped and averaged T-F representation with dimensions d x m
%   unwrap_mag: unwrapped T-F representation with dimensions d x (2*T+1) x m
%
% Author: Donald S. Williamson, 4/17/2014

sliding_window_len       = 2*num_per_side + 1;
[numWrapFreqs,numFrames] = size(impt_mag);
numFreqs                 = numWrapFreqs/sliding_window_len;
unwrap_avgmag            = zeros(numFreqs,numFrames);

if num_per_side > 0
    
    if nargin < 3
        unwrap_mag        = zeros(numFreqs,sliding_window_len+1,numFrames);
        curr_ind_location = ones(numFrames,1);
        
        for frameNum = 1:numFrames
            
            % Get the indices for the frames used in this augmented matrix
            frame_inds                         = frameNum-num_per_side:frameNum+num_per_side;
            frame_inds(frame_inds < 1)         = 1;
            frame_inds(frame_inds > numFrames) = numFrames;
            
            % Unwrap the data for this frame
            slid_win_data = reshape(impt_mag(:,frameNum),numFreqs,sliding_window_len); % Size d x (2*T + 1)
            
            for ind_num = 1:length(frame_inds)
                
                unwrap_mag(:,curr_ind_location(frame_inds(ind_num)), frame_inds(ind_num)) = slid_win_data(:,ind_num);
                
                
                % Update counters
                curr_ind_location(frame_inds(ind_num)) = curr_ind_location(frame_inds(ind_num)) + 1;
            end
            
        end
    end
    
    temp = zeros(numFreqs,sliding_window_len+1);
    for frameNum = 1:numFrames
        
        %     temp(:,:)                 = unwrap_mag(:,:,frameNum);
        unwrap_avgmag(:,frameNum) = mean(unwrap_mag(:,:,frameNum),2);
        
    end
else
    unwrap_avgmag = impt_mag;
    unwrap_mag = 0;
end
end

