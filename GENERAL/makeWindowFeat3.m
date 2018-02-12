function ret = makeWindowFeat3(data, num_per_side)
% Description: Augment the data
%
% Input:
%   - data: d x m data matrix that is to be augmented
%   - side: number of frames (left then right) for augmentation
%
% Output
%   - ret: d*(2*side + 1) x m data matrix. If there aren't 'side' frames
%          before or after the current frame, then repeat the first or last
%          frame, respectively
% Author: Donald S. Williamson, 4/16/2014

% Initialize variables
if num_per_side ~= 0
    
    [numFreqs, numFrames] = size(data);
    ret                   = zeros(numFreqs*(2*num_per_side + 1), numFrames,'single');
    
    for i = 1:numFrames
        
        % Get the indices for the frames used in this augmented matrix
        frame_inds                         = i-num_per_side:i+num_per_side;
        frame_inds(frame_inds < 1)         = 1;
        frame_inds(frame_inds > numFrames) = numFrames;
        
        % Get the data for the frames used in this augmented matrix
        temp     = data(:,frame_inds);
        ret(:,i) = reshape(temp,numFreqs*(2*num_per_side + 1),1);
        
    end
else
    ret = data;
end

