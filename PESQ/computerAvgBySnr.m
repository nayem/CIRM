function avg_pesq = computerAvgBySnr(pesq_scores,snr_list,num_target,num_noise,datatype)
%
% Description: This function computes the average pesq score by snr value
%
% Inputs:
%   - pesq_scores: Scores to be averaged. Assumes data is in the form
%                  (num_target_signals x num_noise_signals x num_snrs)
%   - snr_list: List of snrs used
%
%   - num_target: the number of clean speech signals used
%
%   - num_noise: the number of noise signals used
%
%   - datatype: specifies when format of pesq_scores is different (0 -
%   default) (1 - num_target_signals x num_noise_signals (i.e. only
%   computed at a single snr value))
%
% Outputs:
%   avg_pesq: the average pesq score by snr num
%
% Written by Donald S. Williamson 10/17/2012
%
if nargin < 5
    datatype = 0;
end

switch datatype
    case 0 % Compute at each snr
        % Initialize the data
        avg_pesq = zeros(length(snr_list),1);
        
        for snrNum = 1:length(snr_list)
            
            scores_resh = reshape(pesq_scores(:,:,snrNum),num_target*num_noise,1);
            
            avg_pesq(snrNum) = mean(scores_resh);
        end
        
    case 1 % Only compute at single snr (typically 0 dB)
        
        scores_resh = reshape(pesq_scores,num_target*num_noise,1);
        avg_pesq = mean(scores_resh);
        
    case 2 % Compute using different parameter values at different snrs
        % (num_target_signals x num_noise_signals x num_snrs x num_params)
        
        num_params = size(pesq_scores,4);
        avg_pesq   = zeros(length(snr_list),num_params);
        
        for iter = 1:num_params
            for snrNum = 1:length(snr_list)
                
                scores_resh = reshape(pesq_scores(:,:,snrNum,iter),num_target*num_noise,1);
                
                avg_pesq(snrNum,iter) = mean(scores_resh);
            end
        end
end