function [avg_pesq,scores_resh] = computeAvgPesqBySnr(pesq_scores,snr_list,num_target,num_noise,datatype)
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

% Written by Donald S. Williamson 10/17/2012
%
if nargin < 5
    datatype = 0;
end
%pesq_scores(isnan(pesq_scores) == 1) = 1.6;

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
        
    case 3 % Compute using different parameter values at different snrs
        % (num_target_signals x num_noise_signals x num_snrs x num_params1 x num_params2)
        num_params1 = size(pesq_scores,4);
        num_params2 = size(pesq_scores,5);
        avg_pesq = zeros(length(snr_list),num_params1,num_params2);
        temp = [];
        for iter1 = 1:num_params1
            for iter2 = 1:num_params2
                for snrNum = 1:length(snr_list)
                    scores_resh = reshape(pesq_scores(:,:,snrNum,iter1,iter2),num_target*num_noise,1);
                    avg_pesq(snrNum,iter1,iter2) = mean(scores_resh);
                end
            end
        end
        
    case 4 % Compute average PESQ at each noise type and snr
        
        avg_pesq = zeros(length(snr_list),num_noise);
        scores_resh = pesq_scores;
        for snrNum = 1:length(snr_list)
            
            for noiseNum = 1:num_noise
                avg_pesq(snrNum,noiseNum) = mean(pesq_scores(:,noiseNum,snrNum));
                
            end
        end    
        
    case 5 % Compute average score at each snr by with different data dimensions
        % numLists x numUttsPerList x NumNoises x SNR
        % num_target = numLists*numUttsPerList
        
        avg_pesq = zeros(length(snr_list),1);
        scores_resh = pesq_scores;
        for snrNum = 1:length(snr_list)
            
            %temp1 = pesq_scores(:,:,:,snrNum);
            %inds_isnan = isnan(temp1) == 0;
            %scores_resh = temp1(inds_isnan);

            scores_resh = reshape(pesq_scores(:,:,:,snrNum),num_target*num_noise,1);            
            avg_pesq(snrNum) = mean(scores_resh);
            
        end
    case 6 % Compute average PESQ at each noise type and snr with different data dimensions
        % numLists x numUttsPerList x NumNoises x SNR
        % num_target = numLists*numUttsPerList
        avg_pesq = zeros(length(snr_list),num_noise);
        scores_resh = pesq_scores;
        for snrNum = 1:length(snr_list)
            
            for noiseNum = 1:num_noise
                %temp2 = pesq_scores(:,:,noiseNum,snrNum);
                %inds_isnan = isnan(temp2) == 0;
                %scores_resh = temp2(inds_isnan);
                
                scores_resh = reshape(pesq_scores(:,:,noiseNum,snrNum),num_target,1);
                avg_pesq(snrNum,noiseNum) = mean(scores_resh);
                
            end
        end  
        
end