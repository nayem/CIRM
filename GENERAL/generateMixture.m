function [mixture,target,masker] = generateMixture(target,masker,snr)

%
% Description: This function generates a mixture (target + noise) at a
% particular snr
%
% Inputs:
%   - target: clean speech signal
%
%   - masker: noise signal
%
%   - snr: signal-to-noise ratio
%
% Outputs:
%   - mixture: noisy speech at desired snr level
%
% Written by Donald S. Williamson, 10/31/2012
%

% Make the target and masker files the same length
lt = length(target); lm = length(masker);


%% Random Cut
% Nayem edit, Sep 14, 2017

% For Train, Dev -> isFromFirst_2min= true
% For Test -> isFromFirst_2min= false

isFromFirst_2min = false;

if (isFromFirst_2min==false)
    head = floor(lm/2)+1;
    tail = lm;
    len = tail-head+1;

    possible_cut_numbers = floor(len/lt);
    selected_cut_number = randi(possible_cut_numbers);

    s = head+(selected_cut_number-1)*lt+1;
    t = head+selected_cut_number*lt;
    fprintf('first:%d target-Len:%d, mask-Len:%d, s:%d, t:%d, len:%d\n', isFromFirst_2min, lt,lm,s,t,t-s+1);
    
elseif (isFromFirst_2min==true)
    head = 1;
    tail = floor(lm/2);
    len = tail-head+1;

    possible_cut_numbers = floor(len/lt);
    selected_cut_number = randi(possible_cut_numbers);

    s = head+(selected_cut_number-1)*lt;
    t = head+selected_cut_number*lt-1;
    fprintf('first:%d target-Len:%d, mask-Len:%d, s:%d, t:%d, len:%d\n', isFromFirst_2min,lt,lm,s,t,t-s+1);
end
%%

if (lt >= lm)       % equalize the lengths of the two files
    target = target(1:lm);
else
    masker = masker(s:t);
    % Nayem edit, Sep 14, 2017
%     masker = masker(1:lt);
end

% Scale the masker so the mixture has the appropriate SNR
change = 20*log10(std(target)/std(masker)) - snr;
masker = masker*10^(change/20);     % scale masker to specified input SNR

mixture = target + masker;

