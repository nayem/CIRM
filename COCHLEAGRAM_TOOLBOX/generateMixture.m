function [mixture] = generateMixture(target,masker,snr)

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
if (lt >= lm)       % equalize the lengths of the two files
    target = target(1:lm);
else
    masker = masker(1:lt);
end

% Scale the masker so the mixture has the appropriate SNR
change = 20*log10(std(target)/std(masker)) - snr;
masker = masker*10^(change/20);     % scale masker to specified input SNR

mixture = target + masker;

