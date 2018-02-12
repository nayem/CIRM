function masker = scaleMasker(target,masker,snr)
%Description This function scales the masker, so that the resulting target
% plus masker, mixture has the given snr
% 
% Inputs:
%   - target: the target signal
%   - masker: the noise (or masker) signal
%   - snr: signal-to-noise ratio
%
% Outputs:
%   -masker: the scaled masker
%


change = 20*log10(std(target)/std(masker)) - snr;
masker = masker*10^(change/20);     % scale masker to specified input SNR
    

end

