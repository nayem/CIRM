function snr = computeSNR(clean_speech,proc_speech)
%
% Description: This function computes the signal-to-noise ratio between a
% clean speech signal and a processed signal
%
% Inputs:
%   - clean_speech: clean speech signal
%   - proc_speech: processed speech signal
%
% Outputs:
%   - snr: signal-to-noise ratio
%
% Written by Donald S. Williamson, 6/19/2013
%

% Convert to column vector if necessary
if(~iscolumn(clean_speech))
    clean_speech = clean_speech';
end

if(~iscolumn(proc_speech))
    proc_speech = proc_speech';
end

% Compute the energy of the clean speech signal
speech_energy = clean_speech'*clean_speech;

% Compute the energy of the noise signal
noise = clean_speech-proc_speech;
noise_energy = noise'*noise;

% Compute signal-to-noise ratio
snr = 10*log10(speech_energy/noise_energy);