function [stft_r,stft_i,stft] = comp_spectrogram_realimag(sig, NFFT, win, shift, fs)
%
% allfea = comp_spectrogram(sig, feawin, NFFT)
% sig:    input time-domain signal
% ranges: range cell of the input time-domain signal, if only one signal,
%         ranges{1} = 1:length(lsig)
% feawin: windows features feawin = [# left frames, # of right frames].
% Suggestion: If extract features: feawin=5; if extract target: feawin = 0
%

%addpath('~/PROJECTS/DEREVERB/DNN_SpecMapping/utility/');

if nargin < 4
    win   = 320;
    shift = 160;
    fs    = 16000;
end

% Compute the stft
stft     = single(spectrogram(sig, win, shift, NFFT, fs));
stft_r   = real(stft);
stft_i   = imag(stft);
%realimag = [data_r;imag(stft)];

end


