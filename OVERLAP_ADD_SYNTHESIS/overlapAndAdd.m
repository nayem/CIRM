function [signal,stft] = overlapAndAdd(stft,sigLen,fftlength,hopsize,params)
%
% Description: Function to resynthsize signal from complex STFT estimated
%
% Inputs
%   - stft: short-time Fourier transform
%
%   - sigLen: Length of the time domain signal
%
%   - fftlength: Size of the FFT
%
%   - hopsize: Hop size used in STFT calculation
%
% Outputs
%   - signal: time domain signal

fullFFT = double([stft; flipud(conj(stft(2:fftlength/2, :)))]);

if params.useGPU
    fullFFT = gpuArray(fullFFT);
    signal = zeros(sigLen*2,1,'gpuArray');
else
    signal = zeros(sigLen*2,1);
end

if(isfield(params,'window'))
    window = params.window;
else
    window = hann(params.winlen);
end

if(isfield(params,'scalebywindow'))
    scalebywindow = params.scalebywindow;
else
    scalebywindow = 0;
end

%% overlap-and-add
for i = 1:size(fullFFT, 2)
    
    tS     = real(ifft(fullFFT(:, i), fftlength));
    
    start  = (i - 1)*hopsize + 1;
    finish = start + fftlength - 1;
    
    if scalebywindow == 1
        signal(start:finish) = signal(start:finish) + tS(:).*window(:);
    else
        signal(start:finish) = signal(start:finish) + tS(:);
    end
end
signal = signal(1:sigLen)*(hopsize/sum(window.^2));

if params.useGPU
    signal = double(gather(signal));
end
stft   = spectrogram(signal,window,length(window)-hopsize,fftlength,16e3);
end
