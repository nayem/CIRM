% function [signal] = resyn_from_fft(phase, mag_dB, sigLen)
function [signal] = resyn_from_fft(phase, mag, sigLen,fftlen,hopsize,Fs)
%Function to resynthsize signal from phase and FFT magnitude estimated in
%dB. The signal length is the last parameter

fs = Fs;
if nargin<=3
    fftlength = fs*20/1000;
else    
    fftlength = fftlen;
end

% mag = 10 .^ (mag_dB / 20) - 1;

% j = sqrt(-1);
sig_complex = mag .* exp(phase*1i);

fullFFT = [sig_complex; flipud(conj(sig_complex(2 : fftlength/2, :)))];

signal = zeros(1, sigLen+3000);

% overlap-and-add
for i = 1 : size(fullFFT, 2)
    curFrm = fullFFT(:, i);
    tS = ifft(curFrm, fftlength);
%     tS = ifft(curFrm, 512);
    
    start = (i - 1) * hopsize + 1;
%     start = (i - 1) * 256 + 1;
    finish = start + fftlength - 1;
    
    signal(start : finish) = signal(start : finish) + tS';
end
signal = real(signal(1 : sigLen))';

end
