function [fft_bm,fft_wm] = mel_to_fft_naive(est_tar,est_ns,lc,fftlen,winsize,hopsize,Fs)

fft1 = spectrogram(est_tar,winsize,hopsize,fftlen,Fs);
fft2 = spectrogram(est_ns,winsize,hopsize,fftlen,Fs);

e1 = (abs(fft1)).^2;
e2 = (abs(fft2)).^2;

fft_bm = double(10*log10(e1./e2)>lc);
fft_wm = (e1./(e1+e2));