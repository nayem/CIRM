function [fft_bm, fft_wm] = gt_to_fft_naive(mixsig, gt_mask,CONSTANTS)

lc      = CONSTANTS.lc;
fftlen  = CONSTANTS.fft_len;
winsize = CONSTANTS.win_len;
hopsize = CONSTANTS.hop_size;
Fs      = CONSTANTS.Fs;
fRange = CONSTANTS.fRange;

gt_mask(gt_mask==-1)=0;

est_tar = synthesis(mixsig,gt_mask,fRange,winsize,Fs);
est_ns = synthesis(mixsig,1-gt_mask,fRange,winsize,Fs);

fft1 = spectrogram(est_tar,winsize,winsize - hopsize,fftlen,Fs);
fft2 = spectrogram(est_ns,winsize,winsize - hopsize,fftlen,Fs);

% fft1 = spectrogram(est_tar,512,352,fftlen,16e3);
% fft2 = spectrogram(est_ns,512,352,fftlen,16e3);

e1 = (abs(fft1)).^2;
e2 = (abs(fft2)).^2;


fft_bm = double(10*log10(e1./e2)>lc);
fft_wm = (e1./(e1+e2));
