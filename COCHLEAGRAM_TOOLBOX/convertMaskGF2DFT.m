function [ibm_d,nPow,cPow]=convertMaskGF2DFT(sig,ibm,mode,CONSTANTS,mask_type)
%path(path,'/home1/narayaar/Toolboxes/CASA/');
%path(path,'/home1/narayaar/Toolboxes/rastamat/');

if nargin < 3
    mode = 0;
end

%% Parameters
if nargin < 4
    fs = 16000;
    wavnorm = 3.276797651191444e+004;
    numChan = 512;
    fRange = [50 8000];
    winLength = 20*fs/1000;
    winLength_stft = winLength;
    overlapLength = 10*fs/1000;
    wintime=0.020;
    hoptime=0.010;
    dither=1;
    nbands=26;
    bwidth=1;
    fbtype='htkmel';
    sumpower=0;
    winpts=round(fs*wintime);
    mask_type = 0; % binary mask
else
    fs = CONSTANTS.Fs;
    wavnorm = CONSTANTS.wavnorm;
    numChan = CONSTANTS.fftLen;
    fRange = CONSTANTS.fRange;
    winLength = CONSTANTS.winLen_coch;
    winLength_stft = CONSTANTS.winLen_stft;
    overlapLength = CONSTANTS.overLen;
    wintime=0.020;
    hoptime=0.010;
    dither=1;
    nbands=26;
    bwidth=1;
    fbtype='htkmel';
    sumpower=0;
    winpts=round(fs*wintime);
    
end

if nargin < 5
    mask_type = 0; % binary mask
end

%% Convert masks
if ischar(ibm)
    ibm = dlmread(ibm);
end

clean = synthesisNorm(sig,double(ibm),fRange,winLength,fs);
noise = synthesisNorm(sig,1-double(ibm),fRange,winLength,fs);

% cPow = abs(specgram(clean,numChan,fs,hamming(winLength),winLength/2)).^2;
% nPow = abs(specgram(noise,numChan,fs,hamming(winLength),winLength/2)).^2;

cPow = abs(spectrogram(clean,winLength_stft,overlapLength,numChan,fs)).^2;
nPow = abs(spectrogram(noise,winLength_stft,overlapLength,numChan,fs)).^2;

if ~mode
    if(mask_type == 0)
        ibm_d = double(cPow >= nPow);
    else
       ibm_d =  double( (cPow./(cPow + nPow)).^0.5);
    end
else
    cASpec = audspec(cPow+winpts,fs,nbands,fbtype,fRange(1),fRange(2),sumpower,bwidth);
    nASpec = audspec(nPow+winpts,fs,nbands,fbtype,fRange(1),fRange(2),sumpower,bwidth);
    ibm_d = double(cASpec >= nASpec);
end

clear clean noise sig ibm cASpec nASpec;

end