function feats = get_compFeats_realimag(sig,CONSTANTS,real_domain,imag_domain)

addpath('~/PACKAGES/FEATURE_EXTRACTION/')

%% Compute AMS features from the real and imaginary spectrums
ams_real = get_amsfeature_realimag(sig,CONSTANTS.numGammatoneChans,CONSTANTS.Fs,CONSTANTS.win_len,CONSTANTS.overlap_len,real_domain);
ams_imag = get_amsfeature_realimag(sig,CONSTANTS.numGammatoneChans,CONSTANTS.Fs,CONSTANTS.win_len,CONSTANTS.overlap_len,imag_domain);

%% Compute RASTA-PLP features from the real and imaginary spectrums
ras_cep_real = rastaplp_realimag(sig, CONSTANTS.Fs, 1, 12,CONSTANTS.win_len/CONSTANTS.Fs,CONSTANTS.hop_size/CONSTANTS.Fs,'real_mag');   
ras_cep_imag = rastaplp_realimag(sig, CONSTANTS.Fs, 1, 12,CONSTANTS.win_len/CONSTANTS.Fs,CONSTANTS.hop_size/CONSTANTS.Fs,'imag_mag');   

%% Compute MFCC features from the real and imaginary spectrums
mfcc_real = melfcc_realimag(sig, CONSTANTS.Fs,real_domain,'numcep',31,'nbands',CONSTANTS.numGammatoneChans,'wintime',CONSTANTS.win_len/CONSTANTS.Fs,'hoptime',CONSTANTS.hop_size/CONSTANTS.Fs);
mfcc_imag = melfcc_realimag(sig, CONSTANTS.Fs,imag_domain,'numcep',31,'nbands',CONSTANTS.numGammatoneChans,'wintime',CONSTANTS.win_len/CONSTANTS.Fs,'hoptime',CONSTANTS.hop_size/CONSTANTS.Fs);

%%
% gt = gammatone(sig,CONSTANTS.numGammatoneChans,CONSTANTS.fRange,CONSTANTS.Fs);
% ct = cochleagram(gt,CONSTANTS.win_len,CONSTANTS.overlap_len); 
% ct = ct.^(1/15);
[feats_r,feats_i] = comp_spectrogram_realimag(sig,CONSTANTS.realimag_stft_nfft,CONSTANTS.winlen,CONSTANTS.overlap,CONSTANTS.Fs);  % Extract spectrogram for reverberant signals
melbank = single(full(melbankm(CONSTANTS.numMelChans,CONSTANTS.realimag_stft_nfft,CONSTANTS.Fs)));
feats_r = abs(melbank*feats_r);
feats_i = abs(melbank*feats_i);

%% Stack the features together
%feat_stack = [ams_real,ams_imag,ras_cep_real.',ras_cep_imag.',mfcc_real.',mfcc_imag.',ct.'];
feat_stack = [ams_real,ams_imag,ras_cep_real.',ras_cep_imag.',mfcc_real.',mfcc_imag.',feats_r.',feats_i.'];


%% Compute deltas and stack onto features
del   = deltas(feat_stack.');
feats = [feat_stack del'];