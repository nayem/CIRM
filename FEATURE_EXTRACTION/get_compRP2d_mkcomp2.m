function feat_stack = get_compRP2d_mkcomp2(sig,CONSTANTS)

%addpath('~/PACKAGES/FEATURE_EXTRACTION/')
% path_prefix_corpus = '~/research/data/hint/';

%feat_stack = [];
%fprintf('AMS...')
ams = get_amsfeature_chan_fast2(sig,CONSTANTS.numGammatoneChans,CONSTANTS.Fs,CONSTANTS.winlen,CONSTANTS.overlap);

% feat_stack = [feat_stack ams];


%fprintf('RASTAPLP2D...')
[ras_cep] = rastaplp(sig, CONSTANTS.Fs, 1, 12,CONSTANTS.winlen/CONSTANTS.Fs,CONSTANTS.hopsize/CONSTANTS.Fs);   
% feat_stack = [feat_stack ras_cep'];

% % ras_del = deltas(ras_cep);
% % ras_ddel = deltas(ras_del);
% % feat_stack = [feat_stack ras_cep' ras_del' ras_ddel'];

%fprintf('MODMFCC...')
% [cep] = melfcc(sig, 16000,'numcep',31,'nbands',64);
[cep] = melfcc(sig, CONSTANTS.Fs,'numcep',31,'nbands',CONSTANTS.numGammatoneChans,'wintime',CONSTANTS.winlen/CONSTANTS.Fs,'hoptime',CONSTANTS.hopsize/CONSTANTS.Fs);
% feat_stack = [feat_stack cep']; 

gt = gammatone(sig,CONSTANTS.numGammatoneChans,CONSTANTS.fRange,CONSTANTS.Fs);
ct = cochleagram(gt,CONSTANTS.winlen,CONSTANTS.overlap); 
ct = ct.^(1/15);

%fprintf('DELTAS...\n')
feat_stack = [ams,ras_cep.',cep.',ct.'];
%del = deltas(feat_stack.');
%ddel = deltas(del);
% feat_stack = [feat_stack del' ddel'];

%feat_stack = [ams, ras_cep',cep',del',ddel'];

del = deltas(feat_stack.');
feat_stack = [feat_stack del'];
feat_stack(isnan(feat_stack)) = 0;