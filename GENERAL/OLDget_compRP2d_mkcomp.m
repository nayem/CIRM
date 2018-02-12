function feat_stack = OLDget_compRP2d_mkcomp(sig,CONSTANTS)
% path_prefix_corpus = '~/research/data/hint/';

feat_stack = [];
%fprintf('AMS...')
ams = get_amsfeature_chan_fast(sig,CONSTANTS.numGammatoneChans,CONSTANTS.Fs,CONSTANTS.hop_size);
feat_stack = [feat_stack ams];


%fprintf('RASTAPLP2D...')
[ras_cep] = rastaplp(sig, CONSTANTS.Fs, 1, 12);   
feat_stack = [feat_stack ras_cep'];
% ras_del = deltas(ras_cep);
% ras_ddel = deltas(ras_del);
% feat_stack = [feat_stack ras_cep' ras_del' ras_ddel'];


%fprintf('MODMFCC...')
% [cep] = melfcc(sig, 16000,'numcep',31,'nbands',64);
[cep] = melfcc(sig, CONSTANTS.Fs,'numcep',31,'nbands',CONSTANTS.numGammatoneChans);
feat_stack = [feat_stack cep']; 

%fprintf('DELTAS...\n')
del = deltas(feat_stack');
ddel = deltas(del);
feat_stack = [feat_stack del' ddel'];