clear all;
gpuDevice(1);
addpath(genpath('~/work/ASR/triphone_dnn_organized/'))
%% --setting net params--
% default training options
opts.isNormalize = 1; % need to normalize data?
opts.ARMA_order = 0; % need to do ARMA smoothing and what's the filter order?
opts.cv_interval = 1; % check cv perf. every this many epochs

opts.isPretrain = 0; % pre-training using RBM?
opts.rbm_max_epoch = 0; 
opts.rbm_batch_size = 1024; % batch size for pretraining
opts.rbm_learn_rate_binary = 0.01;
opts.rbm_learn_rate_real = 0.004;

opts.learner = 'ada_sgd'; % 'ada_sgd' or 'sgd'
opts.sgd_max_epoch = 20;
opts.sgd_batch_size = 1024/2; % batch size for SGD
opts.ada_sgd_scale = 0.0015; % scaling factor for ada_grad
opts.sgd_learn_rate = linspace(0.08, 0.001, opts.sgd_max_epoch); % linearly decreasing lrate for plain sgd

opts.initial_momentum = 0.5;
opts.final_momentum = 0.9;
opts.change_momentum_point = 5;

opts.cost_function = 'softmax_xentropy';
opts.hid_struct = [2048 2048 2048 2048 2048]; % num of hid layers and units
% opts.hid_struct = [2048 ]; % num of hid layers and units

opts.unit_type_output = 'softmax';
opts.unit_type_hidden = 'relu';
if strcmp(opts.unit_type_output,'softmax'); opts.cost_function = 'softmax_xentropy'; end;

opts.isDropout = 1; % need dropout regularization?
opts.isDropoutInput = 0; % dropout inputs? do not dropout if raw features are louzy
opts.drop_ratio = 0.2; % ratio of units to drop

opts.isGPU = 1; % use GPU?
opts.eval_on_gpu = 0; 

opts.save_on_fly = 1; % save the current best model along the way
opts.save_model_path = '~/work/ASR/triphone_dnn_organized/models/';
opts.note_str = 'triphone_timit';
%% --load data--
% /// this is reduced training set, having on 3000000 samples
load('~/work/ASR/triphone_dnn_organized/data/train.reduced.mfcc.frm11.mean_var_norm_perUtt.matrix.mat')

test = load('~/work/ASR/triphone_dnn_organized/data/test.mfcc.frm11.mean_var_norm_perUtt.core.mat');
test = test.test_set;
%% --normalization--
% ///////do this if mem is large enough and the data has not been normalized
% ///////for 8k.wet.timit, I did not do the following global normalization

% if opts.isNormalize
%     disp('normalizing data...');
%     if opts.ARMA_order
%         disp('ARMA filtering...');
%         [train_data,opts.tr_mu,opts.tr_std] = meanVarArmaNormalize(train_data,opts.ARMA_order);
%         for i = 1:length(test)
%             test(i).data = meanVarArmaNormalize_Test(test(i).data, opts.ARMA_order,opts.tr_mu,opts.tr_std);
%         end
%     else
%         [train_data,opts.tr_mu,opts.tr_std] = meanVarNormalize(train_data);
%         for i = 1:length(test)
%             test(i).data = meanVarNormalize_Test(test(i).data, opts.tr_mu, opts.tr_std);
%         end
%     end
%     disp('normalization done.');
% else
%     disp('skipped data normalization.');
% end
%% --network training--
% set final structure
[num_samples, dim_input] = size(train_data);
max_id = 1070; % hard coded
opts.net_struct = [dim_input, opts.hid_struct, max_id];
opts %#ok<NOPTS>

%///////set htk params///////
opts.htk.feat_files_dir = '/home/ywang/work/ASR/triphone_dnn_organized/feats/'; % must use full path
opts.htk.hop = 0.016; % or 0.01, depending on the .mfc files provided
opts.htk.hvite_path = '~/work/tools/htk/HTKTools/HVite';
opts.htk.hresults_path = '~/work/tools/htk/HTKTools/HResults';
opts.htk.htk_files_dir = '/home/ywang/work/ASR/triphone_dnn_organized/htk_files/';
opts.htk.hmmdef_path = [opts.htk.htk_files_dir, 'mmf_bypass'];
opts.htk.rec_mlf_path = [opts.htk.htk_files_dir, 'recout.mlf'];
opts.htk.score_path = [opts.htk.htk_files_dir, 'recout.mlf.score'];
opts.htk.ref_mlf_path = [opts.htk.htk_files_dir, 'test_all_timit_39.mlf'];
opts.htk.lang_model_path = [opts.htk.htk_files_dir, 'outLatFile_tri_timit+wsj0+wsj1_39'];
opts.htk.dict_path = [opts.htk.htk_files_dir, 'dict_tri_timit+wsj0+wsj1_sp_39'];
opts.htk.hmmlist_path = [opts.htk.htk_files_dir, 'tiedList_200_1500'];
opts.htk.hvite_opts = ' -T 0 -l ''*'' -y rec -o SNW -p 0.0 -s 3.0 '; 
opts.htk.hresults_opts = ' -A -s -f -T 1 -e ''???'' sil ';

% main training function
tic
% just for quick verification that everything works. 
cut = 300000;
[model, pre_net] = funcDeepNetTrainNoRolling_triphone(train_data(1:cut,:),train_target_ids(1:cut,:),test(1:5),opts);

% change back to this if the code runs
% [model, pre_net] = funcDeepNetTrainNoRolling_triphone(train_data,train_target_ids,test,opts);
train_time = toc;
fprintf('\nTraining done. Elapsed time: %2.2f sec\n',train_time);
