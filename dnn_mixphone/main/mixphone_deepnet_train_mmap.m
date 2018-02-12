clear all;
gpuDevice(1);
addpath(genpath('~/work/ASR/mixphone/'))
addpath('~/work/tools/voicebox/');

matlabpool close force local
matlabpool open local 8
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
% opts.learner = 'sgd'; % 'ada_sgd' or 'sgd'
opts.sgd_max_epoch = 30;
opts.sgd_batch_size = 1024/2; % batch size for SGD
opts.ada_sgd_scale = 0.0015; % scaling factor for ada_grad
opts.sgd_learn_rate = linspace(0.08, 0.001, opts.sgd_max_epoch); % linearly decreasing lrate for plain sgd

opts.initial_momentum = 0.5;
opts.final_momentum = 0.9;
opts.change_momentum_point = 5;

opts.cost_function = 'softmax_xentropy';
% opts.hid_struct = [2048 2048 2048 2048 2048]; % num of hid layers and units
% opts.hid_struct = [512 512 512]; % num of hid layers and units
% opts.hid_struct = [512 512 512 512 512 512 512]; % num of hid layers and units
opts.hid_struct = [2048 2048 2048]; % num of hid layers and units
% opts.hid_struct = [2048]; % num of hid layers and units

opts.unit_type_output = 'softmax';
opts.unit_type_hidden = 'relu';
% opts.unit_type_hidden = 'sigm';
if strcmp(opts.unit_type_output,'softmax'); opts.cost_function = 'softmax_xentropy'; end;

opts.isDropout = 1; % need dropout regularization?
opts.isDropoutInput = 0; % dropout inputs? do not dropout if raw features are louzy
opts.drop_ratio = 0.2; % ratio of units to drop

opts.isGPU = 1; % use GPU?
opts.eval_on_gpu = 0; 

opts.save_on_fly = 1; % save the current best model along the way
opts.save_model_path = '~/work/ASR/mixphone/models/';
opts.note_str = 'cleanAlign_noisyFbank26_mixphone';
%% --load data--
% pointer to a binary file
dim_input = 78*11;
bin_path = '~/work/ASR/mixphone/data/train.fbank26.frm11.mean_var_norm_perUtt_cleanAlign_noisyFeat.tri_mono_targets.bin';
[data_mmap, num_samples] = create_mmap(bin_path, dim_input+2, 'single'); % +2 as we have two labels now

test = load('~/work/ASR/mixphone/data/test.fbank26.frm11.mean_var_norm_perUtt.babble_18_fbank26_core.mat');
test = test.test_set;
%%
%///////set triphone htk params///////
htk_tri.feat_files_dir = '/home/ywang/work/ASR/mixphone/feats/triphone/'; % must use full path
htk_tri.hop = 0.01; 
htk_tri.hvite_path = '~/work/tools/htk_modify/htk/HTKTools/HVite';
htk_tri.hresults_path = '~/work/tools/htk_modify/htk/HTKTools/HResults';

htk_tri.htk_files_dir = '/home/ywang/work/ASR/mixphone/hmm_defs_triphone/';
htk_tri.hmmdef_path = [htk_tri.htk_files_dir, 'mmf_bypass'];
htk_tri.rec_mlf_path = [htk_tri.htk_files_dir, 'dnn_recout_tri.mlf'];
htk_tri.score_path = [htk_tri.htk_files_dir, 'dnn_recout_tri.mlf.score'];

htk_tri.ref_mlf_path = [htk_tri.htk_files_dir, 'test_all_timit_39.mlf'];
htk_tri.lang_model_path = [htk_tri.htk_files_dir, 'outLatFile_tri_timit+wsj0+wsj1_39'];
htk_tri.dict_path = [htk_tri.htk_files_dir, 'dict_tri_timit+wsj0+wsj1_sp_39'];
htk_tri.hmmlist_path = [htk_tri.htk_files_dir, 'tiedList_200_1500'];
htk_tri.hvite_opts = ' -T 0 -l ''*'' -y rec -o SNW -p 0.0 -s 3.0 '; 
htk_tri.hresults_opts = ' -A -s -f -T 1 -e ''???'' sil ';

%///////set triphone htk params///////
htk_mono.feat_files_dir = '/home/ywang/work/ASR/mixphone/feats/monophone/'; % must use full path
htk_mono.hop = 0.01; 
htk_mono.hvite_path = '~/work/tools/htk_modify/htk/HTKTools/HVite';
htk_mono.hresults_path = '~/work/tools/htk_modify/htk/HTKTools/HResults';

htk_mono.htk_files_dir = '/home/ywang/work/ASR/mixphone/hmm_defs_monophone/';
htk_mono.hmmdef_path = [htk_mono.htk_files_dir, 'mmf_mono_bypass'];
htk_mono.rec_mlf_path = [htk_mono.htk_files_dir, 'dnn_recout_mono.mlf'];
htk_mono.score_path = [htk_mono.htk_files_dir, 'dnn_recout_mono.mlf.score'];

htk_mono.ref_mlf_path = [htk_mono.htk_files_dir, 'timitTest.mlf'];
htk_mono.lang_model_path = [htk_mono.htk_files_dir, 'outLatFile_mono'];
htk_mono.dict_path = [htk_mono.htk_files_dir, 'timitPhnDict'];
htk_mono.hmmlist_path = [htk_mono.htk_files_dir, 'monophones1'];
htk_mono.hvite_opts = ' -T 0 -l ''*'' -y rec -o SNW -p 0.0 -s 3.0 '; 
htk_mono.hresults_opts = ' -A -s -f -T 1 -e ''???'' sil ';

%% --network training--
% set final structure
max_id_mono = 120;
max_id_tri = 3316;
opts.net_struct = {dim_input};
for i = 1:length(opts.hid_struct)
    opts.net_struct{end+1} = opts.hid_struct(i);
end
opts.net_struct{end+1} = {max_id_mono, max_id_tri};
opts.htk = {htk_mono, htk_tri};

% main training function
tic
%an example: for quick verification
opts.train_chunck_batch_size = 1350000; % each time streams in and randomly permutes this many samples
% size_info = [num_samples, max_id]; % change to this for full training,
size_info = [1350000, max_id_mono, max_id_tri]; % change to this for full training, 

opts %#ok<NOPTS>
[model, pre_net, perf_rec] = funcDeepNetTrainNoRolling_mixphone_mmap_RandSeq(data_mmap,size_info,test(1:50),opts);

