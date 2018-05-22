function opts = InitiatlizeNN_cIRM(opts,trdata,trlabel)


%% default training options
opts.isNormalize = 1; % need to normalize data?
opts.ARMA_order = 0; % need to do ARMA smoothing and what's the filter order?
opts.cv_interval = 1; % check cv perf. every this many epochs

opts.isPretrain = 0; % pre-training using RBM?
opts.rbm_max_epoch = 0; 
opts.rbm_batch_size = 1024/2; % batch size for pretraining
opts.rbm_learn_rate_binary = 0.01;
opts.rbm_learn_rate_real = 0.004;

opts.learner = 'ada_sgd'; % 'ada_sgd' or 'sgd'
opts.sgd_max_epoch = 80;
opts.sgd_batch_size = 1024/4; % batch size for SGD
opts.ada_sgd_scale = 0.0015; % scaling factor for ada_grad
opts.ada_grad_eps   = eps;

opts.sgd_learn_rate = linspace(1, 0.001, opts.sgd_max_epoch); % linearly decreasing lrate for plain sgd

opts.initial_momentum      = 0.5;
opts.final_momentum        = 0.9;
opts.change_momentum_point = 5;

opts.cost_function = 'mse';
opts.hid_struct = 2*[512,512,512]; % num of hid layers and units
opts.unit_type_output = 'lin'; warning('using linear...')
opts.unit_type_hidden = 'relu';%'relu';
if strcmp(opts.unit_type_output,'softmax'); opts.cost_function = 'softmax_xentropy'; end;

opts.isDropout = 0; % need dropout regularization?
opts.isDropoutInput = 0; % dropout inputs? do not dropout if raw features are louzy
opts.drop_ratio = 0.2; % ratio of units to drop

opts.isGPU       = 1; % use GPU?
opts.eval_on_gpu = 0; 

opts.save_on_fly     = 0; % save the current best model along the way
%opts.save_model_path = [data_path, '/models/'];

%% Network structure
opts.dim_input  = size(trdata,2);%1771;      % Input dimensionality
opts.dim_output = size(trlabel,2);%161;      % Output dimensionality

opts.net_struct = [opts.dim_input, opts.hid_struct, opts.dim_output];

opts.split_tanh1_c1 = 1;
opts.split_tanh1_c2 = 2;

