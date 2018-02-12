clear all;
%%
% gpuDevice
% addpath(genpath('~/research/adtnoise/'))
%% ids
% db = -5;
db = 5;
% db = 10;
% db = 0;
dbfix = get_dbstr(db); dbfix = dbfix(1:end-1);

% nsname = 'adtBabble';
nsname = 'adtCafe';
% nsname = 'ssn';

feat_name = 'comp+gf';
% feat_name = 'comp+RI';
% feat_name = 'gf';
% feat_name = 'mixRealImag';

feat_flank = 2;
% feat_flank = 5;
% feat_flank = 1;
tar_flank = 0;
% tar_flank = 2;
%% --setting net params--
data_path = '~/research/adtnoise/data/';
% default training options
opts.isNormalize = 1; % need to normalize data?
opts.ARMA_order = 0; % need to do ARMA smoothing and what's the filter order?
opts.cv_interval = 1; % check cv perf. every this many epochs

opts.isPretrain = 0; % pre-training using RBM?
opts.rbm_max_epoch = 0; 
opts.rbm_batch_size = 1024/2; % batch size for pretraining
opts.rbm_learn_rate_binary = 0.01;
opts.rbm_learn_rate_real = 0.004;

opts.learner = 'ada_sgd'; % 'ada_sgd' or 'sgd'
% opts.learner = 'sgd'; % 'ada_sgd' or 'sgd'
opts.sgd_max_epoch = 50;
% opts.sgd_max_epoch = 150;
opts.sgd_batch_size = 1024/2; % batch size for SGD
opts.ada_sgd_scale = 0.005; % scaling factor for ada_grad
% opts.ada_sgd_scale = 0.0015; % scaling factor for ada_grad

opts.sgd_learn_rate = linspace(1, 0.001, opts.sgd_max_epoch); % linearly decreasing lrate for plain sgd

opts.initial_momentum = 0.5;
opts.final_momentum = 0.9;
opts.change_momentum_point = 5;

opts.cost_function = 'mse';

% opts.hid_struct = 2*[512]; % num of hid layers and units
opts.hid_struct = 2*[512,512,512]; % num of hid layers and units
% opts.hid_struct = 2*[512,512,512,512]; % num of hid layers and units
% opts.hid_struct = 4*[512,512,512,512]; % num of hid layers and units
% opts.hid_struct = 4*[512,512,512]; % num of hid layers and units
% opts.unit_type_output = 'scaled_sigmoid'; warning('using scaled sigmoid...')
% opts.unit_type_output = 'scaled_tanh'; warning('using scaled tanh...')
% opts.unit_type_output = 'shifted_tanh'; warning('using shifted tanh...')
% opts.unit_type_output = 'tanh'; warning('using tanh...')
opts.unit_type_output = 'lin'; warning('using linear...')

% opts.unit_type_output = 'sigm'; 

opts.unit_type_hidden = 'relu';
if strcmp(opts.unit_type_output,'softmax'); opts.cost_function = 'softmax_xentropy'; end;

opts.isDropout = 1; % need dropout regularization?
opts.isDropoutInput = 0; % dropout inputs? do not dropout if raw features are louzy
opts.drop_ratio = 0.2; % ratio of units to drop

opts.isGPU = 1; % use GPU?
opts.eval_on_gpu = 0; 

opts.save_on_fly = 1; % save the current best model along the way
% opts.save_on_fly = 0; % save the current best model along the way
opts.save_model_path = [data_path, '/models/'];
%% --load data--
nRep = 10;
rep_str = ['rep',int2str(nRep)];

opts.nsname = nsname; opts.feat_name = feat_name; 

dim_y = 161;
target_fpath = [data_path, '/feats/', 'feat', '.', nsname, '.mix_real.', dbfix, '.', rep_str, '.bin'];
fprintf('loading from %s \n', target_fpath);
train_x_real = single(read_binary(target_fpath, dim_y, 'single'))';

target_fpath = [data_path, '/feats/', 'feat', '.', nsname, '.mix_imag.', dbfix, '.', rep_str, '.bin'];
fprintf('loading from %s \n', target_fpath);
train_x_imag = single(read_binary(target_fpath, dim_y, 'single'))';

target_fpath = [data_path, '/feats/', 'feat', '.', nsname, '.clean_real.', dbfix, '.', rep_str, '.bin'];
fprintf('loading from %s \n', target_fpath);
train_y_real = single(read_binary(target_fpath, dim_y, 'single'))';

target_fpath = [data_path, '/feats/', 'feat', '.', nsname, '.clean_imag.', dbfix, '.', rep_str, '.bin'];
fprintf('loading from %s \n', target_fpath);
train_y_imag = single(read_binary(target_fpath, dim_y, 'single'))';


disp('loading training features...');
switch feat_name
    case 'comp+gf'
        dim_x = 59+64;
%         if db == -5
%             feat_fpath = [data_path, '/feats/', 'feat', '.', nsname, '.', feat_name, '.' ,dbfix, '.bin']; % m5db
%         else
        feat_fpath = [data_path, '/feats/', 'feat', '.', nsname, '.', feat_name, '.' ,dbfix, '.', rep_str, '.bin'];
%         end
        train_x = single(read_binary(feat_fpath, dim_x, 'single'))';
        train_x = [train_x deltas(train_x)];
    case 'comp+RI'
        dim_x = 59+64;
%         if db == -5
%             feat_fpath = [data_path, '/feats/', 'feat', '.', nsname, '.', feat_name, '.' ,dbfix, '.bin']; % m5db
%         else
        feat_fpath = [data_path, '/feats/', 'feat', '.', nsname, '.', 'comp+gf', '.' ,dbfix, '.', rep_str, '.bin'];
%         end
        train_x = single(read_binary(feat_fpath, dim_x, 'single'))';
%         train_x = [train_x deltas(train_x)];        
        train_x = [train_x train_x_real train_x_imag];
    case 'gf'
        dim_x = 59+64;
        feat_fpath = [data_path, '/feats/', 'feat', '.', nsname, '.comp+gf.' ,dbfix, '.', rep_str, '.bin'];
        train_x = single(read_binary(feat_fpath, dim_x, 'single'))';
        train_x = train_x(:, end-63:end);
%         train_x = [train_x deltas(train_x)];
    case 'mixRealImag'
        dim_x = 161;
        train_x = [train_x_real train_x_imag];
        warning('did not do deltas for mixRealImag, just windowing...')
end

% assert(2==1);
% cut = 500000;
% train_x = train_x(1:cut,:);
% train_x_real = train_x_real(1:cut,:); train_x_imag = train_x_imag(1:cut,:);
% train_y_real = train_y_real(1:cut,:); train_y_imag = train_y_imag(1:cut,:);


% delta + normalization
disp('feature normalization...')
% arma_order = 0; 
arma_order = 2; 
opts.arma_order = arma_order;
[train_x, opts.tr_mu, opts.tr_std] = meanVarArmaNormalize(train_x, arma_order);
train_x(isnan(train_x)) = 0;

% windowing
disp('windowing...');
opts.feat_flank = feat_flank; opts.tar_flank = tar_flank;
if feat_flank ~= 0
    train_x = makeWindowFeat2_fix(train_x, feat_flank);
end

if tar_flank ~= 0
    train_x_real = makeWindowFeat2_fix(train_x_real, tar_flank);
    train_x_imag = makeWindowFeat2_fix(train_x_imag, tar_flank);
    
    train_y_real = makeWindowFeat2_fix(train_y_real, tar_flank);
    train_y_imag = makeWindowFeat2_fix(train_y_imag, tar_flank);
end


% make actual train and dev set
dev_cut = round(size(train_x,1)*0.9);
% dev_cut = 1500000;
% dev_cut = 1450000;
% dev_cut = 1465000;

crm_clip = 10;
opts.crm_clip = crm_clip;
dev_x = train_x(dev_cut+1:end,:); 
dev_x_real = train_x_real(dev_cut+1:end,:);
dev_x_imag = train_x_imag(dev_cut+1:end,:);
dev_y_real = train_y_real(dev_cut+1:end,:);
dev_y_imag = train_y_imag(dev_cut+1:end,:);
[dev_crm_re, dev_crm_im] = getComplexMask(dev_x_real, dev_x_imag, dev_y_real, dev_y_imag, crm_clip);

train_x = train_x(1:dev_cut,:); 
train_x_real = train_x_real(1:dev_cut,:);
train_x_imag = train_x_imag(1:dev_cut,:);
train_y_real = train_y_real(1:dev_cut,:);
train_y_imag = train_y_imag(1:dev_cut,:);
[train_crm_re, train_crm_im] = getComplexMask(train_x_real, train_x_imag, train_y_real, train_y_imag, crm_clip);


dev_struct = struct;
dev_struct.dev_x = dev_x; 
dev_struct.dev_crm_re = dev_crm_re;
dev_struct.dev_crm_im = dev_crm_im;
clear dev_x dev_x_real dev_x_imag dev_y_real dev_y_imag


disp('Training set loading done...')
%%
% assert(2==1);
%% --network training--
% set final structure

d_in = size(train_x,2);
d_out1 = size(train_y_real,2);
d_out2 = size(train_y_imag,2);
d_out = {d_out1, d_out2};

opts.net_struct = {d_in};
for i = 1:length(opts.hid_struct)
    opts.net_struct{end+1} = opts.hid_struct(i);
end
opts.net_struct{end+1} = d_out;


opts.ada_grad_eps = eps;
% opts.ada_grad_eps = 1e-5;

% opts.note_str = [nsname, '_', feat_name,'_realimag.', dbfix];  
% opts.note_str = [nsname, '_', feat_name,'_realimag.scaled_sigm.', dbfix];  
% opts.note_str = [nsname, '_', feat_name,'_realimag.', dbfix, '.debug'];  
opts.note_str = [nsname, '_', feat_name,'_crm.', dbfix, '.new'];
% opts.note_str = [nsname, '_', feat_name,'_crm.', dbfix, '.realonly'];
% opts.note_str = [nsname, '_', feat_name,'_crm.', dbfix, '.imagonly'];

opts.sgd_max_epoch = 80;
% opts.ada_sgd_scale = 0.003; % scaling factor for ada_grad
opts.ada_sgd_scale = 0.0015; % scaling factor for ada_grad
opts.isDropout = 0;
% opts.isDropout = 1;

opts %#ok<NOPTS>

% cut = 50000;
% [model, pre_net] = funcDeepNetTrainNoRolling_crm(train_x(1:cut,:),...,
%     train_crm_re(1:cut,:), train_crm_im(1:cut,:), dev_struct, opts);

% both
[model, pre_net] = funcDeepNetTrainNoRolling_crm(train_x,  train_crm_re, train_crm_im, dev_struct, opts);
% real only
% [model, pre_net] = funcDeepNetTrainNoRolling_crm(train_x,  train_crm_re, train_crm_re, dev_struct, opts);
% imag only
% [model, pre_net] = funcDeepNetTrainNoRolling_crm(train_x,  train_crm_im, train_crm_im, dev_struct, opts);

disp('done.')
