addpath(genpath('~/work/ASR/triphone_timit_wet_8k/'));
addpath(genpath('~/work/tools/'));
%% -- load training/testing data
load('~/work/ASR/triphone_timit_wet_8k/data_200_1500/train.mfcc.frm11.mean_var_norm_perUtt.matrix.mat')

test = load('~/work/ASR/triphone_timit_wet_8k/data_200_1500/test.mfcc.frm11.mean_var_norm_perUtt.core.mat');
test = test.test_set;
%% -- load previous best model --
prev_model = load('~/work/ASR/triphone_timit_wet_8k/models/model.264.2048.2048.2048.2048.2048.1070.relu.ada_sgd.dropout.bestModel.22-Jun-2013.mat');
opts = prev_model.opts;
% reset some parameter
opts.initial_momentum = 0.9; opts.final_momentum = 0.9;
opts.sgd_max_epoch = 25;
opts.save_model_path = '~/work/ASR/triphone_timit_wet_8k/resume_models/';
opts.note_str = 'resumed_from_ep15';

isGPU = opts.isGPU;
net_iterative = prev_model.best_model;
num_net_layer = length(net_iterative);
net_struct = opts.net_struct;
%% stochastic gradient descent
opts %#ok<NOPTS>
num_samples = size(train_data,1);
batch_id = genBatchID(num_samples,opts.sgd_batch_size);
num_batch = size(batch_id,2);

fprintf('\nNum of Training Samples:%d\n',num_samples);

%/// these need also be replaced!!!
net_weights_inc = zeroInitNet(net_struct, opts.isGPU);
net_grad_ssqr = zeroInitNet(net_struct, opts.isGPU, eps);
net_ada_eta = zeroInitNet(net_struct, opts.isGPU);

best_perf = 0;
cv_rec = repmat(struct,opts.sgd_max_epoch,1);
for epoch = 1:opts.sgd_max_epoch
    seq = randperm(num_samples); % randomize access if data in mem
%     seq = 1:num_samples; % must do sequential reading with mmap, o/w slow!
    cost_sum = 0;
    for bid = 1:num_batch-1
        perm_idx = seq(batch_id(1,bid):batch_id(2,bid));
        
        if isGPU
            batch_data = gpuArray(train_data(perm_idx,:));            
            filled_labels = make_labels(train_target_ids(perm_idx), 1070);
            batch_label = gpuArray(filled_labels);
        else
%             batch_data = train_data(perm_idx,:);
%             batch_label = train_label(perm_idx,:);
        end
        
        if epoch>opts.change_momentum_point;
            momentum=opts.final_momentum;
        else
            momentum=opts.initial_momentum;
        end
        
        %backprop: core code
        [cost,net_grad] = computeNetGradientNoRolling(net_iterative, batch_data, batch_label, opts);
        
        %update each layer without rolling the net
        for ll = 1:num_net_layer
            switch opts.learner
                case 'sgd'
                    net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + opts.sgd_learn_rate(epoch)*net_grad(ll).W;
                    net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + opts.sgd_learn_rate(epoch)*net_grad(ll).b;
                case 'ada_sgd'
                    net_grad_ssqr(ll).W = net_grad_ssqr(ll).W + (net_grad(ll).W).^2;
                    net_grad_ssqr(ll).b = net_grad_ssqr(ll).b + (net_grad(ll).b).^2;
                    
                    net_ada_eta(ll).W = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).W);
                    net_ada_eta(ll).b = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).b);
                                        
                    net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + net_ada_eta(ll).W.*net_grad(ll).W;
                    net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + net_ada_eta(ll).b.*net_grad(ll).b;
            end
            
            net_iterative(ll).W = net_iterative(ll).W - net_weights_inc(ll).W;
            net_iterative(ll).b = net_iterative(ll).b - net_weights_inc(ll).b;
        end
        
        cost_sum = cost_sum + cost;
    end
    fprintf('Objective cost at epoch %d: %2.2f \n', epoch, cost_sum);
    
    % check perf. on cv data
    if ~mod(epoch,opts.cv_interval)        
        disp('HVite Decoding...')
        [perf, perf_str] = check_PER_timit_triphone_coreset(net_iterative,test,opts);
        fprintf('Metric %s on dev_set at epoch %d: %2.4f \n', perf_str, epoch, perf);       
%         perf = 2;
        
        cv_rec(epoch*opts.cv_interval).perf = perf;
%         cv_rec(epoch*opts.cv_interval).model = gather_net(net_iterative);
         
        if best_perf < perf;
            best_model = net_iterative;
            best_perf = perf;
            if opts.save_on_fly
                if ~exist(opts.save_model_path,'dir'); mkdir(opts.save_model_path); end
                
                model_name = [opts.save_model_path, '/model.',getNetParamStr(opts),'.bestModel.', opts.note_str, '.', date,'.mat'];
                best_model = gather_net(best_model);
                save(model_name, 'best_model','opts','epoch','best_perf','perf_str','-v7.3');
                fprintf('saved best model so far to %s\n', model_name);
            end
        end
                
    end
end

%% use the best model on dev_set
[m_v,m_i] = max([cv_rec.perf]);
fprintf('\nBest model at epoch %d with perf %2.4f\n',m_i,m_v);
% model = cv_rec(m_i*opts.cv_interval).model;
model = best_model;
