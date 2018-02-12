function [model, pre_net, cv_rec] = funcDeepNetTrainNoRolling_crm(train_feats,train_label_real, train_label_imag,train_weights,dev_feats,dev_label_real,dev_label_imag,dev_weights, opts)
%% network initialization
net_struct = opts.net_struct;
isGPU      = opts.isGPU;

pre_net = randInitNet_mixphone(net_struct, isGPU);
net_iterative = pre_net;

num_net_layer = length(net_iterative);

%% stochastic gradient descent
num_samples = size(train_feats,1);

batch_id = genBatchID(num_samples, opts.sgd_batch_size);
num_batch = size(batch_id, 2);

fprintf('\nNum of Training Samples:%d\n',num_samples);

net_weights_inc = zeroInitNet_mixphone(net_struct, opts.isGPU);
net_grad_ssqr   = zeroInitNet_mixphone(net_struct, opts.isGPU, opts.ada_grad_eps);
net_ada_eta     = zeroInitNet_mixphone(net_struct, opts.isGPU);

best_perf = -inf;
cv_rec    = repmat(struct,opts.sgd_max_epoch,1);

for epoch = 1:opts.sgd_max_epoch
    
    tic
    seq = randperm(num_samples); % randomize access if data in mem
    cost_sum = 0;
    
    for bid = 1:num_batch-1
        perm_idx = seq(batch_id(1,bid):batch_id(2,bid));
        
        batch_train_feats = gpuArray(train_feats(perm_idx,:));
        %         batch_x_real = gpuArray(x_real(perm_idx, :));
        %         batch_x_imag = gpuArray(x_imag(perm_idx, :));
        batch_label_real = gpuArray(train_label_real(perm_idx, :));
        batch_label_imag = gpuArray(train_label_imag(perm_idx, :));
        
        if strcmp(opts.cost_function,'weighted_mse') == 1 || strcmp(opts.cost_function,'sig_approx_mse') == 1
            batch_weights = gpuArray(train_weights(perm_idx,:));
        else
            batch_weights = [];
        end
        
        if epoch>opts.change_momentum_point;
            momentum=opts.final_momentum;
        else
            momentum=opts.initial_momentum;
        end
        
        %backprop: core code
        [cost, net_grad] = computeNetGradientNoRolling_RealImag_simple(net_iterative, batch_train_feats,batch_weights,...,
            batch_label_real, batch_label_imag, opts);
        
        clear batch_train_feats batch_weights batch_label_real batch_label_imag
        
        %update each layer without rolling the net
        for ll = 1:num_net_layer
            switch opts.learner
                case 'sgd'
                    if ll == num_net_layer
                        net_weights_inc(ll).Wo1 = momentum*net_weights_inc(ll).Wo1 + opts.sgd_learn_rate(epoch)*net_grad(ll).Wo1;
                        net_weights_inc(ll).bo1 = momentum*net_weights_inc(ll).bo1 + opts.sgd_learn_rate(epoch)*net_grad(ll).bo1;
                        net_weights_inc(ll).Wo2 = momentum*net_weights_inc(ll).Wo2 + opts.sgd_learn_rate(epoch)*net_grad(ll).Wo2;
                        net_weights_inc(ll).bo2 = momentum*net_weights_inc(ll).bo2 + opts.sgd_learn_rate(epoch)*net_grad(ll).bo2;
                    else
                        net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + opts.sgd_learn_rate(epoch)*net_grad(ll).W;
                        net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + opts.sgd_learn_rate(epoch)*net_grad(ll).b;
                    end
                case 'ada_sgd'
                    if ll == num_net_layer
                        net_grad_ssqr(ll).Wo1 = net_grad_ssqr(ll).Wo1 + (net_grad(ll).Wo1).^2;
                        net_grad_ssqr(ll).bo1 = net_grad_ssqr(ll).bo1 + (net_grad(ll).bo1).^2;
                        net_grad_ssqr(ll).Wo2 = net_grad_ssqr(ll).Wo2 + (net_grad(ll).Wo2).^2;
                        net_grad_ssqr(ll).bo2 = net_grad_ssqr(ll).bo2 + (net_grad(ll).bo2).^2;
                        net_ada_eta(ll).Wo1 = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).Wo1);
                        net_ada_eta(ll).bo1 = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).bo1);
                        net_ada_eta(ll).Wo2 = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).Wo2);
                        net_ada_eta(ll).bo2 = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).bo2);
                        
                        net_weights_inc(ll).Wo1 = momentum*net_weights_inc(ll).Wo1 + net_ada_eta(ll).Wo1.*net_grad(ll).Wo1;
                        net_weights_inc(ll).bo1 = momentum*net_weights_inc(ll).bo1 + net_ada_eta(ll).bo1.*net_grad(ll).bo1;
                        net_weights_inc(ll).Wo2 = momentum*net_weights_inc(ll).Wo2 + net_ada_eta(ll).Wo2.*net_grad(ll).Wo2;
                        net_weights_inc(ll).bo2 = momentum*net_weights_inc(ll).bo2 + net_ada_eta(ll).bo2.*net_grad(ll).bo2;
                    else
                        net_grad_ssqr(ll).W = net_grad_ssqr(ll).W + (net_grad(ll).W).^2;
                        net_grad_ssqr(ll).b = net_grad_ssqr(ll).b + (net_grad(ll).b).^2;
                        net_ada_eta(ll).W = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).W);
                        net_ada_eta(ll).b = opts.ada_sgd_scale./sqrt(net_grad_ssqr(ll).b);
                        net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + net_ada_eta(ll).W.*net_grad(ll).W;
                        net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + net_ada_eta(ll).b.*net_grad(ll).b;
                    end
            end
            
            if ll == num_net_layer
                net_iterative(ll).Wo1 = net_iterative(ll).Wo1 - net_weights_inc(ll).Wo1;
                net_iterative(ll).bo1 = net_iterative(ll).bo1 - net_weights_inc(ll).bo1;
                net_iterative(ll).Wo2 = net_iterative(ll).Wo2 - net_weights_inc(ll).Wo2;
                net_iterative(ll).bo2 = net_iterative(ll).bo2 - net_weights_inc(ll).bo2;
            else
                net_iterative(ll).W = net_iterative(ll).W - net_weights_inc(ll).W;
                net_iterative(ll).b = net_iterative(ll).b - net_weights_inc(ll).b;
            end
        end
        
        cost_sum = cost_sum + cost;
    end
    fprintf('Objective cost at epoch %d: %2.4f \n', epoch, cost_sum);
    
    
    if ~mod(epoch,opts.cv_interval)
        disp('Predicting...')
        [perfs, perf_strs] = crm_checkCV(net_iterative,dev_feats,dev_label_real,dev_label_imag,dev_weights,opts);
        perf = perfs(end);
        
        fprintf('Metrics at epoch %d --> ', epoch);
        for kk = 1:length(perfs)
            fprintf('%s: %2.4f ', perf_strs{kk}, perfs(kk))
        end
        fprintf('\n');
        
        cv_rec(epoch*opts.cv_interval).perf = perf;
        if perf > best_perf
            best_model = net_iterative;
            best_perf = perf;
            fprintf('****Best model so far ************************\n');
        end
        
    end
    elapsedtime = toc;
    fprintf('Elapsed Time: %0.3f (sec), %0.3f (min)\n\n',elapsedtime,elapsedtime/60)
    
end


%% use the best model on dev_set
[m_v,m_i] = max([cv_rec.perf]);
fprintf('\nBest model at epoch %d with avg_perf %2.4f\n',m_i,m_v);
if opts.isGPU
    model = gather_net_mixphone(best_model);%gather_net_yshaped(best_model);
else
    model = best_model;
end

