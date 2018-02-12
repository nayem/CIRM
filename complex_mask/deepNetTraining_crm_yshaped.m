function [model, pre_net, cv_rec] = deepNetTraining_crm_yshaped(train_x,  y_real, y_imag,dev_x,dev_crm_re,dev_crm_im, opts)
%% network initialization
net_struct         = opts.net_struct;
isGPU              = opts.isGPU;
num_yshaped_layers = opt.num_yshaped_layers;

pre_net = randInitNet_mixphone_yshaped(net_struct, isGPU,num_yshaped_layers);

net_iterative = pre_net;
num_net_layer = length(net_iterative);

yshaped_layers = zeros(num_net_layer,1);
yshaped_count  = num_net_layer;

for lay = num_yshaped_layers:-1:1
    yshaped_layers(yshaped_count) = yshaped_count;
    yshaped_count = yshaped_count - 1;
end

%% stochastic gradient descent
num_samples = size(train_x,1);

batch_id = genBatchID(num_samples, opts.sgd_batch_size);
num_batch = size(batch_id, 2);

fprintf('\nNum of Training Samples:%d\n',num_samples);

net_weights_inc = zeroInitNet_mixphone_yshaped(net_struct, opts.isGPU,0,num_yshaped_layers);
net_grad_ssqr   = zeroInitNet_mixphone_yshaped(net_struct, opts.isGPU,opts.ada_grad_eps,num_yshaped_layers);
net_ada_eta     = zeroInitNet_mixphone_yshaped(net_struct, opts.isGPU,0,num_yshaped_layers);

best_perf = -inf;
cv_rec    = repmat(struct,opts.sgd_max_epoch,1);

for epoch = 1:opts.sgd_max_epoch
    
    tic
    seq      = randperm(num_samples); % randomize access if data in mem
    cost_sum = 0;
    
    for bid = 1:num_batch-1
        
        perm_idx = seq(batch_id(1,bid):batch_id(2,bid));
                
        batch_train_x = gpuArray(train_x(perm_idx,:));
%         batch_x_real = gpuArray(x_real(perm_idx, :));
%         batch_x_imag = gpuArray(x_imag(perm_idx, :));
        batch_y_real = gpuArray(y_real(perm_idx, :));
        batch_y_imag = gpuArray(y_imag(perm_idx, :));
        
        if epoch>opts.change_momentum_point;
            momentum=opts.final_momentum;
        else
            momentum=opts.initial_momentum;
        end
        
        %backprop: core code
        [cost, net_grad] = computeNetGradient_RealImag_yshaped(net_iterative, batch_train_x, 0, 0,...,
                                                                         batch_y_real, batch_y_imag, opts);

        %update each layer without rolling the net
        for ll = 1:num_net_layer
            switch opts.learner
                case 'sgd'
                    if ll == yshaped_layers(ll)
                        net_weights_inc(ll).Wo1 = momentum*net_weights_inc(ll).Wo1 + opts.sgd_learn_rate(epoch)*net_grad(ll).Wo1;
                        net_weights_inc(ll).bo1 = momentum*net_weights_inc(ll).bo1 + opts.sgd_learn_rate(epoch)*net_grad(ll).bo1;
                        net_weights_inc(ll).Wo2 = momentum*net_weights_inc(ll).Wo2 + opts.sgd_learn_rate(epoch)*net_grad(ll).Wo2;
                        net_weights_inc(ll).bo2 = momentum*net_weights_inc(ll).bo2 + opts.sgd_learn_rate(epoch)*net_grad(ll).bo2;
                    else
                        net_weights_inc(ll).W = momentum*net_weights_inc(ll).W + opts.sgd_learn_rate(epoch)*net_grad(ll).W;
                        net_weights_inc(ll).b = momentum*net_weights_inc(ll).b + opts.sgd_learn_rate(epoch)*net_grad(ll).b;
                    end
                case 'ada_sgd'
                    if ll == yshaped_layers(ll)
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
            
            if ll == yshaped_layers(ll)
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
    fprintf('Objective cost at epoch %d: %2.2f \n', epoch, cost_sum);
    
    
    if ~mod(epoch,opts.cv_interval)
        disp('Predicting...')
%         [perfs, perf_strs] = realimag_checkCV(net_iterative, dev_struct, opts);
        %[perfs, perf_strs] = crm_checkCV(net_iterative, dev_struct, opts);
        [perfs, perf_strs] = crm_checkCV_yshaped(net_iterative,dev_x,dev_crm_re,dev_crm_im,opts);
        perf = perfs(end);
%         perf = inf;
        
        fprintf('Metrics at epoch %d --> ', epoch);
        for kk = 1:length(perfs)
            fprintf('%s: %2.4f ', perf_strs{kk}, perfs(kk))
        end
        fprintf('\n');
        
        cv_rec(epoch*opts.cv_interval).perf = perf;
        if perf > best_perf
%         if -inf < perf;
            best_model = net_iterative;
            best_perf = perf;
            fprintf('****Best model so far ************************\n');

            if opts.save_on_fly
                if ~exist(opts.save_model_path,'dir'); mkdir(opts.save_model_path); end
                
                model_name = [opts.save_model_path, '/model.',getNetParamStr_mixphone(opts),'.bestModel.',opts.note_str,'.mat'];
                
                if opts.isGPU; best_model = gather_net_yshaped(best_model); end                
                if opts.isGPU
                    last_net_grad_ssqr = gather_net_yshaped(net_grad_ssqr);
                    last_net_weights_inc = gather_net_yshaped(net_weights_inc);
                else
                    last_net_grad_ssqr = net_grad_ssqr;
                    last_net_weights_inc = net_weights_inc;
                end
                
                save(model_name, 'best_model','last_net_grad_ssqr','last_net_weights_inc',...,
                    'opts','epoch','perfs','perf_strs','-v7.3');
                fprintf('saved best model so far to %s\n', model_name);
            end
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

