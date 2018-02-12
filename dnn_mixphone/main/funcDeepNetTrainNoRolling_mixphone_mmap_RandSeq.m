function [model, pre_net, cv_rec] = funcDeepNetTrainNoRolling_mixphone_mmap_RandSeq(data_mmap,size_info,cv_struct,opts)
%% network initialization
net_struct = opts.net_struct;
isGPU = opts.isGPU;

pre_net = randInitNet_mixphone(net_struct, isGPU);

net_iterative = pre_net;
num_net_layer = length(net_iterative);
%% stochastic gradient descent
% num_samples = size(train_data,1);
num_samples = size_info(1); 
max_id_mono = size_info(2);
max_id_tri = size_info(3);

train_chunck_batch_id = genBatchID(num_samples, opts.train_chunck_batch_size);
num_train_chunck_batch = size(train_chunck_batch_id, 2);

fprintf('\nNum of Training Samples:%d\n',num_samples);

net_weights_inc = zeroInitNet_mixphone(net_struct, opts.isGPU);
net_grad_ssqr = zeroInitNet_mixphone(net_struct, opts.isGPU, 1e-4);
net_ada_eta = zeroInitNet_mixphone(net_struct, opts.isGPU);

best_perf = 0;
cv_rec = repmat(struct,opts.sgd_max_epoch,1);
for epoch = 1:opts.sgd_max_epoch
    cost_sum = 0;
    seq = randperm(num_train_chunck_batch);
    for chunck_bid = 1:num_train_chunck_batch        
        range = train_chunck_batch_id(:,seq(chunck_bid)); % unlike below, range needs to be continuous
        tic
        clear train_data_chunk
        train_data_chunk = mmap_loadPartial(data_mmap, range(1):range(2));
        load_time = toc;
        fprintf('Chunck %d/%d; Loading time: %2.2f sec\n', chunck_bid, num_train_chunck_batch,load_time);
        batch_id = genBatchID(size(train_data_chunk,2), opts.sgd_batch_size);
        num_batch = size(batch_id,2);
        sub_seq = randperm(size(train_data_chunk,2));
        for bid = 1:num_batch
            %### 1st col: mono labels; 2nd col: tri labels; 3rd~end col: features
            perm_idx = sub_seq(batch_id(1,bid) : batch_id(2,bid));            
            batch_data_chunk = train_data_chunk(:, perm_idx)';            
            
            target_ids_mono = batch_data_chunk(:,1);
            filled_labels_mono = make_labels(target_ids_mono, max_id_mono);
            target_ids_tri = batch_data_chunk(:,2);
            filled_labels_tri = make_labels(target_ids_tri, max_id_tri);
            
            if opts.isGPU
                batch_label_mono = gpuArray(filled_labels_mono);
                batch_label_tri = gpuArray(filled_labels_tri);
                batch_data = gpuArray(batch_data_chunk(:,3:end));
            else
                batch_label_mono = filled_labels_mono;
                batch_label_tri = filled_labels_tri;
                batch_data = batch_data_chunk(:,3:end);
            end
            
            if epoch>opts.change_momentum_point;
                momentum=opts.final_momentum;
            else
                momentum=opts.initial_momentum;
            end
            
            %backprop: core code
            [cost,net_grad] = computeNetGradientNoRolling_mixphone(net_iterative, batch_data, batch_label_mono, batch_label_tri, opts);
            
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
    end
    fprintf('Objective cost at epoch %d: %2.2f \n', epoch, cost_sum);
    
    % check perf. on cv data
    if ~mod(epoch,opts.cv_interval)
        clear train_data_chunk
        
        disp('HVite Decoding for Monophone...')
        use_which = 1; % left output layer: mono targets        
        [perf1, perf_str] = check_PER_mixphone(net_iterative, cv_struct, use_which, opts);
        fprintf('Metric %s on dev_set at epoch %d: %2.4f \n', perf_str, epoch, perf1);
        
        disp('HVite Decoding for Triphone...')
        use_which = 2; % right output layer: triphone targets
        [perf2, perf_str] = check_PER_mixphone(net_iterative, cv_struct, use_which, opts);
        fprintf('Metric %s on dev_set at epoch %d: %2.4f \n', perf_str, epoch, perf2);
        
        perf = (perf1 + perf2) / 2;
        cv_rec(epoch*opts.cv_interval).perf = perf; % change this to desired
        
        if best_perf < perf;
            best_model = net_iterative;
            best_perf = perf;
            if opts.save_on_fly
                if ~exist(opts.save_model_path,'dir'); mkdir(opts.save_model_path); end                
                model_name = [opts.save_model_path, '/model.',getNetParamStr_mixphone(opts),'.bestModel.',opts.note_str,'.mat'];
                
                if opts.isGPU; best_model = gather_net_mixphone(best_model); end
                if opts.isGPU
                    last_net_grad_ssqr = gather_net_mixphone(net_grad_ssqr);
                    last_net_weights_inc = gather_net_mixphone(net_weights_inc);
                else
                    last_net_grad_ssqr = net_grad_ssqr;
                    last_net_weights_inc = net_weights_inc;
                end                
                
                save(model_name, 'best_model','last_net_grad_ssqr','last_net_weights_inc',...,
                    'opts','epoch','best_perf','perf_str', 'perf1', 'perf2','-v7.3');
                                
                fprintf('saved best model so far to %s\n', model_name);
            end
        end
        
    end
end

%% use the best model on dev_set
[m_v,m_i] = max([cv_rec.perf]);
fprintf('\nBest model at epoch %d with avg_perf %2.4f\n',m_i,m_v);
model = best_model;
