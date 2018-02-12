%function [dev_perfs, dev_perf_strs] = crm_checkCV(newNet, dev_struct, opts)
function [dev_perfs, dev_perf_strs] = crm_checkCV(newNet, dev_feats,dev_label_real,dev_label_imag,dev_weights, opts)

% dev_feats  = dev_struct.dev_x;
% dev_crm_re = dev_struct.dev_crm_re;
% dev_crm_im = dev_struct.dev_crm_im;

batch_id    = genBatchID(size(dev_feats,1),opts.sgd_batch_size);
num_batch   = size(batch_id,2);
if num_batch == 1; num_batch = num_batch + 1; end;

seq         = randperm(size(dev_feats,1)); % randomize access if data in mem
dev_netout1 = single(zeros(size(dev_label_real)));
dev_netout2 = single(zeros(size(dev_label_imag)));

for bid = 1:num_batch-1
    perm_idx = seq(batch_id(1,bid):batch_id(2,bid));
    
    if opts.isGPU
        batch_data  = gpuArray(dev_feats(perm_idx,:));
    else
        batch_data  = dev_feats(perm_idx,:);
    end
    
    [output1, output2]      = getOutputFromNet_mixphone(newNet, batch_data, opts);
    dev_netout1(perm_idx,:) = gather(output1); dev_netout2(perm_idx,:) = gather(output2);
    
    clear batch_data
end

if strcmp(opts.cost_function,'mse') == 1
    dev_perfs = -mean(sum( (dev_label_real - dev_netout1).^2)) - mean(sum( (dev_label_imag - dev_netout2).^2));
    
elseif strcmp(opts.cost_function,'weighted_mse') == 1
    dev_perfs = -mean(sum( dev_weights.*(dev_label_real - dev_netout1).^2)) - mean(sum( dev_weights.*(dev_label_imag - dev_netout2).^2));
    
elseif strcmp(opts.cost_function,'sig_approx_mse') == 1   
    
    cRM_r = invLogisticFunction(dev_netout1,opts.c1,opts.c2);
    cRM_i = invLogisticFunction(dev_netout2,opts.c1,opts.c2);
    
    cRM = complex(cRM_r,cRM_i);
    
    est_stft = cRM.*dev_weights;
    
    pred_real2 = real(est_stft);
    pred_imag2 = imag(est_stft);
    
    cIRM_r = invLogisticFunction(dev_label_real,opts.c1,opts.c2);
    cIRM_i = invLogisticFunction(dev_label_imag,opts.c1,opts.c2);
    
    cIRM = complex(cIRM_r,cIRM_i);
    
    target_stft = cIRM.*dev_weights;
    label_real2 = real(target_stft);
    label_imag2 = imag(target_stft);
        
    %     cost1 = 0.5*sum(sum( (pred_real2 - label_real2).^2))/num_sample;
    %     cost2 = 0.5*sum(sum( (pred_imag2 - label_imag2).^2))/num_sample;
    
    dev_perfs = -mean(sum( (label_real2 - pred_real2).^2)) - mean(sum( (label_imag2 - pred_imag2).^2));
end

dev_perf_strs = {opts.cost_function};


end
