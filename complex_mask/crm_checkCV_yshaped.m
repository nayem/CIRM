%function [dev_perfs, dev_perf_strs] = crm_checkCV(newNet, dev_struct, opts)
function [dev_perfs, dev_perf_strs] = crm_checkCV_yshaped(newNet, dev_feats,dev_crm_re,dev_crm_im, opts)

% dev_feats  = dev_struct.dev_x;
% dev_crm_re = dev_struct.dev_crm_re;
% dev_crm_im = dev_struct.dev_crm_im;

batch_id    = genBatchID(size(dev_feats,1),opts.sgd_batch_size);
num_batch   = size(batch_id,2);
if num_batch == 1; num_batch = num_batch + 1; end;

seq         = randperm(size(dev_feats,1)); % randomize access if data in mem
dev_netout1 = single(zeros(size(dev_crm_re)));
dev_netout2 = single(zeros(size(dev_crm_im)));

for bid = 1:num_batch-1
    perm_idx = seq(batch_id(1,bid):batch_id(2,bid));
    
    if opts.isGPU
        batch_data  = gpuArray(dev_feats(perm_idx,:));
    else
        batch_data  = dev_feats(perm_idx,:);
    end
    
    [output1, output2]      = getOutputFromNet_mixphone_yshaped(newNet, batch_data, opts);
    dev_netout1(perm_idx,:) = gather(output1); dev_netout2(perm_idx,:) = gather(output2);

end

dev_perfs = -mean(sum( (dev_crm_re - dev_netout1).^2)) - mean(sum( (dev_crm_im - dev_netout2).^2));
dev_perf_strs = {'mse'};
