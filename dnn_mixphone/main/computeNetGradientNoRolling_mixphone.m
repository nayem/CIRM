function [cost, net_gradients] = computeNetGradientNoRolling_mixphone(net, data, label1, label2, opts)
num_net_layer = length(net);
unit_type_output = opts.unit_type_output;
unit_type_hidden = opts.unit_type_hidden;

forward_path = forwardPass_mixphone(net, data, opts);
num_sample = size(data,1);
output1 = forward_path{num_net_layer+1}{1}';
output2 = forward_path{num_net_layer+1}{2}';
%% cost function: softmax_xentropy    
assert(strcmp(unit_type_output,'softmax'));
cost1 = -sum(sum(label1.*log(output1)))/num_sample;
cost2 = -sum(sum(label2.*log(output2)))/num_sample;
cost = cost1 + cost2;

output_delta1 = -(label1-output1);
% output_delta1(isnan(output_delta1)) = 0;
output_delta2 = -(label2-output2);
%% backprop
net_gradients = zeroInitNet_mixphone(opts.net_struct,opts.isGPU);

% upper_layer_delta = output_delta;
for ll = num_net_layer: -1: 1
    
    if ll == num_net_layer
        net_gradients(ll).Wo1 = (forward_path{ll}*output_delta1)'/num_sample;
        net_gradients(ll).bo1 = mean(output_delta1)';
        net_gradients(ll).Wo2 = (forward_path{ll}*output_delta2)'/num_sample;
        net_gradients(ll).bo2 = mean(output_delta2)';
        
        d1 = ((output_delta1*net(ll).Wo1)'.*compute_unit_gradient(forward_path{ll},unit_type_hidden))';
        d2 = ((output_delta2*net(ll).Wo2)'.*compute_unit_gradient(forward_path{ll},unit_type_hidden))';
        upper_layer_delta = d1 + d2;        
    else
        net_gradients(ll).W = (forward_path{ll}*upper_layer_delta)'/num_sample;
        net_gradients(ll).b = mean(upper_layer_delta)';
        upper_layer_delta = ((upper_layer_delta*net(ll).W)'.*compute_unit_gradient(forward_path{ll},unit_type_hidden))';
    end         
    
end
