% function [cost, net_gradients] = funcMSECost(net_weights, data, label, opts)
function [cost, grads] = funcMSECost(net_weights, data, label, opts)
[net, junk] = netRolling(net_weights, opts.net_struct);
assert(isempty(junk));

num_net_layer = length(net);
num_data_layer = num_net_layer+1;

unit_types = opts.unit_types;
forward_path = forwardPass(net, data, opts);
num_sample = size(data,1);
output = forward_path{end}';
%% cost function: mse, xentropy, softmax_xentropy
switch opts.cost_function
    case 'mse'
        cost = 0.5*sum(sum((label-output).^2))/num_sample;
        output_delta = -(label-output).*compute_unit_gradient(output,unit_types{num_data_layer});
%         unit_types{num_data_layer}
    case 'xentropy'
        cost = -mean(label.*log(output) + (1-label).*log(1-output));
        output_delta = -(label-output);
    case 'softmax_xentropy'
        cost = -sum(sum(label.*log(prob)))/num_sample;
        output_delta = -(label-output);
end
%% backprop
net_gradients = repmat(struct,num_net_layer,1);
% net_gradients = [];

upper_layer_delta = output_delta;
for ll = num_net_layer: -1: 1 
    net_gradients(ll).weights_gradient = (forward_path{ll}*upper_layer_delta)'/num_sample;    
    net_gradients(ll).bias_gradient = mean(upper_layer_delta)';

%     weights_gradient = (forward_path{ll}*upper_layer_delta)'/num_sample;
%     bias_gradient = mean(upper_layer_delta)';
%     net_gradients = [weights_gradient(:); bias_gradient(:); net_gradients]; % TODO: pre-allocate this

    delta = (upper_layer_delta*net(ll).W)'.*compute_unit_gradient(forward_path{ll},unit_types{ll+1});
%     delta = (upper_layer_delta*net(ll).W)'.*relu_grad(forward_path{ll});
%     unit_types{ll}
    upper_layer_delta = delta';
end

grads = [];
for ll = 1:num_net_layer
    grads = [grads; net_gradients(ll).weights_gradient(:); net_gradients(ll).bias_gradient(:)];
end
