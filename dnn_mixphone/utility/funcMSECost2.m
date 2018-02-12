function [cost, net_gradients] = funcMSECost2(net_weights, data, label, opts)
[net, junk] = netRolling(net_weights, opts.net_struct);
assert(isempty(junk));

num_net_layer = length(net);
unit_type_output = opts.unit_type_output;
unit_type_hidden = opts.unit_type_hidden;

forward_path = forwardPass(net, data, opts);
num_sample = size(data,1);
output = forward_path{end}';
%% cost function: mse, xentropy, softmax_xentropy
switch opts.cost_function
    case 'mse'
        cost = 0.5*sum(sum((label-output).^2))/num_sample;
        output_delta = -(label-output).*compute_unit_gradient(output,unit_type_output);
    case 'xentropy'
        cost = -mean(label.*log(output) + (1-label).*log(1-output));
        if strcmp(unit_type_output,'sigm');
            output_delta = -(label-output);
        else
            output_delta = -(label-output)./(output.*(1-output)).*compute_unit_gradient(output,unit_type_output);
        end
    case 'softmax_xentropy'
        cost = -sum(sum(label.*log(output)))/num_sample;
        assert(strcmp(unit_type_output,'softmax'));
        output_delta = -(label-output);
end
%% backprop
if opts.isGPU
    net_gradients = gpuArray.zeros(size(net_weights));
else
    net_gradients = zeros(size(net_weights));
end

pos = length(net_weights);
upper_layer_delta = output_delta;
for ll = num_net_layer: -1: 1
    
    weights_gradient = (forward_path{ll}*upper_layer_delta)'/num_sample;
    bias_gradient = mean(upper_layer_delta)';
    
    [size_w_p,size_w_q] = size(weights_gradient);
    [size_b_p,size_b_q] = size(bias_gradient);
        
    net_gradients(pos-size_b_p*size_b_q+1:pos) = bias_gradient(:);
    pos = pos-size_b_p*size_b_q;
    net_gradients(pos-size_w_p*size_w_q+1:pos) = weights_gradient(:);
    pos = pos-size_w_p*size_w_q;

    delta = (upper_layer_delta*net(ll).W)'.*compute_unit_gradient(forward_path{ll},unit_type_hidden);
    upper_layer_delta = delta';
end
