function [cost, net_gradients] = computeNetGradient_RealImag_yshaped(net, data, x_real, x_imag, y_real, y_imag, opts)

num_net_layer      = length(net);
unit_type_output   = opts.unit_type_output;
unit_type_hidden   = opts.unit_type_hidden;
num_sample         = size(data,1);
num_yshaped_layers = opt.num_yshaped_layers;

yshaped_layers = zeros(num_net_layer,1);
yshaped_count  = num_net_layer;

for lay = num_yshaped_layers:-1:1
    yshaped_layers(yshaped_count) = yshaped_count;
    yshaped_count = yshaped_count - 1;
end

forward_path = forwardPass_mixphone_yshaped(net, data, opts);
% output1 = forward_path{num_net_layer+1}{1}';
% output2 = forward_path{num_net_layer+1}{2}'; 
pred_real = forward_path{num_net_layer+1}{1}';
pred_imag = forward_path{num_net_layer+1}{2}';

%% cost function: softmax_xentropy    
% assert(strcmp(unit_type_output,'softmax'));

% pred_real = output1 .* x_real;
% pred_imag = output2 .* x_imag;

cost1 = 0.5 * sum(sum( (pred_real - y_real).^2)) / num_sample;
cost2 = 0.5 * sum(sum( (pred_imag - y_imag).^2)) / num_sample;

cost = cost1 + cost2;

% output_delta1 = -(y_real - pred_real) .* x_real .* compute_unit_gradient(output1,unit_type_output);
% output_delta2 = -(y_imag - pred_imag) .* x_imag .* compute_unit_gradient(output2,unit_type_output);

output_delta1 = -(y_real - pred_real) .* compute_unit_gradient(pred_real,unit_type_output,opts);
output_delta2 = -(y_imag - pred_imag) .* compute_unit_gradient(pred_imag,unit_type_output,opts);
%% backprop
net_gradients = zeroInitNet_mixphone(opts.net_struct,opts.isGPU);

% upper_layer_delta = output_delta;
for ll = num_net_layer: -1: 1
    
    if ll == yshaped_layers(ll)
        net_gradients(ll).Wo1 = (forward_path{ll}*output_delta1)'/num_sample;
        net_gradients(ll).bo1 = mean(output_delta1)';
        net_gradients(ll).Wo2 = (forward_path{ll}*output_delta2)'/num_sample;
        net_gradients(ll).bo2 = mean(output_delta2)';
        
        d1 = ((output_delta1*net(ll).Wo1)'.*compute_unit_gradient(forward_path{ll},unit_type_hidden,opts))';
        d2 = ((output_delta2*net(ll).Wo2)'.*compute_unit_gradient(forward_path{ll},unit_type_hidden,opts))';
        upper_layer_delta = d1 + d2;        
    else
        net_gradients(ll).W = (forward_path{ll}*upper_layer_delta)'/num_sample;
        net_gradients(ll).b = mean(upper_layer_delta)';
        upper_layer_delta = ((upper_layer_delta*net(ll).W)'.*compute_unit_gradient(forward_path{ll},unit_type_hidden,opts))';
    end         
    
end
