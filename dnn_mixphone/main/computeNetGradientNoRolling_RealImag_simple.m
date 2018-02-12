function [cost, net_gradients] = computeNetGradientNoRolling_RealImag_simple(net, feat_data, weights,label_real, label_imag, opts)

num_net_layer    = length(net);
unit_type_output = opts.unit_type_output;
unit_type_hidden = opts.unit_type_hidden;

forward_path = forwardPass_mixphone(net, feat_data, opts);
num_sample = size(feat_data,1);
% output1 = forward_path{num_net_layer+1}{1}';
% output2 = forward_path{num_net_layer+1}{2}'; 
pred_real = forward_path{num_net_layer+1}{1}';
pred_imag = forward_path{num_net_layer+1}{2}';

%% cost function: softmax_xentropy    
% assert(strcmp(unit_type_output,'softmax'));

% pred_real = output1 .* x_real;
% pred_imag = output2 .* x_imag;

if strcmp(opts.cost_function,'mse') == 1
    cost1 = 0.5*sum(sum( (pred_real - label_real).^2))/num_sample;
    cost2 = 0.5*sum(sum( (pred_imag - label_imag).^2))/num_sample;
    
    cost = cost1 + cost2;
    
    output_delta1 = -(label_real - pred_real).*compute_unit_gradient(pred_real,unit_type_output,opts);
    output_delta2 = -(label_imag - pred_imag).*compute_unit_gradient(pred_imag,unit_type_output,opts);

elseif strcmp(opts.cost_function,'weighted_mse') == 1
    cost1 = 0.5*sum(sum( weights.*(pred_real - label_real).^2))/num_sample;
    cost2 = 0.5*sum(sum( weights.*(pred_imag - label_imag).^2))/num_sample;
    
    cost = cost1 + cost2;
    
    output_delta1 = -weights.*(label_real - pred_real).*compute_unit_gradient(pred_real,unit_type_output,opts);
    output_delta2 = -weights.*(label_imag - pred_imag).*compute_unit_gradient(pred_imag,unit_type_output,opts);

elseif strcmp(opts.cost_function,'sig_approx_mse') == 1
    
    cRM_r = invLogisticFunction(pred_real,opts.c1,opts.c2);
    cRM_i = invLogisticFunction(pred_imag,opts.c1,opts.c2);
    
    cRM = complex(cRM_r,cRM_i); clear cRM_r cRM_i
    
    est_stft = cRM.*weights; clear cRM
    
    pred_real2 = real(est_stft);
    pred_imag2 = imag(est_stft); clear est_stft
    
    cIRM_r = invLogisticFunction(label_real,opts.c1,opts.c2);
    cIRM_i = invLogisticFunction(label_imag,opts.c1,opts.c2);
    
    cIRM = complex(cIRM_r,cIRM_i); clear cIRM_r cIRM_i
    
    target_stft = cIRM.*weights; clear cIRM
    label_real2 = real(target_stft);
    label_imag2 = imag(target_stft); clear target_stft
        
    cost1 = 0.5*sum(sum( (pred_real2 - label_real2).^2))/num_sample;
    cost2 = 0.5*sum(sum( (pred_imag2 - label_imag2).^2))/num_sample;
    
    cost = cost1 + cost2; clear cost1 cost2
    
    Y_r = real(weights);
    %Y_i = imag(weights);
        
    prelim_delta1 = (2*opts.c1*Y_r)./(opts.c2*(opts.c1^2 - pred_real.^2));
    prelim_delta2 = (2*opts.c1*Y_r)./(opts.c2*(opts.c1^2 - pred_imag.^2)); clear Y_r
    output_delta1 = -(label_real2 - pred_real2).*prelim_delta1.*compute_unit_gradient(pred_real,unit_type_output,opts); clear prelim_delta1 pred_real2 label_real2 pred_real
    output_delta2 = -(label_imag2 - pred_imag2).*prelim_delta2.*compute_unit_gradient(pred_imag,unit_type_output,opts); clear prelim_delta2 pred_imag2 label_imag2 pred_imag
    

end


%% backprop
net_gradients = zeroInitNet_mixphone(opts.net_struct,opts.isGPU);

% upper_layer_delta = output_delta;
for ll = num_net_layer: -1: 1
    
    if ll == num_net_layer
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
