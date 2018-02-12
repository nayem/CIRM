function [output1, output2] = getOutputFromNet_mixphone(net,data,opts)

num_net_layer = length(net);
num_data_layer = num_net_layer+1; % count input layer in

if opts.isDropout
    drop_ratio = opts.drop_ratio;
    drop_scale = 1/(1-drop_ratio);
else
    drop_scale = 1;
end

for ii = 1:num_data_layer
    if ii == 1
        net_activation = data';
    else
        if ii == num_data_layer
            net_potential1 = bsxfun(@plus, drop_scale*net(ii-1).Wo1*net_activation, net(ii-1).bo1);
            net_potential2 = bsxfun(@plus, drop_scale*net(ii-1).Wo2*net_activation, net(ii-1).bo2);
            net_activation1 = compute_unit_activation(net_potential1, opts.unit_type_output,opts);
            net_activation2 = compute_unit_activation(net_potential2, opts.unit_type_output,opts);
        else            
            net_potential = bsxfun(@plus, drop_scale*net(ii-1).W*net_activation, net(ii-1).b);
            net_activation = compute_unit_activation(net_potential, opts.unit_type_hidden,opts);
        end
    end
end

output1 = net_activation1';
output2 = net_activation2';
