function forward_path = forwardPass_mixphone(net,data,opts)

num_net_layer = length(net);
num_data_layer = num_net_layer+1; % count input layer in

forward_path = cell(num_data_layer,1);

if opts.isDropout
    drop_ratio = opts.drop_ratio;
    drop_scale = 1/(1-drop_ratio);
else
    drop_scale = 1;
end

for ii = 1:num_data_layer
    if ii == 1
        net_activation = data';
        drop_flag = opts.isDropout && opts.isDropoutInput; %sometimes it's better not to drop inputs
    else
        drop_flag = opts.isDropout && (ii<=num_net_layer); % never dropout outputs
        
        if ii == num_data_layer
            net_potential_1 = bsxfun(@plus, drop_scale*net(ii-1).Wo1*net_activation, net(ii-1).bo1);
            net_activation_1 = compute_unit_activation(net_potential_1,opts.unit_type_output,opts);
            
            net_potential_2 = bsxfun(@plus, drop_scale*net(ii-1).Wo2*net_activation, net(ii-1).bo2);
            net_activation_2 = compute_unit_activation(net_potential_2,opts.unit_type_output,opts);
            
            net_activation = {net_activation_1, net_activation_2};
        else
            net_potential = bsxfun(@plus, drop_scale*net(ii-1).W*net_activation, net(ii-1).b);          
            net_activation = compute_unit_activation(net_potential,opts.unit_type_hidden,opts);
        end
    end
    
    if drop_flag
        if opts.isGPU
            drop_mask = gpuArray.rand(size(net_activation),'single')<(1-drop_ratio); %faster than rand on CPU
        else
            drop_mask = rand(size(net_activation),'single')<(1-drop_ratio);
        end        
        net_activation = net_activation.*drop_mask; % drop hid units, for *each* data
    end
   
    forward_path{ii} = net_activation;
end
