function grad = compute_unit_gradient(net_activation,unit_type,opts)
    switch unit_type
        case 'sigm'            
            grad = net_activation.*(1-net_activation); 
        case 'relu'
            grad = single(net_activation>0);
%             grad = double(net_activation>0);
        case 'relum1'
            grad = single(net_activation>-1);
        case 'softmax'
%             grad = net_activation.*(1-net_activation); % same 
%             error('softmax not supposed to appear as hidden...')
        case 'lin'
            grad = 1;
        case 'tanh1'
            c1 = opts.split_tanh1_c1;
            c2 = opts.split_tanh1_c2;
            Fz = tanh1(net_activation,c1,c2);
            grad = (c2/(2*c1))*(c1^2 - Fz.^2);
        otherwise
            error(['unknown activation function:' unit_type])
    end
end