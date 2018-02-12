function net_activation = compute_unit_activation(net_potential,unit_type,opts)

if nargin < 3
    opts.split_tanh1_c1 = 1;
    opts.split_tanh1_c2 = 0.5;
end

switch unit_type
    case 'sigm'
        net_activation = sigmoid(net_potential);
    case 'lin'
        net_activation = net_potential;
    case 'relu'
        net_activation = relu(net_potential);
    case 'relum1'
        net_activation = relum1(net_potential);
    case 'tanh1'
%         if nargin < 2
%         c1 = 1;
%         c2 = 2;
%         else
            c1 = opts.split_tanh1_c1;
            c2 = opts.split_tanh1_c2;
%         end
        net_activation = tanh1(net_potential,c1,c2);
    case 'softmax'
        net_activation = softmax(net_potential);
    otherwise
        error(['unknown activation function:' unit_type])
end


% function net_activation = compute_unit_activation(net_potential,unit_type)
% 
% switch unit_type
%     case 'sigm'
%         net_activation = sigmoid(net_potential);
%     case 'lin'
%         net_activation = net_potential;
%     case 'relu'
%         net_activation = relu(net_potential);
%     case 'tanh'
%     case 'softmax'
%         net_activation = softmax(net_potential);
%     otherwise
%         error(['unknown activation function:' unit_type])
% end

