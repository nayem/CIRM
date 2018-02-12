function net = zeroInitNet_mixphone(net_struct, isGPU, epsilon)

if nargin<3; epsilon = 0; end;

num_net_layer = length(net_struct) - 1;
net           = repmat(struct,num_net_layer,1);

for i = 1:num_net_layer   
    
    nIn = net_struct{i};
    
    if i == num_net_layer
        nOut1 = net_struct{i+1}{1};
        nOut2 = net_struct{i+1}{2};
    else
        nHid = net_struct{i+1};
    end
    
    if isGPU
        if i == num_net_layer
            net(i).Wo1 = gpuArray.zeros(nOut1, nIn ,'single');
            net(i).bo1 = gpuArray.zeros(nOut1, 1, 'single');            
            net(i).Wo2 = gpuArray.zeros(nOut2, nIn ,'single');
            net(i).bo2 = gpuArray.zeros(nOut2, 1, 'single');
        else
            net(i).W = gpuArray.zeros(nHid, nIn ,'single');
            net(i).b = gpuArray.zeros(nHid, 1, 'single');        
        end
    else
        if i == num_net_layer            
            net(i).Wo1 = zeros(nOut1, nIn ,'single');
            net(i).bo1 = zeros(nOut1, 1, 'single');            
            net(i).Wo2 = zeros(nOut2, nIn ,'single');
            net(i).bo2 = zeros(nOut2, 1, 'single');        
        else
            net(i).W = zeros(nHid, nIn ,'single');
            net(i).b = zeros(nHid, 1, 'single');        
        end        
    end
    
    % this is for initializing net_grad_ssqr in ada_sgd
    if epsilon ~=0
        if i == num_net_layer
            net(i).Wo1 = net(i).Wo1 + epsilon;
            net(i).bo1 = net(i).bo1 + epsilon;
            net(i).Wo2 = net(i).Wo2 + epsilon;
            net(i).bo2 = net(i).bo2 + epsilon;
        else
            net(i).W = net(i).W + epsilon;
            net(i).b = net(i).b + epsilon;
        end
    end
end
