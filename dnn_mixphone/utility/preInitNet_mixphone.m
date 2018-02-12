function net = preInitNet_mixphone(net_struct,pre_net, isGPU)

num_net_layer = length(net_struct) - 1;
net           = repmat(struct,num_net_layer,1);

for i = 1:num_net_layer    
        
    if i == num_net_layer
        net(i).Wo1 = pre_net(i).W;
        net(i).bo1 = pre_net(i).b;
        net(i).Wo2 = pre_net(i).W;
        net(i).bo2 = pre_net(i).b;
        
        if isGPU
            net(i).Wo1 = gpuArray(net(i).Wo1);
            net(i).Wo2 = gpuArray(net(i).Wo2);
            net(i).bo1 = gpuArray(net(i).bo1);
            net(i).bo2 = gpuArray(net(i).bo2);            
        end        
    else
        net(i).W = pre_net(i).W;
        net(i).b = pre_net(i).b;
        
        if isGPU
            net(i).W = gpuArray(net(i).W);
            net(i).b = gpuArray(net(i).b);
        end
        
    end           
end

%%
% num_net_layer = length(net_struct) - 1;
% net = repmat(struct,num_net_layer,1);
% for i = 1:num_net_layer
% %     net(i).W = initRandW(net_struct(i+1), net_struct(i));
% %     net(i).b = zeros(net_struct(i+1),1);
%     
% 
%     
%     % L2 weight norm normalization
%     if isL2Norm
%         tmp = net(i).W;
%         net(i).W = tmp./repmat(sqrt(sum(tmp.^2,2)), 1, size(tmp,2)); 
%     end
% 
%     net(i).W = single(net(i).W);
%     net(i).b = single(net(i).b);
%      
%     if isGPU
%         net(i).W = gpuArray(net(i).W);
%         net(i).b = gpuArray(net(i).b);
%     end
% end
