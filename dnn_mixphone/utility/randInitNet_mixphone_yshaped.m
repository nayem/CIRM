function net = randInitNet_mixphone_yshaped(net_struct, isGPU, num_yshaped_layers)

num_net_layer = length(net_struct) - 1;
net           = repmat(struct,num_net_layer,1);

yshaped_layers = zeros(num_net_layer,1);
yshaped_count  = num_net_layer;

for lay = num_yshaped_layers:-1:1
    yshaped_layers(yshaped_count) = yshaped_count;
    yshaped_count = yshaped_count - 1;
end

for i = 1:num_net_layer    
    
    nIn = net_struct{i};
    
    if i == yshaped_layers(i)
        nOut1 = net_struct{i+1}{1};
        nOut2 = net_struct{i+1}{2};
    else
        nHid = net_struct{i+1};
    end
    
    if i == yshaped_layers(i)
        net(i).Wo1 = initRandW_simple(nOut1, nIn);
        net(i).bo1 = initRandW_simple(nOut1, 1);
        net(i).Wo2 = zeros(nOut2, nIn ,'single');
        net(i).bo2 = zeros(nOut2, 1, 'single');
        
        if isGPU
            net(i).Wo1 = gpuArray(net(i).Wo1);
            net(i).Wo2 = gpuArray(net(i).Wo2);
            net(i).bo1 = gpuArray(net(i).bo1);
            net(i).bo2 = gpuArray(net(i).bo2);            
        end        
    else
        net(i).W = initRandW_simple(nHid, nIn);
        net(i).b = zeros(nHid, 1, 'single');
        
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
