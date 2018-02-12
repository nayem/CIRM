function [net,rest] = netRolling_mixphone(theta, net_struct)

num_net_layer = length(net_struct) - 1;
net = repmat(struct,num_net_layer,1);
pos = 1;
for ii = 1: num_net_layer
    vis = net_struct{ii};
    if ii == num_net_layer
        hid1 = net_struct{ii+1}{1};
        hid2 = net_struct{ii+1}{2};
    
        net(ii).Wo1 = reshape(theta(pos:pos+vis*hid1-1), hid1, vis);
        pos = pos+vis*hid1;        
        net(ii).bo1 = theta(pos:pos+hid1-1);
        pos = pos+hid1;
        
        net(ii).Wo2 = reshape(theta(pos:pos+vis*hid2-1), hid2, vis);
        pos = pos+vis*hid2;
        net(ii).bo2 = theta(pos:pos+hid2-1);
        pos = pos+hid2;                
    else
        hid = net_struct{ii+1};
        net(ii).W = reshape(theta(pos:pos+vis*hid-1), hid, vis);
        pos = pos+vis*hid;
    
        net(ii).b = theta(pos:pos+hid-1);
        pos = pos+hid;           
    end
   
end
rest = theta(pos:end);

assert(isempty(rest))
