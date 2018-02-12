function net_cpu = gather_net_mixphone(net_gpu)

net_cpu = net_gpu;
for i = 1:length(net_cpu)
    net_cpu(i).W = gather(net_cpu(i).W);
    net_cpu(i).b = gather(net_cpu(i).b);
    net_cpu(i).Wo1 = gather(net_cpu(i).Wo1);
    net_cpu(i).Wo2 = gather(net_cpu(i).Wo2);    
    net_cpu(i).bo1 = gather(net_cpu(i).bo1);
    net_cpu(i).bo2 = gather(net_cpu(i).bo2);
end
