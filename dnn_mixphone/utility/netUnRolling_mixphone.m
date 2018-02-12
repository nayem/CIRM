function theta = netUnRolling_mixphone(net)

theta = [];
for ll = 1:length(net);
    if ll == length(net)
        theta = [theta; net(ll).Wo1(:); net(ll).bo1(:)];
        theta = [theta; net(ll).Wo2(:); net(ll).bo2(:)];
    else
        theta = [theta; net(ll).W(:); net(ll).b(:)];
    end    
end
