function mva_data = doARMA(mv_data,order)
% do ARMA smoothing on mean/var normed data
% ARMA is IIR, so it can be sped up

m = size(mv_data,1);
tmp_data = [repmat(mv_data(1,:),order,1); mv_data; repmat(mv_data(m,:),order,1)];
for i = order+1:order+1+m-1
    ss = tmp_data(i,:);
    for k = 1:order
        ss = ss + tmp_data(i-k,:) + tmp_data(i+k,:);
    end    
    tmp_data(i,:) = ss / (2*order+1);
end

mva_data = tmp_data(order+1: order+1+m-1,:);

end
