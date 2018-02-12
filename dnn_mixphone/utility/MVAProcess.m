function [mva_data mu std] = MVAProcess(data,order, isLowPass)
% per-utterance level normalization

[m d] = size(data);
% tmp_data = zeros(m+2*order,d);

if order ~= -1
    [mv_data mu std] = mean_var_norm(data);    
%     [mv_data mu std] = mean_var_norm_l2(data);    
%     [mv_data mu std] = mean_var_norm_l2(data,[15 54 85 100 139 170]);    
    if order ~= 0
        tmp_data = [repmat(mv_data(1,:),order,1); mv_data; repmat(mv_data(m,:),order,1)];
        for i = order+1:order+1+m-1
            ss = tmp_data(i,:);
            for k = 1:order
                ss = ss + tmp_data(i-k,:) + tmp_data(i+k,:);
            end
            
            tmp_data(i,:) = ss / (2*order+1);
        end
        
        mva_data = tmp_data(order+1: order+1+m-1,:);
    else
        mva_data = mv_data;
    end
else
    mva_data = data;
end

if isLowPass
    a = 0.3; % smaller -> smoother
    mva_data =  filter(a, [1 a-1], mva_data);
end
