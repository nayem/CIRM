function ret = makeWindowFeat2_fix(data, side)
% the difference is that the current frame is centered.

if side ~= 0
    [m, d] = size(data);
    % ret = [data];
    ret = [];
    
    % left hand
%     for i = 1:side
    for i = side:-1:1
        %    tmp = data;
        %    tmp = [repmat(tmp(1,:),i,1); tmp];
        
        tmp = [repmat(data(1,:),i,1); data];
        tmp = tmp(1:m,:);
        ret = [ret tmp];
    end
    
%     ret = [fliplr(ret) data];
    ret = [ret data];
    
    % right hand
    for i = 1:side
        %    tmp = data;
        %    cp = tmp(m,:);
        %    tmp = tmp(i+1:m,:);
        
        
        cp = data(m,:);
        tmp = data(i+1:m,:);
        tmp = [tmp;repmat(cp,i,1)];
        ret = [ret tmp];
    end
else
    ret = data;
end