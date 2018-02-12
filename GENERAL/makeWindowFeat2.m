function ret = makeWindowFeat2(data, side)
% the difference is that the current frame is centered.


[m d] = size(data);
% ret = [data];
ret = [];

% left hand
for i = 1:side
%    tmp = data;
%    tmp = [repmat(tmp(1,:),i,1); tmp];
   
   tmp = [repmat(data(1,:),i,1); data];
   tmp = tmp(1:m,:);
   ret = [ret tmp];    
end

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
