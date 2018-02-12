function y = removeBadElement(x,bad_elem,replace_elem)
if nargin<3
    replace_elem = 0;
end

% mask = (x~=bad_elem);
% y = x.*mask; % much faster than y(isinf(y))=0; but doesnt work as inf*0=nan, need a MEX perhaps

%slow workaround
mask=(x==bad_elem);
if sum(sum(mask))>0    
    x(x==bad_elem)=replace_elem;
    y = x;
else
    y = x;
end
