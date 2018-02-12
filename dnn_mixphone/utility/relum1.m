function y = relum1(x)
   
% slow version
%    y = x;
%    y(y<=0) = 0;

% faster version: make use of matrix opts
   mask = single(x>-1);
   y = x.*mask;
end
