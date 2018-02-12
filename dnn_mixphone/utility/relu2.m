function y = relu2(x)   
   mask = single(x>0);
   y = x.*mask;
end
