function y = softmax(x)
%     tmp = repmat(sum(exp(x)),size(x,1),1);
%     y = exp(x)./tmp;              
%       y = bsxfun(@rdivide,exp(x),sum(exp(x)));

      exp_x = exp(x);
      y = bsxfun(@rdivide, exp_x, sum(exp_x));
end
