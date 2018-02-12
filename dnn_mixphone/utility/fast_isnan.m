%
% nn = gpuArray.randn(10000,1000,1,'single');
% nn = randn(10000,1000,1,'single');

% nn = gpuArray.randn(10000000,1,1,'single');
% nn = randn(10000000,1,1,'single');

% tic
% for i=1:100; nn(isinf(nn))=0;end
% toc

%%
% xx = gpuArray.randn(1000,2000,'single');
xx = gpuArray.randn(2000000,1,'single');
% xx = randn(2000000,1,'single');
% xx = randn(1000,1000,'single');
% xx(end)=inf;
tic
disp(' ');
for i = 1:1000
%     y = relu(xx);

%     y = relu2(xx);
%     y = sigmoid(xx);

%       y = compute_unit_gradient(xx,'sigm');
%       y = compute_unit_gradient(xx,'relu');

    y = removeBadElement(xx,inf);
%     y(isnan(y)) = 0;
end
toc