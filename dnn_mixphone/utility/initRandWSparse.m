function W = initRandWSparse(nhid, nvis, isSpecNorm, radius)

if nargin < 3;
    isSpecNorm = 0;
end

W = zeros(nhid,nvis);
P = 15; % only P units are set to nonzeros

for i = 1:nhid
    seq = randperm(nvis,P);
    W(i,seq) = randn(1,P,'single');
end

% spectral normalization: useful for RNNs
% if isSpecNorm
%    [~,v] = eig(W);
%    W = W/max(diag(abs(v))); % spec-radius = 1
%    W = radius*W; % set to the specified radius
% end