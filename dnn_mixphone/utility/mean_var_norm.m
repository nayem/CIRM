function [nm mu std] = mean_var_norm(data)
% mean/var norm w.r.t features

% NOTE: when data is a single point, the following still
%       performs mean/var norm, but w.r.t. that single point
%       rather than features. --> change to mean(data,1); var(data,1)
%       esstially the same as mean_var_norm_row


m = size(data,1);
mu = mean(data);
var_d = var(data);

% mean removal
tmp = data-repmat(mu,m,1);
clear data
% variance normalization
nm = tmp ./ repmat(sqrt(var_d),m,1);
std = sqrt(var_d);
