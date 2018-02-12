function [nm] = mean_var_norm_testing(data,mu,std)
% mean removal
tmp = data-repmat(mu,size(data,1),1);
% variance normalization
nm = tmp ./ repmat(std,size(data,1),1);

nm(isnan(nm)) = 0; % if std=0