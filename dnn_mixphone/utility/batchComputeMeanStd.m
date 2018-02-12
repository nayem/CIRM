function [mean_vec, std_vec] = batchComputeMeanStd(data, batch_size)
if nargin < 2
    batch_size = 1024*2;
end

% mem consuming?
mean_vec = mean(data);

[m, dim] = size(data);

batch_id = genBatchID(m, batch_size);

ss = 0;
for bid = 1:length(batch_id)    
    range = batch_id(1,bid):batch_id(2,bid);
%     kk = mmap_loadPartial(mmap, range);
    chunck = data(range, :);
    ss = ss + sum((bsxfun(@minus, chunck, mean_vec)).^2);
end

std_vec = sqrt(ss/(m-1));
%%
% mmap = create_mmap('~/test.bin',dim,'single');
% kk = mmap_loadPartial(mmap,1:m);
% 
% batch_id = genBatchID(m, 100);
% 
% ss = 0;
% for bid = 1:length(batch_id)
%     range = batch_id(1,bid):batch_id(2,bid);
%     kk = mmap_loadPartial(mmap, range);
%     ss = ss + (bsxfun(@minus, kk, mean_vec)).^2;
% end
% 
% my_var_vec = sum(ss)/(m-1);
% 
% [my_var_vec' var_vec']
% norm(my_var_vec-var_vec)