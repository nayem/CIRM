%%
clear all;
%%
opts.cost_function = 'softmax_xentropy';

% opts.hid_struct = [20,18,50]; % num of hid layers and units
opts.hid_struct = [20, 18, 30];

opts.unit_type_output = 'softmax';
opts.unit_type_hidden = 'sigm';
% opts.unit_type_hidden = 'relu';

if strcmp(opts.unit_type_output,'softmax'); opts.cost_function = 'softmax_xentropy'; end;

opts.isDropout = 0; % need dropout regularization?
opts.drop_ratio = 0.2; % ratio of units to drop

opts.isGPU = 0; % use GPU?
%%
num_samples = 50;
dim_input = 16;
dim_output1 = 12;
dim_output2 = 15;
data = randn(num_samples,dim_input);

label1 = zeros(num_samples,dim_output1);
for i = 1:size(label1,1)
    k = randi(size(label1,2));
    label1(i,k) = 1;
end

label2 = zeros(num_samples,dim_output2);
for i = 1:size(label2,1)
    k = randi(size(label2,2));
    label2(i,k) = 1;
end

[m, d_in] = size(data);
d_out1 = size(label1,2);
d_out2 = size(label2,2);
d_out = {d_out1, d_out2};

opts.net_struct = {d_in};
for i = 1:length(opts.hid_struct)
    opts.net_struct{end+1} = opts.hid_struct(i);
end
opts.net_struct{end+1} = d_out;

opts

pre_net = randInitNet_mixphone(opts.net_struct, opts.isGPU);
alltheta = netUnRolling_mixphone(pre_net);
%% note: use double for gradient checking
%computeNetGradient_mixphone(theta, data, label1, label2, opts)
alltheta = double(alltheta);
[cost, grad] = computeNetGradient_mixphone(alltheta,data,label1,label2,opts);
numgrad = computeNumericalGradient(@(p)computeNetGradient_mixphone(p,data,label1,label2,opts), alltheta);

[grad numgrad]
diff = norm(numgrad-grad)/norm(numgrad+grad)
clf;plot(grad); hold on; plot(numgrad,'r');
