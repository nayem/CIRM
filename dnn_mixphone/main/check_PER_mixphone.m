function [perf, perf_str, output] = check_PER_mixphone(net, cv_struct, use_which, opts)

output = predictPhonePosterior_mixphone(net,cv_struct,opts);
[~, ph_accuracy] = getHResults_mixphone_timit(output, use_which, opts);
perf = ph_accuracy;

if use_which == 1
    perf_str = 'Phone_Acc_Monophone';
elseif use_which == 2
    perf_str = 'Phone_Acc_Triphone';
end
    
