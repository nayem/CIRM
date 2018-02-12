function str = getNetParamStr_mixphone(opts)

str = '';

for i = 1:length(opts.net_struct)-1
    str = [str,num2str(opts.net_struct{i}),'.'];
end

d_out = opts.net_struct{end};
str = [str, num2str(d_out{1}), '+', num2str(d_out{2}), '.'];

str = [str, opts.unit_type_hidden];
str = [str,'.',opts.learner];
if opts.isDropout
    str = [str,'.dropout'];
else
    str = [str,'.no_dropout'];
end