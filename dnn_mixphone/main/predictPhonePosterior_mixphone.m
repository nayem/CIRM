function output = predictPhonePosterior_mixphone(net,cv_struct,opts)

output = repmat(struct,length(cv_struct),1);
for i = 1:length(cv_struct)
    output(i).uid = cv_struct(i).uid;
    [o1, o2] = getOutputFromNet_mixphone(net, cv_struct(i).data,opts);
    output(i).posteriors = cell(2,1);
    output(i).posteriors{1} = o1;
    output(i).posteriors{2} = o2;
end
