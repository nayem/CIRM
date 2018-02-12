function output = predictPhonePosterior(net,cv_struct,opts)

output = repmat(struct,length(cv_struct),1);
for i = 1:length(cv_struct)
% parfor i = 1:length(cv_struct)
    output(i).uid = cv_struct(i).uid;
    output(i).posteriors = getOutputFromNet(net,cv_struct(i).data,opts);
%     output(i).target_id = cv_struct(i).target_id;
end
