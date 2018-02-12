function [correct, accuracy] = getHResults_triphone_timit(posteriors, opts)
%% writing dnn posteriors as htk features and also the feature_file_list (.scp)
feat_files_dir = opts.htk.feat_files_dir;
if ~exist(feat_files_dir, 'dir'); mkdir(feat_files_dir); end
feat_list_path = [feat_files_dir, 'feat_list.scp'];
hop = opts.htk.hop;

fid = fopen(feat_list_path,'w');
for m = 1:length(posteriors)
    feat = posteriors(m).posteriors;
    if opts.isGPU;feat = gather(feat);end
    feat(isnan(feat)) = 0 ;
    feat(feat<=eps) = eps; feat(feat>=1) =1-eps; % fail safe?
    feat = sqrt(-2*log(feat));
    feat_fname=sprintf('%s/%s.posterior', feat_files_dir, posteriors(m).uid);
    fprintf(fid,'%s\n',feat_fname);
    
    writehtk(feat_fname,feat,hop,9); % 9 means user defined feature
end

fclose(fid);
%% htk commands

hvite_path = opts.htk.hvite_path;
hresults_path = opts.htk.hresults_path;

hmmdef_path = opts.htk.hmmdef_path;
rec_mlf_path = opts.htk.rec_mlf_path;
score_path = opts.htk.score_path;

ref_mlf_path = opts.htk.ref_mlf_path;

lang_model_path = opts.htk.lang_model_path;
dict_path = opts.htk.dict_path;

hmmlist_path = opts.htk.hmmlist_path;

% serial mode
% hvite_opts = opts.htk.hvite_opts;
% hvite_cmd = sprintf('%s %s -H %s -i %s -S %s -w %s %s %s', hvite_path, hvite_opts, hmmdef_path, rec_mlf_path ,...,
%                       feat_list_path, lang_model_path, dict_path, hmmlist_path);
% system(hvite_cmd);

% batch mode, need parfor
run_hvite_split(feat_list_path, 8, opts.htk);

hresults_opts = opts.htk.hresults_opts;
hresults_cmd = sprintf('%s %s -I %s %s %s > %s', hresults_path, hresults_opts,...,
                       ref_mlf_path, hmmlist_path, rec_mlf_path, score_path);
system(hresults_cmd);

%% readout
fid = fopen(score_path,'r');
while ~feof(fid)
    this_line = fgetl(fid);
    if strcmp(this_line(1:5),'WORD:')
        perf = sscanf(this_line,'WORD: %%Corr=%f, Acc=%f');
        correct = perf(1);
        accuracy = perf(2);
        break;
    end
end
fclose(fid);
