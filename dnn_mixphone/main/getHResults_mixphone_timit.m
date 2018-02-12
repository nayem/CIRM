function [correct, accuracy] = getHResults_mixphone_timit(posteriors, use_which, opts)
htk = opts.htk{use_which};
%% writing dnn posteriors as htk features and also the feature_file_list (.scp)
feat_files_dir = htk.feat_files_dir;
if ~exist(feat_files_dir, 'dir'); mkdir(feat_files_dir); end
feat_list_path = [feat_files_dir, 'feat_list.scp'];
hop = htk.hop;

fid = fopen(feat_list_path,'w');
for m = 1:length(posteriors)
    feat = posteriors(m).posteriors{use_which};
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

hvite_path = htk.hvite_path;
hresults_path = htk.hresults_path;

hmmdef_path = htk.hmmdef_path;
rec_mlf_path = htk.rec_mlf_path;
score_path = htk.score_path;

ref_mlf_path = htk.ref_mlf_path;

lang_model_path = htk.lang_model_path;
dict_path = htk.dict_path;

hmmlist_path = htk.hmmlist_path;

% serial mode
% hvite_opts = htk.hvite_opts;
% hvite_cmd = sprintf('%s %s -H %s -i %s -S %s -w %s %s %s', hvite_path, hvite_opts, hmmdef_path, rec_mlf_path ,...,
%                       feat_list_path, lang_model_path, dict_path, hmmlist_path);
% system(hvite_cmd);

% batch mode, need parfor
run_hvite_split(feat_list_path, 8, htk);

hresults_opts = htk.hresults_opts;
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
