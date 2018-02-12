function run_hvite_split(feat_list_path, num_split, htk_opts)
%%
fid = fopen(feat_list_path,'r');
all_lines = textscan(fid, '%s', 'delimiter', '\n', 'whitespace', '', 'bufSize', 16000);
all_lines = all_lines{1};
fclose(fid);
num_files = length(all_lines);
%%
% batch_size = num_files / num_split;
% assert(round(batch_size) == batch_size)
num_b = ceil(num_files/num_split);
batch = genBatchID(num_files, num_b); 

feat_files_dir = htk_opts.feat_files_dir;
% for i = 1:num_split
for i = 1:size(batch,2);
    split_scp_path = [feat_files_dir, 'feat_list_split', num2str(i), '.scp'];
%     range = (i-1)*batch_size+1:i*batch_size;
    range = batch(1,i):batch(2,i);
    
    fid = fopen(split_scp_path,'w');
    for j = 1:length(range)
        fprintf(fid, '%s\n',all_lines{range(j)});
    end
    fclose(fid);    
end
%% batch decoding
parfor i = 1:size(batch,2);
    rec_mlf_path = [feat_files_dir, '/dnn_recout_split', num2str(i), '.mlf'];
    feat_list_path = [feat_files_dir, 'feat_list_split', num2str(i), '.scp'];
    hvite_cmd = sprintf('%s %s -H %s -i %s -S %s -w %s %s %s', htk_opts.hvite_path, htk_opts.hvite_opts, htk_opts.hmmdef_path,...,
                rec_mlf_path , feat_list_path, htk_opts.lang_model_path, htk_opts.dict_path, htk_opts.hmmlist_path);
    system(hvite_cmd);
end
%% merge MLFs and decoding
all_mlf_lines = {};
for i = 1:size(batch,2);
    rec_mlf_path = [feat_files_dir, '/dnn_recout_split', num2str(i), '.mlf'];
    fid = fopen(rec_mlf_path,'r');
    t_lines = textscan(fid, '%s', 'delimiter', '\n', 'whitespace', '', 'bufSize', 16000);
    t_lines = t_lines{1};
    fclose(fid);
    t_lines = t_lines(2:end);
    all_mlf_lines = [all_mlf_lines; t_lines];
end
all_mlf_lines = ['#!MLF!#'; all_mlf_lines];

fid = fopen(htk_opts.rec_mlf_path,'w');
for i = 1:length(all_mlf_lines)
    fprintf(fid, '%s\n',all_mlf_lines{i});
end
% fprintf(fid, '%s','.');
fclose(fid);

