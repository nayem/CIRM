function [cvData, cvLabel_r,cvLabel_i, para, numframes] = prepareDevData_cIRM_denoise41(STATS_FILE,globalpara,cs_dev_data_path,noisy_dev_data_path)

% Description: Prepare Cross Validation features for LSTM. 
%       -Save uncompressed labels
%       -Save compressed labels
%       -Save spectrograms clean+mixture
%
% Input:
%   - globalpara: global parameters
%   - cs_dev_data_path: Clean signal path for Cross Validation
%   - noisy_dev_data_path: Noisy signal path for Cross Validation
%
% Output
%   - cvData: Feature data for all files appended one after another
%   - cvLabel_r: Cross Validation label real value
%   - cvLabel_i: Cross Validation label imaginary value
%   - para: Parameters
%   - numframes: Number of frames for each file
%
% Author: Khandokar Md. Nayem, May 29, 2018


%% Load cross validation data and generate features

feawin      = globalpara.feawin;
labwin      = globalpara.labwin;
winlen      = globalpara.winlen;
overlap     = globalpara.overlap;
nfft        = globalpara.nfft;
Fs          = globalpara.Fs;
cliplevel   = globalpara.clip_level;
arma_order  = globalpara.arma_order;
labcompress = globalpara.labcompress;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Process Reverberant/Clean Cross Validation Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

files     = dir(fullfile(noisy_dev_data_path,'*.wav'));
num_files = length(files);
numframes = zeros(num_files,1);

%%=============== Files for statestics Analysis ============%%
SERVER = 'Eagles';

if strcmpi(SERVER,'Eagles') == 1
    path_data_directory = '/data/knayem';
    path_code_directory = '/home/knayem/EaglesBigred2/cIRM';
end

% comment>>>
% STATS_FILE = fullfile(path_code_directory,'dnn_models/CrossValidation_datas.mat');

% comment>>>
% snr_exp = '(-)?(\d)+dB';
%%===========================================================%%

fprintf('\tComputing number of frames...\n\t\t')
ten_percent = ceil(0.1*num_files);
for fileNum = 1:num_files

    filename      = files(fileNum).name;
    MIX_FILENAME = strcat(noisy_dev_data_path,filename);
    mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz

    numframes(fileNum) = fix((length(mix_sig) - overlap)/(winlen-overlap));

    if(~mod(fileNum,ten_percent))
        fprintf('%d...',(fileNum/ten_percent)*10)
    end
end

% ------------------------ Initialize data -------------------------------%
cvData=0; cvLabel_r=0; cvLabel_i=0; para=0;
% prelimFeatData = zeros(sum(numframes),246,'single');
% cvLabel_r   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');
% cvLabel_i   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');
cvFilename  = strings(num_files,1);

% nayem ------------------------------------------------------------------%
% spectrogram
spec_clean_r = zeros(sum(numframes),(nfft/2 + 1)*labwin,'single');
spec_clean_i = zeros(sum(numframes),(nfft/2 + 1)*labwin,'single');
spec_mixture_r = zeros(sum(numframes),(nfft/2 + 1)*labwin,'single');
spec_mixture_i = zeros(sum(numframes),(nfft/2 + 1)*labwin,'single');
cleanFilename  = strings(num_files,1);

% uncompressed labels
% uncomp_mixture_label_r   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');
% uncomp_mixture_label_i   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');

% comment>>>
% compressed labels = cvLabel_r,cvLabel_i
%-------------------------------------------------------------------------%

start_samp = 1;

fprintf('\n\tExtracting features and labels...\n\t\t')

for fileNum = 1:num_files%num_files

    filename      = files(fileNum).name;
    MIX_FILENAME = strcat(noisy_dev_data_path,filename);
    mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz

    clean_filename = filename(1:11);
    CLEAN_FILENAME = sprintf('%s%s.wav',cs_dev_data_path,clean_filename);
    clean_sig      = audioread(CLEAN_FILENAME);

    cvFilename(fileNum) = filename;
    cleanFilename(fileNum) = clean_filename;
    
    stop_samp = start_samp + numframes(fileNum) - 1;

    % ----------------------- Compute Features --------------------------%
%     feats = get_compRP2d_mkcomp2(mix_sig.',globalpara).';
%     prelimFeatData(start_samp:stop_samp,:) = feats.';

    % ----------------------- Compute Labels ----------------------------%
    mix_stft   = spectrogram(mix_sig,winlen,overlap,nfft,Fs);
    clean_stft = spectrogram(clean_sig,winlen,overlap,nfft,Fs);
    

%     [labels_r, labels_i, uncomp_labels_r, uncomp_labels_i] = cIRM12(clean_stft,mix_stft, cliplevel,labcompress,globalpara.logistic_max,globalpara.logistic_steep);
% 
%     if labwin > 0
%         labels_r = makeWindowFeat3(labels_r,labwin).';
%         labels_i = makeWindowFeat3(labels_i,labwin).';
%         % nayem -> windowed uncompressed labels
%         uncomp_labels_r = makeWindowFeat3(uncomp_labels_r,labwin).';
%         uncomp_labels_i = makeWindowFeat3(uncomp_labels_i,labwin).';
%     end
% 
%     cvLabel_r(start_samp:stop_samp,:) = labels_r;
%     cvLabel_i(start_samp:stop_samp,:) = labels_i;
    
    % ======================== Store States ============================ %
    spec_clean_r(start_samp:stop_samp,:) = real(clean_stft)';
    spec_clean_i(start_samp:stop_samp,:) = imag(clean_stft)';
    spec_mixture_r(start_samp:stop_samp,:) = real(mix_stft)';
    spec_mixture_i(start_samp:stop_samp,:) = imag(mix_stft)';

    % labels
%     uncomp_mixture_label_r(start_samp:stop_samp,:) = uncomp_labels_r;
%     uncomp_mixture_label_i(start_samp:stop_samp,:) = uncomp_labels_i;

    % ----------------------- Update counters ---------------------------%
    start_samp = stop_samp + 1;

    if(~mod(fileNum,ten_percent))
        fprintf('%d...',(fileNum/ten_percent)*10)
    end
end

% clearvars -except STATS_FILE para prelimFeatData arma_order feawin cvData cvLabel_r cvLabel_i numframes spec_clean_r spec_clean_i spec_mixture_r spec_mixture_i uncomp_mixture_label_r uncomp_mixture_label_i cvFilename cleanFilename
clearvars -except STATS_FILE para arma_order feawin cvData cvLabel_r cvLabel_i numframes spec_clean_r spec_clean_i spec_mixture_r spec_mixture_i cvFilename cleanFilename


fprintf('\n\tNormalizing cross validation data...\n')

% [prelimFeatData, para.cv_mu, para.cv_std] = meanVarArmaNormalize(prelimFeatData, arma_order);
% prelimFeatData(isnan(prelimFeatData)) = 0;
% cvData = makeWindowFeat4(prelimFeatData.',feawin);


if exist(STATS_FILE, 'file')
    fprintf('Path-> %s exists!\n',STATS_FILE );
%     cvData_t = cvData;  
%     cvLabel_r_t = cvLabel_r;    
%     cvLabel_i_t = cvLabel_i;
    numframes_t = numframes;    
    cvFilename_t = cvFilename;
    cleanFilename_t = cleanFilename;    
    spec_clean_r_t = spec_clean_r;  
    spec_clean_i_t = spec_clean_i;
    spec_mixture_r_t = spec_mixture_r;  
    spec_mixture_i_t = spec_mixture_i;
%     uncomp_mixture_label_r_t = uncomp_mixture_label_r;  
%     uncomp_mixture_label_i_t = uncomp_mixture_label_i;
    
%     load(STATS_FILE, 'cvData','cvLabel_r','cvLabel_i','numframes','cvFilename','cleanFilename','spec_clean_r','spec_clean_i','spec_mixture_r','spec_mixture_i','uncomp_mixture_label_r','uncomp_mixture_label_i');
    load(STATS_FILE,'numframes','cvFilename','cleanFilename','spec_clean_r','spec_clean_i','spec_mixture_r','spec_mixture_i');
    
%     cvData = [cvData;cvData_t];  
%     cvLabel_r = [cvLabel_r;cvLabel_r_t];    
%     cvLabel_i = [cvLabel_i;cvLabel_i_t];
    numframes = [numframes;numframes_t];    
    cvFilename = [cvFilename;cvFilename_t];
    cleanFilename = [cleanFilename;cleanFilename_t];    
    spec_clean_r = [spec_clean_r;spec_clean_r_t];  
    spec_clean_i = [spec_clean_i;spec_clean_i_t];
    spec_mixture_r = [spec_mixture_r;spec_mixture_r_t];  
    spec_mixture_i = [spec_mixture_i;spec_mixture_i_t];
%     uncomp_mixture_label_r = [uncomp_mixture_label_r;uncomp_mixture_label_r_t];  
%     uncomp_mixture_label_i = [uncomp_mixture_label_i;uncomp_mixture_label_i_t];
    
else
    fprintf('Path-> %s NOT exists!\n',STATS_FILE);
end



% save(STATS_FILE, 'cvData','cvLabel_r','cvLabel_i','numframes','cvFilename','cleanFilename','spec_clean_r','spec_clean_i','spec_mixture_r','spec_mixture_i','uncomp_mixture_label_r','uncomp_mixture_label_i', '-v7.3');
save(STATS_FILE, 'numframes','cvFilename','cleanFilename','spec_clean_r','spec_clean_i','spec_mixture_r','spec_mixture_i', '-v7.3');


end
