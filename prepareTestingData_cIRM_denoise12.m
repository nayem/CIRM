function [teData, teLabel_r,teLabel_i, para, numframes] = prepareTestingData_cIRM_denoise12(globalpara,cs_testing_data_path,testing_mix_wav_save_path)

% Description: Prepare Testing features for LSTM. 
%       -Save uncompressed labels
%       -Save compressed labels
%       -Save spectrograms clean+mixture
%
% Input:
%   - globalpara: global parameters
%   - cs_testing_data_path: Clean signal path for Testing
%   - testing_mix_wav_save_path: Noisy signal path for Testing
%
% Output
%   - teData: Feature data for all files appended one after another
%   - teLabel_r: Testing label real value
%   - teLabel_i: Testing label imaginary value
%   - para: Parameters
%   - numframes: Number of frames for each file
%
% Author: Khandokar Md. Nayem, May 29, 2018


%% Load training data and generate features

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
%% Process Reverberant/Clean Testing Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

files     = dir(fullfile(testing_mix_wav_save_path,'*.wav'));
num_files = length(files);
numframes = zeros(num_files,1);

%%=============== Files for statestics Analysis ============%%
SERVER = 'Eagles';

if strcmpi(SERVER,'Eagles') == 1
    path_data_directory = '/data/knayem';
    path_code_directory = '/home/knayem/EaglesBigred2/cIRM';
end

STATS_FILE = fullfile(path_code_directory,'dnn_models/Test_datas.mat');

% snr_exp = '(-)?(\d)+dB';
%%===========================================================%%

fprintf('\tComputing number of frames...\n\t\t')
ten_percent = ceil(0.1*num_files);
for fileNum = 1:num_files

    filename      = files(fileNum).name;
    MIX_FILENAME = strcat(testing_mix_wav_save_path,filename);
    mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz

    numframes(fileNum) = fix((length(mix_sig) - overlap)/(winlen-overlap));

    if(~mod(fileNum,ten_percent))
        fprintf('%d...',(fileNum/ten_percent)*10)
    end
end

% ------------------------ Initialize data -------------------------------%

prelimFeatData = zeros(sum(numframes),246,'single');
teLabel_r   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');
teLabel_i   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');
teFilename  = strings(sum(numframes),1);

% nayem ------------------------------------------------------------------%
% spectrogram
spec_clean_r = zeros(sum(numframes),(nfft/2 + 1)*labwin,'single');
spec_clean_i = zeros(sum(numframes),(nfft/2 + 1)*labwin,'single');
spec_mixture_r = zeros(sum(numframes),(nfft/2 + 1)*labwin,'single');
spec_mixture_i = zeros(sum(numframes),(nfft/2 + 1)*labwin,'single');
cleanFilename  = strings(sum(numframes),1);

% uncompressed labels
uncomp_mixture_label_r   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');
uncomp_mixture_label_i   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');
% compressed labels = teLabel_r,teLabel_i
%-------------------------------------------------------------------------%

start_samp = 1;

fprintf('\n\tExtracting features and labels...\n\t\t')

for fileNum = 1:num_files%num_files

    filename      = files(fileNum).name;
    MIX_FILENAME = strcat(testing_mix_wav_save_path,filename);
    mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz

    clean_filename = filename(1:11);
    CLEAN_FILENAME = sprintf('%s%s.wav',cs_testing_data_path,clean_filename);
    clean_sig      = audioread(CLEAN_FILENAME);
    
    teFilename(fileNum) = filename;
    cleanFilename(fileNum) = clean_filename;

    stop_samp = start_samp + numframes(fileNum) - 1;

    % ----------------------- Compute Features --------------------------%
    feats = get_compRP2d_mkcomp2(mix_sig.',globalpara).';
    prelimFeatData(start_samp:stop_samp,:) = feats.';

    % ----------------------- Compute Labels ----------------------------%
    mix_stft   = spectrogram(mix_sig,winlen,overlap,nfft,Fs);
    clean_stft = spectrogram(clean_sig,winlen,overlap,nfft,Fs);
    

    [labels_r, labels_i, uncomp_labels_r, uncomp_labels_i] = cIRM12(clean_stft,mix_stft, cliplevel,labcompress,globalpara.logistic_max,globalpara.logistic_steep);

    if labwin > 0
        labels_r = makeWindowFeat3(labels_r,labwin).';
        labels_i = makeWindowFeat3(labels_i,labwin).';
        % nayem -> windowed uncompressed labels
        uncomp_labels_r = makeWindowFeat3(uncomp_labels_r,labwin).';
        uncomp_labels_i = makeWindowFeat3(uncomp_labels_i,labwin).';
    end

    teLabel_r(start_samp:stop_samp,:) = labels_r;
    teLabel_i(start_samp:stop_samp,:) = labels_i;
    
    % ======================== Store States ============================ %
    spec_clean_r(start_samp:stop_samp,:) = real(clean_stft)';
    spec_clean_i(start_samp:stop_samp,:) = imag(clean_stft)';
    spec_mixture_r(start_samp:stop_samp,:) = real(mix_stft)';
    spec_mixture_i(start_samp:stop_samp,:) = imag(mix_stft)';

    % labels
    uncomp_mixture_label_r(start_samp:stop_samp,:) = uncomp_labels_r;
    uncomp_mixture_label_i(start_samp:stop_samp,:) = uncomp_labels_i;

    % ----------------------- Update counters ---------------------------%
    start_samp = stop_samp + 1;

    if(~mod(fileNum,ten_percent))
        fprintf('%d...',(fileNum/ten_percent)*10)
    end
end

clearvars -except STATS_FILE para prelimFeatData arma_order feawin teData teLabel_r teLabel_i numframes spec_clean_r spec_clean_i spec_mixture_r spec_mixture_i uncomp_mixture_label_r uncomp_mixture_label_i teFilename cleanFilename


fprintf('\n\tNormalizing training data...\n')

[prelimFeatData, para.te_mu, para.te_std] = meanVarArmaNormalize(prelimFeatData, arma_order);
prelimFeatData(isnan(prelimFeatData)) = 0;
teData = makeWindowFeat4(prelimFeatData.',feawin);

save(STATS_FILE, 'teData','teLabel_r','teLabel_i','numframes','teFilename','cleanFilename','spec_clean_r','spec_clean_i','spec_mixture_r','spec_mixture_i','uncomp_mixture_label_r','uncomp_mixture_label_i', '-v7.3');
end
