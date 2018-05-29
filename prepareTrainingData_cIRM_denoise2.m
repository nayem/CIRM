function [trData, trLabel_r,trLabel_i, para, numframes] = prepareTrainingData_cIRM_denoise2(globalpara,cs_training_data_path,training_mix_wav_save_path)

% Description: Prepare Training features for LSTM
%
% Input:
%   - globalpara: global parameters
%   - cs_training_data_path: Clean signal path for Training
%   - training_mix_wav_save_path: Noisy signal path for Training
%
% Output
%   - trData: Feature data for all files appended one after another
%   - trLabel_r: Training label real value
%   - trLabel_i: Training label imaginary value
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
%% Process Reverberant/Clean Training Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

files     = dir(fullfile(training_mix_wav_save_path,'*.wav'));
num_files = length(files);
% num_files = 5;
numframes = zeros(num_files,1);

fprintf('\tComputing number of frames...\n\t\t')
ten_percent = ceil(0.1*num_files);
for fileNum = 1:num_files

    filename      = files(fileNum).name;
    MIX_FILENAME = strcat(training_mix_wav_save_path,filename);
    mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz

    numframes(fileNum) = fix((length(mix_sig) - overlap)/(winlen-overlap));

    if(~mod(fileNum,ten_percent))
        fprintf('%d...',(fileNum/ten_percent)*10)
    end
end

% ------------------------ Initialize data -------------------------------%

prelimFeatData = zeros(sum(numframes),246,'single');
trLabel_r   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');
trLabel_i   = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');

start_samp = 1;

fprintf('\n\tExtracting features and labels...\n\t\t')

for fileNum = 1:num_files%num_files

    filename      = files(fileNum).name;
    MIX_FILENAME = strcat(training_mix_wav_save_path,filename);
    mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz

    str            = strtok(filename,'_');
%
    clean_files     = dir(fullfile(cs_training_data_path,'*.wav'));
    CLEAN_FILENAME = '';

    for i=1:length(clean_files)
        if( strncmpi(clean_files(i).name,filename, 11)==true )
            CLEAN_FILENAME = sprintf('%s%s',cs_training_data_path,clean_files(i).name);
%             fprintf('%s -> %s\n',filename,clean_files(i).name)
            break;
        end
    end

    clean_sig      = audioread(CLEAN_FILENAME);

    stop_samp = start_samp + numframes(fileNum) - 1;

    % ----------------------- Compute Features --------------------------%
    feats = get_compRP2d_mkcomp2(mix_sig.',globalpara).';
    prelimFeatData(start_samp:stop_samp,:) = feats.';

    % ----------------------- Compute Labels ----------------------------%
    mix_stft   = spectrogram(mix_sig,winlen,overlap,nfft,Fs);
    clean_stft = spectrogram(clean_sig,winlen,overlap,nfft,Fs);

    [labels_r, labels_i] = cIRM(clean_stft,mix_stft, cliplevel,labcompress,globalpara.logistic_max,globalpara.logistic_steep);

    if labwin > 0
        labels_r = makeWindowFeat3(labels_r,labwin).';
        labels_i = makeWindowFeat3(labels_i,labwin).';
    end

    trLabel_r(start_samp:stop_samp,:) = labels_r;
    trLabel_i(start_samp:stop_samp,:) = labels_i;

    % ----------------------- Update counters ---------------------------%
    start_samp = stop_samp + 1;

    if(~mod(fileNum,ten_percent))
        fprintf('%d...',(fileNum/ten_percent)*10)
    end
end
% Nayem edit, Sep 20
%
% for fileNum = 1:num_files%num_files
%
%     filename      = files(fileNum).name;
%     MIX_FILENAME = strcat(training_mix_wav_save_path,filename);
%     mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz
%
%     str            = strtok(filename,'_');
%     CLEAN_FILENAME = sprintf('%s%s',cs_training_data_path,filename);
% %     Nayem edit, Sep 5 CLEAN_FILENAME =
% %     sprintf('%sclean%02d_%dkHz.wav',cs_training_data_path,str2double(str(4:end)),Fs/1000);
%     clean_sig      = audioread(CLEAN_FILENAME);
%
%     stop_samp = start_samp + numframes(fileNum) - 1;
%
%     % ----------------------- Compute Features --------------------------%
%     feats = get_compRP2d_mkcomp2(mix_sig.',globalpara).';
%     prelimFeatData(start_samp:stop_samp,:) = feats.';
%
%     % ----------------------- Compute Labels ----------------------------%
%     mix_stft   = spectrogram(mix_sig,winlen,overlap,nfft,Fs);
%     clean_stft = spectrogram(clean_sig,winlen,overlap,nfft,Fs);
%
%     [labels_r, labels_i] = cIRM(clean_stft,mix_stft, cliplevel,labcompress,globalpara.logistic_max,globalpara.logistic_steep);
%
%     if labwin > 0
%         labels_r = makeWindowFeat3(labels_r,labwin).';
%         labels_i = makeWindowFeat3(labels_i,labwin).';
%     end
%
%     trLabel_r(start_samp:stop_samp,:) = labels_r;
%     trLabel_i(start_samp:stop_samp,:) = labels_i;
%
%     % ----------------------- Update counters ---------------------------%
%     start_samp = stop_samp + 1;
%
%     if(~mod(fileNum,ten_percent))
%         fprintf('%d...',(fileNum/ten_percent)*10)
%     end
% end


clear clean_sig revb_sig filename str REVB_FILENAME CLEAN_FILENAME start_samp stop_samp fileNum
clear files ten_percent globalpara winlen overlap numframes revb_training_data_path
clear nfft labwin hopsize cs_training_data_path Fs

fprintf('\n\tNormalizing training data...\n')

[prelimFeatData, para.tr_mu, para.tr_std] = meanVarArmaNormalize(prelimFeatData, arma_order);
prelimFeatData(isnan(prelimFeatData)) = 0;
trData = makeWindowFeat4(prelimFeatData.',feawin);

end
