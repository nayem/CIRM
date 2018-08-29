function  [cvData, cvLabel_r,cvLabel_i,numframes] = prepareDevData_cIRM_denoise2(para,cs_dev_data_path,noisy_dev_data_path)

% Description: Prepare Dev features for LSTM
%
% Input:
%   - globalpara: global parameters
%   - cs_dev_data_path: Clean signal path for Dev
%   - noisy_dev_data_path: Noisy signal path for Dev
%
% Output
%   - cvData: Feature data for all files appended one after another
%   - cvLabel_r: Dev label real value
%   - cvLabel_i: Dev label imaginary value
%   - numframes: Number of frames for each file
%
% Author: Khandokar Md. Nayem, May 29, 2018

%% Load cross validation data and generate features
feawin        = para.feawin;
labwin        = para.labwin;
winlen        = para.winlen;
overlap       = para.overlap;
nfft          = para.nfft;
Fs            = para.Fs;
cliplevel     = para.clip_level;
arma_order    = para.arma_order;
labcompress   = para.labcompress;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Process Reverberant/Clean Training Data
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

STATS_FILE = fullfile(path_code_directory,'dnn_models/dev_stats.mat');

% list for 6 SNRs and Clean
clean = []; snr_n6 = []; snr_n3 = []; 
snr_0 = []; snr_3 = []; snr_6 = []; snr_=[];

snr_exp = '(-)?(\d)+dB';
%%===========================================================%%

fprintf('\tComputing number of frames...\n\t\t')
ten_percent = ceil(0.1*num_files);

for fileNum = 1:num_files

    filename     = files(fileNum).name;
    MIX_FILENAME = strcat(noisy_dev_data_path,filename);
    mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz

    numframes(fileNum) = fix((length(mix_sig) - overlap)/(winlen-overlap));

    if(~mod(fileNum,ten_percent))
        fprintf('%d...',(fileNum/ten_percent)*10)
    end
end

% ------------------------ Initialize data -------------------------------%

prelimFeatData = zeros(sum(numframes),246,'single');

cvLabel_r    = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');
cvLabel_i    = zeros(sum(numframes),(nfft/2 + 1)*(2*labwin + 1),'single');

start_samp = 1;

fprintf('\n\tExtracting features and labels...\n\t\t')

for fileNum = 1:num_files

    filename      = files(fileNum).name;
    MIX_FILENAME = strcat(noisy_dev_data_path,filename);
    mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz

    str            = strtok(filename,'_');
%
    clean_files     = dir(fullfile(cs_dev_data_path,'*.wav'));
    CLEAN_FILENAME = '';

    for i=1:length(clean_files)
        if( strncmpi(clean_files(i).name,filename, 11)==true )
            CLEAN_FILENAME = sprintf('%s%s',cs_dev_data_path,clean_files(i).name);
%            fprintf('%s -> %s\n',filename,clean_files(i).name)
            break;
        end
    end

%     Nayem edit, Sep 20
%     CLEAN_FILENAME = sprintf('%sclean%02d_%dkHz.wav',cs_dev_data_path,str2double(str(4:end)),Fs/1000);
    clean_sig      = audioread(CLEAN_FILENAME);

    stop_samp = start_samp + numframes(fileNum) - 1;

    % ----------------------- Compute Features --------------------------%

    feats = get_compRP2d_mkcomp2(mix_sig.',para).';
    prelimFeatData(start_samp:stop_samp,:) = feats.';

    % ----------------------- Compute Labels ----------------------------%

    mix_stft   = spectrogram(mix_sig,winlen,overlap,nfft,Fs);
    clean_stft = spectrogram(clean_sig,winlen,overlap,nfft,Fs);

    [labels_r, labels_i] = cIRM(clean_stft,mix_stft, cliplevel,labcompress,para.logistic_max,para.logistic_steep);

    if labwin > 0
        labels_r = makeWindowFeat3(labels_r,labwin).';
        labels_i = makeWindowFeat3(labels_i,labwin).';
    end

    cvLabel_r(start_samp:stop_samp,:) = labels_r;
    cvLabel_i(start_samp:stop_samp,:) = labels_i;

    % ======================== Store States ============================ %
    clean=[clean, clean_stft];
    [s,e]=regexp(filename,snr_exp, 'once');
    
    switch filename(s:e)
        case '-6dB'
            snr_n6=[snr_n6, mix_stft];
        case '-3dB'
            snr_n3=[snr_n3, mix_stft];
        case '0dB'
            snr_0=[snr_0, mix_stft];
        case '3dB'
            snr_3=[snr_3, mix_stft];
        case '6dB'
            snr_6=[snr_6, mix_stft];
        otherwise
            snr_=[snr_, mix_stft];
    end
    % ================================================================== %
    
    start_samp = stop_samp + 1;

    if(~mod(fileNum,ten_percent))
        fprintf('%d...',(fileNum/ten_percent)*10)
    end
end

save(STATS_FILE, 'snr_n6', 'snr_n3', 'snr_0', 'snr_3', 'snr_6','snr_', 'clean','numframes','-v7.3');

clear clean_sig revb_sig filename str REVB_FILENAME CLEAN_FILENAME start_samp stop_samp fileNum
% clear files ten_percent para winlen revb_dev_data_path overlap numframes
clear files ten_percent para winlen revb_dev_data_path overlap 
clear nfft labwin hopsize cs_dev_data_path Fs

fprintf('\n\tNormalizing cross validation data...\n')

[prelimFeatData, para.tr_mu, para.tr_std] = meanVarArmaNormalize(prelimFeatData, arma_order);
prelimFeatData(isnan(prelimFeatData)) = 0;
cvData = makeWindowFeat4(prelimFeatData.',feawin);

end
