function scriptTrainDNN_cIRM_denoise(noise)

% This file is an dereverberation example for training and test. The user only need to specify the data location.

addpath('./complex_mask/')
addpath('./dnn_mixphone/')
addpath('./dnn_mixphone/costFunc/')
addpath('./dnn_mixphone/main/')
addpath('./dnn_mixphone/utility')
addpath('./dnn_mixphone/debug/')

addpath('./RASTA_TOOLBOX/')
addpath('./FEATURE_EXTRACTION/')
addpath('./FEATURE_EXTRACTION/ams/')
addpath('./COCHLEAGRAM_TOOLBOX/')
addpath('./GENERAL/')
addpath('./OVERLAP_ADD_SYNTHESIS/')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data preparation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Initializing parameters...\n')

[~,sys_name] = system('hostname');

globalpara   = InitParams_cIRM(noise);
globalpara %#ok<NOPTS>

test_noise    = globalpara.noise;


if strcmp(test_noise,'SSN') == 1
    cs_training_data_path      = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/training_16k/';
    training_mix_wav_save_path = '/data/knayem/denoising_mix_wavs_SSN_10noisespercs/training_16k/';
%     Nayem edit, Sep 5
%     cs_training_data_path      = './denoising_clean_wavs_SSN_10noisespercs/training/';
%     training_mix_wav_save_path = './denoising_mix_wavs_SSN_10noisespercs/training/';
elseif strcmp(test_noise,'Cafe') == 1
    cs_training_data_path      = './denoising_clean_wavs_Cafe_10noisespercs/training/';
    training_mix_wav_save_path = './denoising_mix_wavs_Cafe_10noisespercs/training/';
elseif strcmp(test_noise,'Babble') == 1
    cs_training_data_path      = './denoising_clean_wavs_Babble_10noisespercs/training/';
    training_mix_wav_save_path = './denoising_mix_wavs_Babble_10noisespercs/training/';
elseif strcmp(test_noise,'Factory') == 1
    cs_training_data_path      = './denoising_clean_wavs_Factory_10noisespercs/training/';
    training_mix_wav_save_path = './denoising_mix_wavs_Factory_10noisespercs/training/';
end

fprintf('Extracting Features/Labels from Training Data...\n')
[trData, trLabel_r, trLabel_i, opts] = prepareTrainingData_cIRM_denoise(globalpara,cs_training_data_path,training_mix_wav_save_path);

if strcmp(test_noise,'SSN') == 1
    cs_dev_data_path      = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/development_16k/';
    noisy_dev_data_path   = '/data/knayem/denoising_mix_wavs_SSN_10noisespercs/development_16k/';
%     Nayem edit, Sep 5
%     cs_dev_data_path      = './denoising_clean_wavs_SSN_10noisespercs/development/';
%     noisy_dev_data_path   = './denoising_mix_wavs_SSN_10noisespercs/development/';
elseif strcmp(test_noise,'Cafe') == 1
    cs_dev_data_path      = './denoising_clean_wavs_Cafe_10noisespercs/development/';
    noisy_dev_data_path   = './denoising_mix_wavs_Cafe_10noisespercs/development/';
elseif strcmp(test_noise,'Babble') == 1
    cs_dev_data_path      = './denoising_clean_wavs_Babble_10noisespercs/development/';
    noisy_dev_data_path   = './denoising_mix_wavs_Babble_10noisespercs/development/';
elseif strcmp(test_noise,'Factory') == 1
    cs_dev_data_path      = './denoising_clean_wavs_Factory_10noisespercs/development/';
    noisy_dev_data_path   = './denoising_mix_wavs_Factory_10noisespercs/development/';
end

fprintf('Extracting Features/Labels from Development Data...\n')
[cvData, cvLabel_r, cvLabel_i]  = prepareDevData_cIRM_denoise(globalpara,cs_dev_data_path,noisy_dev_data_path);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% trData = zeros(195192,1230);
% trLabel_r = zeros(195192,963);
% trLabel_i = zeros(195192,963);
% cvData = zeros(44961,1230);
% cvLabel_r = zeros(44961,963);
% cvLabel_i = zeros(44961,963);
% opts.nayem = 0;
% opts.noise = 'SSN';

fprintf('Initializing DNN parameters...\n')
opts = InitiatlizeNN_cIRM(opts,trData,trLabel_r);

ModelFN = sprintf('./dnn_models/dnncirm.noise%s.mat',globalpara.noise);

label_dims         = size(trLabel_r,2);
opts.net_struct    = {size(trData,2)};
opts.net_struct{2} = 1024;
opts.net_struct{3} = 1024;
opts.net_struct{4} = 1024;
opts.net_struct{5} = {label_dims,label_dims};
opts %#ok<NOPTS>


DATA_SAVE_FILE = './dnn_models/DNN_datas.mat';
PARAM_SAVE_FILE = './dnn_models/DNN_params.mat';
% train_feats = trData
% train_label_real = trLabel_r
% train_label_imag = trLabel_r
% train_weights = []
% dev_feats = cvData
% dev_label_real = cvLabel_r
% dev_label_imag = cvLabel_i
% dev_weights = []
save(DATA_SAVE_FILE, 'trData', 'trLabel_r', 'trLabel_i', 'cvData', 'cvLabel_r', 'cvLabel_i','-v7.3');
% opts = opts
save(PARAM_SAVE_FILE, 'opts', '-v7.3');

fprintf('Training DNN...\n')
[net, pre_net] = funcDeepNetTrainNoRolling_crm(trData,trLabel_r,trLabel_r,[],cvData,cvLabel_r,cvLabel_i, [],opts);

disp(ModelFN)
save(ModelFN,'net','opts','pre_net','-v7.3');

end