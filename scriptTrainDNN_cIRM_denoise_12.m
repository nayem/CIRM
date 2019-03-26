function scriptTrainDNN_cIRM_denoise_12(noise)

% Description: Prepare feature function gives framesize too for LSTM
%               Also store STFT of train, dev and test for Signal
%               Approzimation
%
% Input:
%   - noise: Noise type, e.g. 'SSN'
%
% Output:
%   - DATA_SAVE_FILE: Save Data 
%                       trData(Training input data), trLabel_r(Training label real part),
%                       trLabel_i(Training label imaginary part),
%                       trNumframes(Training num frames), trSTFT(Training complex STFT)
%
%                       cvData(Dev input data), cvLabel_r(Dev label real part),
%                       cvLabel_i(Dev label imaginary part),
%                       cvNumframes(Dev num frames), cvSTFT(Dev complex STFT)
%                       
%                       teData(Test input data), teLabel_r(Test label real part),
%                       teLabel_i(Test label imaginary part),
%                       teNumframes(Test num frames), teSTFT(Test complex STFT)
%
%   - PARAM_SAVE_FILE: Save parameters
%   - ModelFN: Matlab Trained Model (Fully connected 3 layer)
%
% Author: Khandokar Md. Nayem, Sep 24, 2018


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
noise = 'SSN';
[~,sys_name] = system('hostname');

globalpara   = InitParams_cIRM(noise);
globalpara %#ok<NOPTS>

test_noise    = globalpara.noise;

SERVER = 'Eagles'; % 'BigRed2'
VERSION = '_e12v1';

%% Setup SERVER path

if strcmpi(SERVER,'BigRed2') == 1
    path_data_directory = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data';
    path_code_directory = '/N/u/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM';

elseif strcmpi(SERVER,'Eagles') == 1
    path_data_directory = '/data/knayem';
    path_code_directory = '/home/knayem/EaglesBigred2/cIRM';
end


%% Train paths

if strcmp(test_noise,'SSN') == 1
    cs_training_data_path      = strcat(path_data_directory,'/denoising_clean_wavs_SSN_10noisespercs/training_16k/');
    training_mix_wav_save_path = strcat(path_data_directory,'/denoising_mix_wavs_SSN_15000noisespercs/training_16k/');

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
%[trData, trLabel_r, trLabel_i, opts, trNumframes] = prepareTrainingData_cIRM_denoise12(globalpara,cs_training_data_path,training_mix_wav_save_path);


%% Dev paths

if strcmp(test_noise,'SSN') == 1
    cs_dev_data_path      = strcat(path_data_directory,'/denoising_clean_wavs_SSN_10noisespercs/development_16k/');
    noisy_dev_data_path   = strcat(path_data_directory,'/denoising_mix_wavs_SSN_15000noisespercs/development_16k/');

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
%[cvData, cvLabel_r, cvLabel_i, opts, cvNumframes]  = prepareDevData_cIRM_denoise12(globalpara,cs_dev_data_path,noisy_dev_data_path);


%% Test paths
if strcmp(test_noise,'SSN') == 1
    cs_testing_data_path       = strcat(path_data_directory,'/denoising_clean_wavs_SSN_10noisespercs/testing_16k/');
    testing_mix_wav_save_path  = strcat(path_data_directory,'/denoising_mix_wavs_SSN_15000noisespercs/testing_matched/');
end

fprintf('Extracting Features/Labels from Testing Data...\n')
% [teData, teLabel_r, teLabel_i, opts, teNumframes]  = prepareTestingData_cIRM_denoise12(globalpara,cs_testing_data_path,testing_mix_wav_save_path);


load('/home/knayem/EaglesBigred2/cIRM/dnn_models/Train_datas.mat', 'trData', 'trLabel_r');
load('/home/knayem/EaglesBigred2/cIRM/dnn_models/CrossValidation_datas.mat');
load('/home/knayem/EaglesBigred2/cIRM/dnn_models/Test_datas.mat');
load('/home/knayem/EaglesBigred2/cIRM/dnn_models/DNN_params.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Initializing DNN parameters...\n')
opts = InitiatlizeNN_cIRM(opts,trData,trLabel_r);

%% Matlab write model file
ModelFN = sprintf('./dnn_models/dnncirm.noise%s%s.mat',globalpara.noise,VERSION);
opts = updateOpts(opts, globalpara);

label_dims         = size(trLabel_r,2);
opts.net_struct    = {size(trData,2)};
opts.net_struct{2} = 1024;
opts.net_struct{3} = 1024;
opts.net_struct{4} = 1024;
opts.net_struct{5} = {label_dims,label_dims};
opts %#ok<NOPTS>

%label_dims         = size(trLabel_r,2);
%opts.net_struct    = {size(trData,2)};
%opts.net_struct{2} = 4;
%opts.net_struct{3} = 4;
%opts.net_struct{4} = 4;
%opts.net_struct{5} = {label_dims,label_dims};
%opts

%% Python support files
% opts = opts
PARAM_SAVE_FILE = sprintf('./dnn_models/DNN_params.mat');
save(PARAM_SAVE_FILE, 'opts', '-v7.3');

DATA_SAVE_FILE = sprintf('./dnn_models/DNN_datas%s.mat',VERSION);
PARAM_SAVE_FILE = sprintf('./dnn_models/DNN_params%s.mat',VERSION);

save(DATA_SAVE_FILE, 'trData', 'trLabel_r', 'trLabel_i', 'trNumframes', 'cvData', 'cvLabel_r', 'cvLabel_i', 'cvNumframes', '-v7.3');


fprintf('Training DNN...\n')
[net, pre_net] = funcDeepNetTrainNoRolling_crm_05(trData,trLabel_r,trLabel_r,[],cvData,cvLabel_r,cvLabel_i, [],opts);


disp(ModelFN)
save(ModelFN,'net','opts','pre_net','-v7.3');

end