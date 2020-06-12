function TestDNN_Spec_denoise_01(VERSION, noise)

%% Load training data and generate features
addpath('./complex_mask/')
addpath('./dnn_mixphone/')
addpath('./dnn_mixphone/costFunc/')
addpath('./dnn_mixphone/main/')
addpath('./dnn_mixphone/utility')
addpath('./dnn_mixphone/debug/')

addpath('./FEATURE_EXTRACTION/')
addpath('./FEATURE_EXTRACTION/ams/')
addpath('./RASTA_TOOLBOX/')
addpath('./COCHLEAGRAM_TOOLBOX/')
addpath('./GENERAL/')
addpath('./PESQ/')
addpath('./SpeechQuality_Toolbox/')
addpath('./OVERLAP_ADD_SYNTHESIS/')

warning('off','all')

noise        = 'SSN';
globalpara   = InitParams_cIRM(noise);
globalpara %#ok<NOPTS>

feawin          = globalpara.feawin;
labwin          = globalpara.labwin;
winlen          = globalpara.winlen;
overlap         = globalpara.overlap;
nfft            = globalpara.nfft;
hopsize         = globalpara.hopsize;
arma_order      = globalpara.arma_order;
noise           = globalpara.noise;

%% Save Test parameters
% teData = []   ;
% teLabel_r = [];
% teLabel_i = [];
% teNumframes = [];
% teFilename_mix =[];

Fs       = 16e3;

ENHANCED_PHRASE = strcat('_specEnh',VERSION);
% VERSION = '_e41';

%%  Server selection
SERVER = 'Eagles'; % 'BigRed2'
CODE = 'Python'; % 'Python'

if strcmpi(CODE,'Python') == 1
    Code_VERSION = VERSION;
    VERSION = strcat(VERSION,'_py');
end

if strcmpi(SERVER,'BigRed2') == 1
    path_data_directory = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data';
    path_code_directory = '/N/u/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM';

elseif strcmpi(SERVER,'Eagles') == 1
    path_data_directory = '/data/knayem';
    path_code_directory = '/home/knayem/EaglesBigred2/cIRM';
    
    enhanced_wavs_data_path = strcat(path_data_directory,'/denoise_specEnh_wavs',VERSION);
    
    if exist(enhanced_wavs_data_path, 'dir')
        fprintf('Path-> %s exists!\n',enhanced_wavs_data_path );
    else
        fprintf('Path-> %s NOT exists!\n',enhanced_wavs_data_path);
        mkdir(enhanced_wavs_data_path)
    end
end

%% Noise type
if nargin < 2
    noise_types = {'SSN','Cafe','Factory','Babble'};
else
    noise_types = {noise};
end
    
TestData_FILe = './dnn_models/Test_datas_spec.mat';
Denoised_spec_File = sprintf('%s/dnn_models/dnn_Spec%s.mat',path_data_directory,Code_VERSION);

load(Denoised_spec_File)

load(TestData_FILe)
teNumframes_sz = size(numframes);
num_files = teNumframes_sz(1)/length(noise_types);

cumsum_numframes = cumsum(numframes);

total_file_count = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Process Reverberant/Clean Training Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for noise_ind = 1:length(noise_types)

%     testing_clean_wav_save_path       = strcat(path_data_directory,'/denoising_clean_wavs_SSN_10noisespercs/testing_16k');
%     testing_matched_mix_wav_save_path = strcat(path_data_directory,'/denoising_mix_wavs_',noise_types{noise_ind},'_10noisespercs/testing_matched');

%     files     = dir(fullfile(testing_matched_mix_wav_save_path,'*.wav'));

    params.winlen = winlen; params.useGPU = 0; params.scalebywindow = 0;
    scores_denoise_fcIRM       = cell(num_files,1);

    if noise_ind == 1
        tot_scores_denoise_fcIRM       = cell(length(noise_types)*num_files,1);
    end

    
    if strcmp(noise_types{noise_ind},'SSN') == 1
        cs_testing_data_path       = strcat(path_data_directory,'/denoising_clean_wavs_SSN_10noisespercs/testing_16k/'); %clean
        testing_mix_wav_save_path  = strcat(path_data_directory,'/denoising_mix_wavs_SSN_15000noisespercs/testing_matched/'); %mixture
    elseif strcmp(noise_types{noise_ind},'Cafe') == 1
        cs_testing_data_path       = strcat(path_data_directory,'/denoising_clean_wavs_SSN_10noisespercs/testing_16k/'); %clean
        testing_mix_wav_save_path  = strcat(path_data_directory,'/denoising_mix_wavs_Cafe_15000noisespercs/testing_matched/'); %mixture
    elseif strcmp(noise_types{noise_ind},'Babble') == 1
        cs_testing_data_path       = strcat(path_data_directory,'/denoising_clean_wavs_SSN_10noisespercs/testing_16k/'); %clean
        testing_mix_wav_save_path  = strcat(path_data_directory,'/denoising_mix_wavs_Babble_15000noisespercs/testing_matched/'); %mixture
    elseif strcmp(noise_types{noise_ind},'Factory') == 1
        cs_testing_data_path       = strcat(path_data_directory,'/denoising_clean_wavs_SSN_10noisespercs/testing_16k/'); %clean
        testing_mix_wav_save_path  = strcat(path_data_directory,'/denoising_mix_wavs_Factory_15000noisespercs/testing_matched/'); %mixture
    end
    
    enhanced_noise_specific_data_path = sprintf('%s/enhanced_%s',enhanced_wavs_data_path,noise_types{noise_ind}); %enhanced
    
    if exist(enhanced_noise_specific_data_path, 'dir')
        fprintf('Path-> %s exists!\n',enhanced_noise_specific_data_path );
    else
        fprintf('Path-> %s NOT exists!\n',enhanced_noise_specific_data_path);
        mkdir(enhanced_noise_specific_data_path)
    end
    
    
    ten_percent = ceil(0.1*num_files);
    fprintf('Processing %d Files for %s noise...\n\t',num_files,noise_types{noise_ind});
    
    
    for fileNum = 1:num_files

        mixFilename = teFilename(total_file_count);
        mixFilename_full = fullfile(testing_mix_wav_save_path,mixFilename);
        mix_stft_r      = spec_mixture_r(total_file_count);
        mix_stft_i      = spec_mixture_i(total_file_count);
        mix_sig_f       = mix_stft_r + 1i.*mix_stft_i;
        mix_sig         = ifft(mix_sig_f);
        
        
        cleanFilename = cleanFilename(total_file_count);
        cleanFilename_full = fullfile(cs_testing_data_path,cleanFilename);
        clean_stft_r    = spec_clean_r(total_file_count);
        clean_stft_i    = spec_clean_i(total_file_count);
        clean_sig_f     = clean_stft_r + 1i.*clean_stft_i;
        clean_sig         = ifft(clean_sig_f);
        
        
        mixNameSplits = strsplit(mixFilename,'.');
        name = char(mixNameSplits(1));
        ext = char(mixNameSplits(2));
        if contains(name,noise_types{noise_ind}) == 0
        	fprintf('Mismatch Noise (%s - %s) at %d.', name, noise_types{noise_ind}, total_file_count)
        end
        
        enhancedFilename = sprintf('%s_%s.%s',name,ENHANCED_PHRASE,ext);
        enhancedFilename_full = fullfile(enhanced_noise_specific_data_path,enhancedFilename);
        if total_file_count == 1
            enhancedMagnitude = y_hat(1:cumsum_numframes(total_file_count));
        else
            enhancedMagnitude = y_hat(cumsum_numframes(total_file_count-1):cumsum_numframes(total_file_count));
        end
        mixPhase = angle(mix_sig_f);
%         mixPhase = atan(mix_stft_i/mix_stft_r);
        
        denoise_sig_f = enhancedMagnitude.*exp(mixPhase*1i);
        denoise_sig = ifft(denoise_sig_f);
        
        
        audiowrite(enhancedFilename_full,denoise_sig/max(abs(denoise_sig)),Fs)
        scores_denoise_fcIRM{fileNum} = comp_dereverb_metrics(clean_sig,mix_sig,denoise_sig,Fs,mixFilename_full,mixFilename);
        tot_scores_denoise_fcIRM{total_file_count} = scores_denoise_fcIRM{fileNum};

        
        if(~mod(fileNum,ten_percent))
            fprintf('%d...',(fileNum/ten_percent)*10)
        end
        
        total_file_count = total_file_count + 1;

    end

    fprintf('\n')

    save(sprintf('./scores/SPECscores_denoising.noise%s%s.mat',noise_types{noise_ind},VERSION), 'scores_*');


end


save(sprintf('./scores/SPECscores_denoising.noisesALL%s.mat',VERSION))
% save(DATA_SAVE_FILE, 'teData', 'teLabel_r', 'teLabel_i', 'teNumframes', 'teFilename_mix','-v7.3');

end
