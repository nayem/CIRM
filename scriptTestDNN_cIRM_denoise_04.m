function scriptTestDNN_cIRM_denoise_04()
%
%   This is for DNN testing
%

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
teData = []   ;
teLabel_r = [];
teLabel_i = [];
teNumframes = [];
teFilename_mix =[];

%% Save Results for visualization

est_cirm_mat = [];
unwrap_est_cirm_mat = [];
gen_stft_mat = [];

est_cirm = [];
unwrap_est_cirm= [];
gen_stft= [];

mixture_stft= [];
clean_stft = [];

%%

Fs       = 16e3;

ENHANCED_PHRASE = 'crmenh_e05_f';
VERSION = '_e05_f';
BEST_MAT_VERSION = '_e11v1';

DATA_SAVE_FILE = sprintf('./dnn_models/Test_datas%s.mat',VERSION);
RESULT_SAVE_FILE = sprintf('./dnn_models/results%s.mat',VERSION);

SERVER = 'Eagles'; % 'BigRed2'
CODE = 'Python'; % 'Python'

% if strcmpi(CODE,'Python') == 1
%     VERSION = strcat(VERSION,'_py');
% end

if strcmpi(SERVER,'BigRed2') == 1
    path_data_directory = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data';
    path_code_directory = '/N/u/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM';

elseif strcmpi(SERVER,'Eagles') == 1
    path_data_directory = '/data/knayem';
    path_code_directory = '/home/knayem/EaglesBigred2/cIRM';
end


mix_wavs_data_path = strcat(path_data_directory,'/denoise_complex_domain_wavs',VERSION);

if exist(mix_wavs_data_path, 'dir')
    fprintf('Path-> %s exists!\n',mix_wavs_data_path );
else
    fprintf('Path-> %s NOT exists!\n',mix_wavs_data_path);
    mkdir(mix_wavs_data_path)
end



if strcmp(noise,'ALL') == 1
    noise_types = {'SSN','Cafe','Babble','Factory'};
else
    noise_types = {noise};
end

total_file_count = 1;

for noise_ind = 1:length(noise_types)

    %%%%%%%%%%%%%%%%%% MATLAB / PYTHON %%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmpi(CODE,'Matlab') == 1
        net_file = strcat('./dnn_models/dnncirm.noise',noise_types{noise_ind}, VERSION, '.mat');

    elseif strcmpi(CODE,'Python') == 1
        net_file = strcat('./dnn_models/dnncirm.noise',noise_types{noise_ind}, BEST_MAT_VERSION, '.mat');
        py_result = strcat('./dnn_models/Real_Imag',VERSION, '.mat');
        load(py_result, 'y_hat');
    end
    
    %%---------------- MATLAB / PYTHON ------------------------%%
    testing_clean_wav_save_path       = strcat(path_data_directory,'/denoising_clean_wavs_',noise_types{noise_ind},'_10noisespercs/testing_16k');
    testing_matched_mix_wav_save_path = strcat(path_data_directory,'/denoising_mix_wavs_',noise_types{noise_ind},'_10noisespercs/testing_matched');


    net_file %#ok<NOPTS>
    load(net_file)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Process Reverberant/Clean Training Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    files     = dir(fullfile(testing_matched_mix_wav_save_path,'*.wav'));

    params.winlen = winlen; params.useGPU = 0; params.scalebywindow = 0;
    scores_denoise_fcIRM       = cell(length(files),1);

    if noise_ind == 1
        tot_scores_denoise_fcIRM       = cell(length(noise_types)*length(files),1);
    end

    ten_percent = ceil(0.1*length(files));
    fprintf('Processing %d Files for %s noise...\n\t',length(files),noise_types{noise_ind});

    for fileNum = 1:length(files)

        filename      = files(fileNum).name;
        MIX_FILENAME = fullfile(testing_matched_mix_wav_save_path,filename);
        mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz
        [mix_stft,freqs,t]     = spectrogram(mix_sig, hann(winlen), overlap, nfft, Fs);

        [tmpe,str]       = strtok(filename,'_');
        [~,str] = strtok(str(2:end),'_');
        str              = strtok(str(2:end),'_');
        snrNum           = str(4:end);
        %
        clean_files     = dir(fullfile(testing_clean_wav_save_path,'*.wav'));
        CLEAN_FILENAME = '';

        for i=1:length(clean_files)
            if( strncmpi(clean_files(i).name,filename, 11)==true )
                CLEAN_FILENAME = fullfile(testing_clean_wav_save_path,clean_files(i).name);
                %fprintf('%s -> %s\n',filename,clean_files(i).name);
                break;
            end
        end

        clean_sig        = audioread(CLEAN_FILENAME);

        % -------------------- Compute Features ------------------------------%
        feats = get_compRP2d_mkcomp2(mix_sig.',globalpara).';

        % --------------------- Compute feature deltas ----------------------%

        featData = meanVarArmaNormalize(feats.', arma_order); %% TRY USING meanVarArmaNormalize_Test instead
        featData(isnan(featData)) = 0;
        featData = makeWindowFeat3(featData.',feawin).';

        % ----------------- Process features through network ----------------%
        [output1, output2] = getOutputFromNet_mixphone(net, featData, opts);
        output1 = gather(output1); output2 = gather(output2);

        % ----------------- Generate Estimate -------------------------------%
        real_output = output1;
        imag_output = output2;
        
        start_frame = sum(teNumframes)+1;
        te_sz = size(featData);
        end_frame = start_frame+te_sz(1)-1;
        real_y_hat = y_hat(start_frame:end_frame, 1:963);
        imag_y_hat = y_hat(start_frame:end_frame, 964:end);

        %% ============ MAT cirm and estimate ============ %%
        complex_irmmask_mat = complex(real_output.',imag_output.');

        if ~labwin
            estimate_mat = complex_irmmask_mat.*mix_stft;
        else
            real_output_unwrap = unwrapAugmentedTF_wAvg(real_output.',labwin);
            imag_output_unwrap = unwrapAugmentedTF_wAvg(imag_output.',labwin);
            complex_irmmask_mat    = complex(real_output_unwrap,imag_output_unwrap);
            estimate_mat           = complex_irmmask_mat.*mix_stft;
        end
        
        %% ============ Python cirm and estimate ============ %%
        complex_irmmask = complex(real_y_hat.',imag_y_hat.');

        if ~labwin
            estimate = complex_irmmask.*mix_stft;
        else
            real_output_unwrap = unwrapAugmentedTF_wAvg(real_y_hat.',labwin);
            imag_output_unwrap = unwrapAugmentedTF_wAvg(imag_y_hat.',labwin);
            complex_irmmask    = complex(real_output_unwrap,imag_output_unwrap);
            estimate           = complex_irmmask.*mix_stft;
        end

        %%
        denoise_sig_mat  = overlapAndAdd(estimate_mat,length(mix_sig),nfft,hopsize,params);
        denoise_sig  = overlapAndAdd(estimate,length(mix_sig),nfft,hopsize,params);
        
        [denoise_stft_mat,freqs,t]     = spectrogram(denoise_sig_mat/max(abs(denoise_sig_mat)), hann(winlen), overlap, nfft, Fs);
        [denoise_stft,freqs,t]     = spectrogram(denoise_sig/max(abs(denoise_sig)), hann(winlen), overlap, nfft, Fs);
        
        [clean_spec,freqs,t] = spectrogram(clean_sig, hann(winlen), overlap, nfft, Fs);
        
        %% ====== Gather Parameters for Test ======= %%
        teData = [teData; featData]   ;
        teLabel_r = [teLabel_r; real_output];
        teLabel_i = [teLabel_i; imag_output];
        te_sz = size(featData);
        teNumframes = [teNumframes; te_sz(1)];
        teFilename_mix = [teFilename_mix; string(filename)];
        
        % concating all cirm and stft along time axis
        est_cirm_mat = [est_cirm_mat,complex_irmmask_mat];
        unwrap_est_cirm_mat = [unwrap_est_cirm_mat,estimate_mat];
        gen_stft_mat = [gen_stft_mat,denoise_stft_mat];

        est_cirm = [est_cirm,complex_irmmask];
        unwrap_est_cirm= [unwrap_est_cirm,estimate];
        gen_stft= [gen_stft,denoise_stft];

        mixture_stft= [mixture_stft,mix_stft];% concating stft along time axis
        clean_stft = [clean_stft,clean_spec];% concating stft along time axis


        

        % ----------------------- Compute Scores ----------------------------%
        f = (strsplit(filename,'.'));
        dir_path = fullfile(mix_wavs_data_path,char(f(1)));
        filename    = sprintf('%s_%s.%s',dir_path,ENHANCED_PHRASE,char(f(2)) );

%         Nayem edit, Sep 12
%         filename    = sprintf('%s%d.cIRM_denoised.noise%s.snrNum%s.wav',mix_wavs_data_path,fileNum,noise_types{noise_ind},snrNum);%cs_count,rem);

        audiowrite(filename,denoise_sig/max(abs(denoise_sig)),Fs)
        scores_denoise_fcIRM{fileNum} = comp_dereverb_metrics(clean_sig,mix_sig,denoise_sig,Fs,MIX_FILENAME,filename);
        tot_scores_denoise_fcIRM{total_file_count} = scores_denoise_fcIRM{fileNum};

        if(~mod(fileNum,ten_percent))
            fprintf('%d...',(fileNum/ten_percent)*10)
        end
        total_file_count = total_file_count + 1;
    end

    fprintf('\n')

    save(sprintf('./scores/cIRMscores_denoising.noise%s%s.mat',noise_types{noise_ind},VERSION), 'scores_*');


end


save(sprintf('./scores/cIRMscores_denoising.noisesALL%s.mat',VERSION))
save(DATA_SAVE_FILE, 'teData', 'teLabel_r', 'teLabel_i', 'teNumframes', 'teFilename_mix','-v7.3');
save(RESULT_SAVE_FILE,'est_cirm_mat','unwrap_est_cirm_mat','gen_stft_mat','est_cirm','unwrap_est_cirm','gen_stft','mixture_stft','clean_stft','-v7.3');

end
