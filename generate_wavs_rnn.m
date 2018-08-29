function generate_wavs_rnn(predicted_file)
%   This function will predictions of RNN output and generate enhaned wavs.
%
%
%
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

    feawin          = globalpara.feawin;
    labwin          = globalpara.labwin;
    winlen          = globalpara.winlen;
    overlap         = globalpara.overlap;
    nfft            = globalpara.nfft;
    hopsize         = globalpara.hopsize;
    arma_order      = globalpara.arma_order;
    noise           = globalpara.noise;

    %% load parameters\
    TEST_PARAMS = './dnn_models/Test_datas_e10v5.mat';
    predicted_file = strcat('./dnn_models/',predicted_file);
    
    load(TEST_PARAMS,'teLabel_r','teLabel_i','teNumframes','teFilename_mix');
    load(predicted_file,'y_hat');

    
    Fs       = 16e3;
    base_name_exp = 'S_(\d)+_(\d)+_(\d)+k';
    Numframes = cumsum(teNumframes);

    rxp = '_e(\d)+v[-\d]+';
    VERSION=string(regexp(predicted_file,rxp,'match'));
    ENHANCED_PHRASE = string(strcat('crmenh',VERSION));

    SERVER = 'Eagles'; % 'BigRed2'
    CODE = 'Python'; % 'Python'

    if strcmpi(CODE,'Python') == 1
        VERSION = strcat(VERSION,'_py');
    end

    if strcmpi(SERVER,'BigRed2') == 1
        path_data_directory = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data';
        path_code_directory = '/N/u/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM';

    elseif strcmpi(SERVER,'Eagles') == 1
        path_data_directory = '/data/knayem';
        path_code_directory = '/home/knayem/EaglesBigred2/cIRM';
    end


    mix_wavs_data_path = char(strcat(path_data_directory,'/denoise_complex_domain_wavs',VERSION));

    if exist(char(mix_wavs_data_path), 'dir')
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
        if strcmpi(CODE,'Python') == 1
            pyVERSION = strtok(VERSION, '_');
            pyVERSION = strcat('_',pyVERSION);
        end
        %%---------------- MATLAB / PYTHON ------------------------%%


        testing_clean_wav_save_path       = strcat(path_data_directory,'/denoising_clean_wavs_',noise_types{noise_ind},'_10noisespercs/testing_16k');
        testing_matched_mix_wav_save_path = strcat(path_data_directory,'/denoising_mix_wavs_',noise_types{noise_ind},'_10noisespercs/testing_matched');

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Process Reverberant/Clean Training Data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         files     = dir(fullfile(testing_matched_mix_wav_save_path,'*.wav'));
        files = size(teFilename_mix);

        params.winlen = winlen; params.useGPU = 0; params.scalebywindow = 0;
        scores_denoise_fcIRM       = cell(files(1),1);

        if noise_ind == 1
            tot_scores_denoise_fcIRM       = cell(length(noise_types)*files(1),1);
        end

        ten_percent = ceil(0.1*files(1));
        fprintf('Processing %d Files for %s noise...\n\t',files(1),noise_types{noise_ind});

        
        for fileNum = 1:files(1)

            filename      = char(teFilename_mix(fileNum));
            MIX_FILENAME = fullfile(testing_matched_mix_wav_save_path,filename);
            mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz
            mix_stft     = spectrogram(mix_sig, hann(winlen), overlap, nfft, Fs);
            
            
            [s,e]=regexp(filename,base_name_exp);
            base_name = filename(s:e);
            mix_name =  strtok(filename,'.');
            
            clean_file     = strcat(base_name,'.wav');
            CLEAN_FILENAME = fullfile(testing_clean_wav_save_path,clean_file);

            clean_sig        = audioread(CLEAN_FILENAME);

            % ----------------- Generated Estimate -------------------------------%
            real_output = y_hat(fileNum,1:teNumframes(fileNum),1:963);
            imag_output = y_hat(fileNum,1:teNumframes(fileNum),964:1926);
            
            [i ,r,c] = size(real_output);
            real_output=reshape(real_output, [r,c]);
            imag_output=reshape(imag_output, [r,c]);

            complex_irmmask = complex(real_output.',imag_output.');

            if ~labwin
                estimate = complex_irmmask.*mix_stft;
            else
                real_output_unwrap = unwrapAugmentedTF_wAvg(real_output.',labwin);
                imag_output_unwrap = unwrapAugmentedTF_wAvg(imag_output.',labwin);
                complex_irmmask    = complex(real_output_unwrap,imag_output_unwrap);
                estimate           = complex_irmmask.*mix_stft;
            end

            denoise_sig  = overlapAndAdd(estimate,length(mix_sig),nfft,hopsize,params);

            % ----------------------- Compute Scores ----------------------------%
            f = sprintf('%s_%s.%s',mix_name,ENHANCED_PHRASE, 'wav');
            filename = fullfile(mix_wavs_data_path,f);

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

end
