function generate_wavs_rnn02(codeVersion, sNR, noises)
%   This function will predictions of RNN output and generate enhaned wavs.
%   Input Arguments:
%       paramVersion: <str> version of parameter .mat; 
%       codeVersion : <str> version of parameter .mat
%       sNR         : <cell array> SNR list, 
%       noises      : <cell array> noise list
%
%   Nayem, Nov 13, 2018
    
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

    codeVersion  = '_e12v1';
    sNR = {-6, -3, 0, 3, 6};
    noise_types = {'SSN'};
    %noise_types = {'SSN', 'CAFE', 'BABBLE', 'FACTORY'};
    
    
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

    params.winlen = winlen; 
    params.useGPU = 0; 
    params.scalebywindow = 0;
    
    %% load parameters, data, output
    PARA_FILE = fullfile('./dnn_models/DNN_params.mat');
    TestData_FILE = fullfile('./dnn_models/Test_datas.mat');
    OUTPUT_FILE = strcat('./dnn_models/Real_Imag',codeVersion,'.mat');
    
    load(PARA_FILE, 'opts');
    load(TestData_FILE,'teData', 'teLabel_r','teLabel_i','numframes','teFilename','cleanFilename','spec_mixture_r', 'spec_mixture_i');
    load(OUTPUT_FILE,'y_hat');

    %%
    path_data_directory = '/data/knayem';
    mix_wavs_data_path = char(strcat(path_data_directory,'/denoise_complex_domain_wavs',codeVersion));
    testing_matched_mix_wav_save_path= fullfile(path_data_directory,'denoising_mix_wavs_SSN_15000noisespercs/testing_matched');
    testing_clean_wav_save_path= fullfile(path_data_directory,'denoising_clean_wavs_SSN_10noisespercs/testing_16k');

    if exist(char(mix_wavs_data_path), 'dir')
        fprintf('Path-> %s exists!\n',mix_wavs_data_path );
    else
        fprintf('Path-> %s NOT exists!\n',mix_wavs_data_path);
        mkdir(mix_wavs_data_path)
    end
    
    ENHANCED_PHRASE = string(strcat('crmenh',codeVersion));
    
    %%
    Fs       = 16e3;
%     base_name_exp = 'S_(\d)+_(\d)+_(\d)+k';
%     Numframes = cumsum(teNumframes);
% 
%     rxp = '_e(\d)+v[-\d]+';
%     VERSION=string(regexp(predicted_file,rxp,'match'));
    
    
    %% Score matrix (samples,Noises,SNRs)
    sz = size(numframes, 1);
    if (nargin==4)
        if strcmp(noises,'ALL') == 1
            noise_types = {'SSN', 'CAFE', 'BABBLE', 'FACTORY'};
        end
    elseif (nargin==3)
        noise_types = {'SSN'};
    elseif (nargin==2)
        sNR = {-6, -3, 0, 3, 6};
        noise_types = {'SSN'};
    end
    
    %scores = zeros(sz/length(noises)/length(sNR), length(noise_types),length(sNR));
    scores_denoise_fcIRM = cell(sz,1);
    tot_scores_denoise_fcIRM = cell(sz,1);
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Process Reverberant/Clean Training Data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for noise_ind = 1:length(noise_types)
    
        noise_rexp = strcat('(.)+',noise_types{noise_ind},'(.)+');
        nfile = 0;
        ten_percent = ceil(0.1*sz);
        fprintf('Processing %d Files for %s noise...\n\t', sz, noise_types{noise_ind});

        
        for fileNum = 1:sz

            if fileNum == 1
                start_index = 1;
                end_index = numframes(fileNum);
            else
                start_index = end_index+1;
                end_index = start_index+numframes(fileNum)-1;
            end
            
            
            mix_filename      = char(teFilename(fileNum));
            
%             if isempty(regexp(mix_filename,noise_rexp,'match'))
%                 continue
%             else
                nfile = nfile+1;
%             end
            
            MIX_FILENAME = fullfile(testing_matched_mix_wav_save_path,mix_filename);
            mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz
            %mix_stft     = complex(spec_mixture_r(start_index:end_index,:).', spec_mixture_i(start_index:end_index,:).');
            mix_stft     = spectrogram(mix_sig, hann(winlen), overlap, nfft, Fs);

            clean_filename      = char(strcat(cleanFilename(fileNum),'.wav'));
            CLEAN_FILENAME = fullfile(testing_clean_wav_save_path,clean_filename);
            clean_sig      = audioread(CLEAN_FILENAME);

            % ----------------- Generated Estimate -------------------------------%
            real_output = y_hat(fileNum, 1:numframes(fileNum), 1:963);
            imag_output = y_hat(fileNum, 1:numframes(fileNum), 964:1926);

            [~,r,c] = size(real_output);
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
            f = sprintf('%s_%s.wav',mix_filename(1:(end-4)),ENHANCED_PHRASE);
            enhance_filename = fullfile(mix_wavs_data_path,f);

            audiowrite(enhance_filename,denoise_sig/max(abs(denoise_sig)),Fs)
            scores_denoise_fcIRM{nfile} = comp_dereverb_metrics(clean_sig,mix_sig,denoise_sig,Fs,MIX_FILENAME,enhance_filename);
            tot_scores_denoise_fcIRM{fileNum} = scores_denoise_fcIRM{nfile};

            if(~mod(fileNum,ten_percent))
                fprintf('%d...',(fileNum/ten_percent)*10)
            end

        end
        save(sprintf('./scores/cIRMscores_denoising.noise%s%s.mat',noise_types{noise_ind},codeVersion), 'scores_*');
    end
    
    save(sprintf('./scores/cIRMscores_denoising.noisesALL%s.mat',codeVersion))

end
