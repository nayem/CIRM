function scriptTestDNN_cIRM_denoise_02()

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

Fs       = 16e3;

ENHANCED_PHRASE = 'crmenh_v2_8';
VERSION = '_v2_8';

%%%%%%%%%%%%%%%%%% MATLAB / PYTHON %%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab testing
% mix_wavs_data_path = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoise_complex_domain_wavs/';

% Python testing
mix_wavs_data_path = sprintf('/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoise_complex_domain_wavs%s/',VERSION);
%%---------------- MATLAB / PYTHON ------------------------%%


if strcmp(noise,'ALL') == 1
    noise_types = {'SSN','Cafe','Babble','Factory'};
else
    noise_types = {noise};
end

total_file_count = 1;

for noise_ind = 1:length(noise_types)

    %%%%%%%%%%%%%%%%%% MATLAB / PYTHON %%%%%%%%%%%%%%%%%%%%%%%%%
    % Matlab testing
    % net_file = sprintf('./dnn_models/dnncirm.noise%s_02.mat',noise_types{noise_ind});

    % Python testing
    net_file = sprintf('./dnn_models/DNN_CIRM_net%s.mat',VERSION);
    %%---------------- MATLAB / PYTHON ------------------------%%


    testing_clean_wav_save_path       = sprintf('/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_clean_wavs_SSN_10noisespercs/testing_16k/',noise_types{noise_ind});
    testing_matched_mix_wav_save_path = sprintf('/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_mix_wavs_SSN_10noisespercs/testing_matched/',noise_types{noise_ind});


%     Nayem edit, Sep 10
%     testing_clean_wav_save_path       = sprintf('./denoising_clean_wavs_%s_10noisespercs/testing/',noise_types{noise_ind});
%     testing_matched_mix_wav_save_path = sprintf('./denoising_mix_wavs_%s_10noisespercs/testing_matched/',noise_types{noise_ind});

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
    fprintf('Processing %d Files for %s noise...\n\t',length(files),noise_types{noise_ind})

    for fileNum = 1:length(files)

        filename      = files(fileNum).name;
        MIX_FILENAME = strcat(testing_matched_mix_wav_save_path,filename);
        mix_sig      = audioread(MIX_FILENAME);   % make sure the sampling frequency is 16 kHz
        mix_stft     = spectrogram(mix_sig, hann(winlen), overlap, nfft, Fs);

        [tmpe,str]       = strtok(filename,'_');
        [~,str] = strtok(str(2:end),'_');
        str              = strtok(str(2:end),'_');
        snrNum           = str(4:end);
        %
        clean_files     = dir(fullfile(testing_clean_wav_save_path,'*.wav'));
        CLEAN_FILENAME = '';

        for i=1:length(clean_files)
            if( strncmpi(clean_files(i).name,filename, 11)==true )
                CLEAN_FILENAME = sprintf('%s%s',testing_clean_wav_save_path,clean_files(i).name);
                %fprintf('%s -> %s\n',filename,clean_files(i).name)
                break;
            end
        end

%     Nayem edit, Sep 20
%         Nayem edit, Sep 10
%         CLEAN_FILENAME   = sprintf('%sclean%02d_%dkHz.wav',testing_clean_wav_save_path,str2double(tmpe(4:end)),Fs/1000);
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

        complex_irmmask = complex(real_output.',imag_output.');

        if ~labwin
            estimate = complex_irmmask.*mix_stft;
        else
            real_output_unwrap = unwrapAugmentedTF_wAvg(real_output.',labwin);
            imag_output_unwrap = unwrapAugmentedTF_wAvg(imag_output.',labwin);
            complex_irmmask    = complex(real_output_unwrap,imag_output_unwrap);
            estimate           = complex_irmmask.*mix_stft;
        end

        %figure(1);
        %subplot(2,1,1);
        %view([0,90])
        %axis tight;
        %imagesc(real_output_unwrap);
        %xlabel('Time (seconds)');
        %ylabel('Frequency (Hz)');
        %set(gca,'YScale','log');
        %title('Real output unwrap');

        %subplot(2,1,2);
        %imagesc(imag_output_unwrap);
        %xlabel('Time (seconds)');
        %ylabel('Frequency (Hz)');
        %title('Imaginary output unwrap');
        %axis xy;

        denoise_sig  = overlapAndAdd(estimate,length(mix_sig),nfft,hopsize,params);

        % ----------------------- Compute Scores ----------------------------%
        f = (strsplit(filename,'.'));
        filename    = sprintf('%s%s_%s.%s',mix_wavs_data_path,char(f(1)),ENHANCED_PHRASE,char(f(2)) );
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

    %%%%%%%%%%%%%%%%%% MATLAB / PYTHON %%%%%%%%%%%%%%%%%%%%%%%%%
    % Matlab testing
    % save(sprintf('./scores/cIRMscores_denoising.noise%s.mat',noise_types{noise_ind}), 'scores_*');

    % Python testing
    save(sprintf('./scores/cIRMscores_denoising.noise%s%s.mat',noise_types{noise_ind},VERSION), 'scores_*');
    %%---------------- MATLAB / PYTHON ------------------------%%


%     save(sprintf('./scores/cIRMscores_denoising_tf.noise%smat',noise_types{noise_ind}), 'scores_*');



end

%%%%%%%%%%%%%%%%%% MATLAB / PYTHON %%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab testing
% save(sprintf('./scores/cIRMscores_denoising.noisesALL.mat'))

% Python testing
save(sprintf('./scores/cIRMscores_denoising.noisesALL%s.mat',VERSION))
%%---------------- MATLAB / PYTHON ------------------------%%

% save(sprintf('./scores/cIRMscores_denoising_tf.noisesALL.mat'))

