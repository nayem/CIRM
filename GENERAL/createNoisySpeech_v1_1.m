function createNoisySpeech_v1_1(NOISE,TASK, SNR)

% createNoisySpeech_v1_1() Summary of this function goes here
%
%   Create noisy wav files
%
%   Nayem, Apr 7, 2018

    % Train-> -3dB, 0dB, +3dB
    % Dev-> -3dB, 0dB, +3dB
    % Test-> -6dB, -3dB, 0dB, +3dB, +6dB

    %SNR = 6;

    NUMS_OF_CUTS = 10;

    NOISY_PHRASE = 'noisyspeech';

    % TASK = 'TRAIN';
    % TASK = 'DEV';
    % TASK = 'TEST';

    if strcmpi(TASK, 'TRAIN')
        % Create Noisy Training
        Clean_Wav_Save_Path = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_clean_wavs_SSN_10noisespercs/training_16k/';
        Noisy_Wav_Save_Path = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_mix_wavs_SSN_15000noisespercs/training_16k/';

    elseif strcmpi(TASK, 'DEV')
        % Create Noisy Development
        Clean_Wav_Save_Path = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_clean_wavs_SSN_10noisespercs/development_16k/';
        Noisy_Wav_Save_Path = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_mix_wavs_SSN_15000noisespercs/development_16k/';

    elseif strcmpi(TASK, 'TEST')
        % Create Noisy Testing

        Clean_Wav_Save_Path = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_clean_wavs_SSN_10noisespercs/testing_16k/';
        Noisy_Wav_Save_Path = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_mix_wavs_SSN_15000noisespercs/testing_matched/';
    end


    Noise_Wav_Path = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/NOISES/';


    file_list = dir( strcat(Clean_Wav_Save_Path,'*.wav'));

    if strcmpi(NOISE, 'SSN')
        Noise_file_name = 'ssn.wav';

    elseif strcmpi(NOISE, 'CAFE')
        Noise_file_name = 'cafe_16k.wav';

    elseif strcmpi(NOISE, 'BABBLE')
        Noise_file_name = 'babble_noise.wav';

    elseif strcmpi(NOISE, 'FACTORY')
        Noise_file_name = 'factory.wav';
    end


    %% Convert Noise File to 16kHz (desired freq)
    DesiredFrequency = 16e3;
    noise_file_path = sprintf('%s%s', Noise_Wav_Path, Noise_file_name);

    [Y,Fs] = audioread(noise_file_path);
    Ynew = resample(Y,DesiredFrequency,Fs);

    fname = strsplit( Noise_file_name,'.');
    Noise_file_name = sprintf('%s%s_%s.%s',Noise_Wav_Path, fname{1},'16k',fname{2});
    audiowrite(Noise_file_name, Ynew, DesiredFrequency);


    %%
    [Masker, F_masker] = audioread(Noise_file_name );

    for n = 1:length(file_list)
        target_path = strcat(Clean_Wav_Save_Path,file_list(n).name );
        [Target, F_target] = audioread(target_path);

        fprintf('target:%s\n', target_path);
        mixture_target_masker = generateMixture_v1_1(double(Target),double(Masker),SNR , TASK,NUMS_OF_CUTS);

        mtm_size = size(mixture_target_masker);

        for k = [1:mtm_size(3)]
            mixture = mixture_target_masker(:,1,k);
            target = mixture_target_masker(:,2,k);
            masker = mixture_target_masker(:,3,k);

            fname = (strsplit(file_list(n).name,'.'));
            noisy_file_name = sprintf('%s%s_%d_%ddB_%s.%s',Noisy_Wav_Save_Path,fname{1}, (k-1), SNR,NOISY_PHRASE,fname{2});

            fprintf('[%d,%d] -> noisy_file_name: %s\n', n, k-1, (noisy_file_name));
            audiowrite(noisy_file_name, mixture./max(abs(mixture)), F_target);

        end




    end
end