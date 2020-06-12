function createNoisySpeech_v4(NOISE, SNRS, TASK, SRC_PATH, DESC_PATH)

% createNoisySpeech_v2_1() Summary of this function goes here
%
%   Create noisy wav files
% Parameters:
%       <NOISE>     SSN, CAFE, BABBLE, FACTORY
%       <SNR>       [-6, -3, 0, 3, 6]
%       <TASK>      TRAIN, DEV, TEST
%       <SRC_PATH>  /data/knayem/TIMIT_clean_16k/dev_16k/
%                   /data/knayem/IEEE_male_clean_16k/train_16k/
%                   /data/knayem/IEEE_male_clean_16k/dev_16k/
%                   /data/knayem/IEEE_male_clean_16k/test_16k/
%       <DESC_PATH> /data/knayem/TIMIT_mixture/ssn/dev_16k/
%                   SSN
%                   /data/knayem/IEEE_male_mixture/ssn/train_16k/
%                   /data/knayem/IEEE_male_mixture/ssn/dev_16k/
%                   /data/knayem/IEEE_male_mixture/ssn/test_16k/
%                   CAFE
%                   /data/knayem/IEEE_male_mixture/cafe/train_16k/
%                   /data/knayem/IEEE_male_mixture/cafe/dev_16k/
%                   /data/knayem/IEEE_male_mixture/cafe/test_16k/
%                   BABBLE
%                   /data/knayem/IEEE_male_mixture/babble/train_16k/
%                   /data/knayem/IEEE_male_mixture/babble/dev_16k/
%                   /data/knayem/IEEE_male_mixture/babble/test_16k/
%                   FACTORY
%                   /data/knayem/IEEE_male_mixture/factory/train_16k/
%                   /data/knayem/IEEE_male_mixture/factory/dev_16k/
%                   /data/knayem/IEEE_male_mixture/factory/test_16k/
%   Nayem, Apr 7, 2020

    % Train-> -3dB, 0dB, +3dB
    % Dev-> -3dB, 0dB, +3dB
    % Test-> -6dB, -3dB, 0dB, +3dB, +6dB

    %SNR = 6;

    NUMS_OF_CUTS = 10;

    NOISY_PHRASE = 'noisyspeech';

    % TASK = 'TRAIN';
    % TASK = 'DEV';
    % TASK = 'TEST';

    % BigRed2 path
    % Base_path = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/';
    % Eagles path
    Base_path = '/data/';

    Noise_Wav_Path = fullfile(Base_path, 'NOISES/');


    if strcmpi(NOISE, 'SSN')
        Noise_file_name = 'ssn.wav';

    elseif strcmpi(NOISE, 'CAFE')
        Noise_file_name = 'cafe_16k.wav';

    elseif strcmpi(NOISE, 'BABBLE')
        Noise_file_name = 'babble.wav';

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

    file_list = dir( strcat(SRC_PATH,'*.wav'));
    
    for snr_indx = 1:length(SNRS)
    
        SNR = SNRS[snr_indx]

        for n = 1:length(file_list)
            target_path = strcat(SRC_PATH,file_list(n).name );
            [Target, F_target] = audioread(target_path);

            fprintf('target:%s\n', target_path);
            mixture_target_masker = generateMixture_v1_1(double(Target),double(Masker), SNR , TASK, NUMS_OF_CUTS);

            mtm_size = size(mixture_target_masker);

            for k = [1:mtm_size(3)]
                mixture = mixture_target_masker(:,1,k);
                target = mixture_target_masker(:,2,k);
                masker = mixture_target_masker(:,3,k);

                fname = (strsplit(file_list(n).name,'.'));
                noisy_file_name = sprintf('%s%s_%d_%ddB_%s_%s.%s',DESC_PATH,fname{1}, (k-1), SNR, upper(NOISE), NOISY_PHRASE,fname{2});

                fprintf('[%d,%d] -> noisy_file_name: %s\n', n, k-1, (noisy_file_name));
                %audiowrite(noisy_file_name, mixture./max(abs(mixture)), F_target);

            end

        end
    end
end

