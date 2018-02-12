function createNoisySpeech()
%createNoisySpeech() Summary of this function goes here
%
%   Create noisy wav files
%
%   Nayem, Sep 20, 2017

    % Train-> -3dB, 0dB, +3dB
    % Dev-> -3dB, 0dB, +3dB
    % Test-> -6dB, -3dB, 0dB, +3dB, +6dB
    SNR = 6;
    
    NOISY_PHRASE = 'noisyspeech';
    
    %% Open only 1 block at a time
%     % Create Noisy Training
%     Clean_Wav_Save_Path = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/training_16k/';
%     Noisy_Wav_Save_Path = '/data/knayem/denoising_mix_wavs_SSN_10noisespercs/training_16k/';
    % Do change to generateMixture() LINE 30: isFromFirst_2min = true;
    
    % Create Noisy Development
%     Clean_Wav_Save_Path = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/development_16k/';
%     Noisy_Wav_Save_Path = '/data/knayem/denoising_mix_wavs_SSN_10noisespercs/development_16k/';
    % Do change to generateMixture() LINE 30: isFromFirst_2min = true;
%     
%     % Create Noisy Testing
    Clean_Wav_Save_Path = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/testing_16k/';
    Noisy_Wav_Save_Path = '/data/knayem/denoising_mix_wavs_SSN_10noisespercs/testing_matched/';
    % Do change to generateMixture() LINE 30: isFromFirst_2min = false;
    %%
    
    Noise_Wav_Path = '/data/knayem/NOISES/';
    
    file_list = dir( strcat(Clean_Wav_Save_Path,'*.wav'));
    Noise_file_name = 'ssn.wav';
%     Noise_file_name = 'cafe_16k.wav';
%     Noise_file_name = 'babble_noise.wav';
%     Noise_file_name = 'factory.wav';

    %% Convert Noise File to 16kHz (desired freq)
    DesiredFrequency = 16e3;
    noise_file_path = sprintf('%s%s',Noise_Wav_Path,Noise_file_name);
    [Y,Fs] = audioread(noise_file_path);
%     [P,Q] = rat(DesiredFrequency / Fs);
%     Ynew = resample(Y,P,Q);
    Ynew = resample(Y,DesiredFrequency,Fs);
        
    fname = string(strsplit(Noise_file_name,'.'));
    Noise_file_name = sprintf('%s%s_%s.%s',Noise_Wav_Path, fname(1),'16k',fname(2));
    audiowrite(Noise_file_name, Ynew, DesiredFrequency);
    
    
    %%
    [Masker, F_masker] = audioread(Noise_file_name );
    
    for n = 1:length(file_list)
        target_path = strcat(Clean_Wav_Save_Path,file_list(n).name );
        [Target, F_target] = audioread(target_path);
        
        fprintf('target:%s,', target_path);
        [mixture,target,masker] = generateMixture(double(Target),double(Masker),SNR);
        
        fname = string(strsplit(file_list(n).name,'.'));
        noisy_file_name = sprintf('%s%s_%ddB_%s.%s',Noisy_Wav_Save_Path,fname(1),SNR,NOISY_PHRASE,fname(2));
        audiowrite(noisy_file_name, mixture./max(abs(mixture)), F_target);
        
%         mask_file_name = sprintf('%s%s_%ddB_mask.%s',Noisy_Wav_Save_Path,fname(1),SNR,fname(2));
%         audiowrite(mask_file_name, masker./max(abs(masker)), F_target);
%         
%         target_file_name = sprintf('%s%s_%ddB_target.%s',Noisy_Wav_Save_Path,fname(1),SNR,fname(2));
%         audiowrite(target_file_name, target./max(abs(target)), F_target);
        
%         10*log10(sum(target.^2/sum(masker.^2)))
%         
%         figure
%         subplot(3,1,1)
%         plot(target)
%         title('Speech')
%         
%         subplot(3,1,2)
%         plot(masker)
%         title('Masker')
%         
%         subplot(3,1,3)
%         plot(mixture)
%         title('Mixture')
        
    end
end