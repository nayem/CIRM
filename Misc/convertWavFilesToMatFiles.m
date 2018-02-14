function convertWavFilesToMatFiles()
%convertWavFilesToMatFiles convert .wav files to .mat files
%
%   Nayem, Feb 13, 2018
%
    DATA_ROOT_PATH = '/Volumes/NAYEM/Eagles_Backup/Data/knayem/';
    
%% Clean .wavs folders
%     clean_input_path = 'denoising_clean_wavs_SSN_10noisespercs/';
%     clean_output_path = 'denoising_clean_mats_SSN_10noisespercs/';
%     
%     train_path = 'training_16k/';
%     dev_path = 'development_16k/';
%     test_path = 'testing_16k/';
% 
%     for path=[ string(train_path), string(dev_path), string(test_path) ]
%         wavs = dir( char(strcat(DATA_ROOT_PATH,clean_input_path,path, '*.wav')));
%         
%         for p = 1:length(wavs)
%             fprintf('Converting-> %s ... ',wavs(p).name);
%             
%             save_path = char(strcat(DATA_ROOT_PATH,clean_output_path,path)) ;
%             wav2mat(wavs(p).folder,wavs(p).name, save_path);
%             
%             fprintf('... Complete\n');
%         end
%     end

%% Mix .wavs folders
    mix_input_path = 'denoising_mix_wavs_SSN_10noisespercs/';
    mix_output_path = 'denoising_mix_mats_SSN_10noisespercs/';
    
    train_path = 'training_16k/';
    dev_path = 'development_16k/';
    test_path = 'testing_matched/';

    for path=[ string(train_path), string(dev_path), string(test_path) ]
        wavs = dir( char(strcat(DATA_ROOT_PATH,mix_input_path,path, '*.wav')));
        
        for p = 1:length(wavs)
            fprintf('Converting-> %s ... ',wavs(p).name);
            
            save_path = char(strcat(DATA_ROOT_PATH,mix_output_path,path)) ;
            wav2mat(wavs(p).folder,wavs(p).name, save_path);
            
            fprintf('... Complete\n');
        end
    end

end

