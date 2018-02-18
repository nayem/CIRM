function convertMatFilesToWavFiles()
%convertWavFilesToMatFiles convert .wav files to .mat files
%
%   Nayem, Feb 13, 2018
%
    DATA_ROOT_PATH = '/Volumes/NAYEM/Eagles_Backup/Data/knayem/';

%% .mats folders
    mix_input_path = 'denoise_complex_domain_mats_BR2/';
    mix_output_path = 'denoise_complex_domain_wavs_BR2/';

    for path=[ string(mix_input_path) ]
        mats = dir( char(strcat(DATA_ROOT_PATH,path, '*.mat')));
        
        for p = 1:length(wavs)
            fprintf('Converting-> %s ... ',mats(p).name);
            
            save_path = char(strcat(DATA_ROOT_PATH,mix_output_path)) ;
            mat2wav(wavs(p).folder,wavs(p).name, save_path);
            
            fprintf('... Complete\n');
        end
    end

end

