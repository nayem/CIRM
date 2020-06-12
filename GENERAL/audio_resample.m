function audio_resample(SRC_AUDIO_PATH, DESC_AUDIO_PATH, RESAMPLE_FREQ)

% audio_resample()
% Summary of this function goes here
%
%   Resample audio wav files
    % SRC_AUDIO_PATH: /data/knayem/TIMIT_processed/train/
    % DESC_AUDIO_PATH: /data/knayem/TIMIT_clean_16k/train_16k/
%
%   Nayem, OCT 13, 2019


    file_list = dir( strcat(SRC_AUDIO_PATH,'*.wav'));

    for n = 1:length(file_list)
        target_path = strcat(SRC_AUDIO_PATH, file_list(n).name );
        fname = strsplit( file_list(n).name,'.');
        
        [Target, F_target] = audioread(target_path);
        Tnew = resample(Target, RESAMPLE_FREQ, F_target);
        
        if RESAMPLE_FREQ == 16e3
            New_file_name = sprintf('%s%s_%s.%s',DESC_AUDIO_PATH, fname{1},'16k',fname{2});
        else
            New_file_name = sprintf('%s%s_%s.%s',DESC_AUDIO_PATH, fname{1},'xxk',fname{2});
        end
        
        audiowrite(New_file_name, Tnew, RESAMPLE_FREQ);
      

        fprintf('SOURCE:%s --> DESC:%s\n', target_path, New_file_name);

    end
    
    fprintf('Total file: %f\n', n);
    fprintf('File regEx: %s, Read files: %f\n', strcat(SRC_AUDIO_PATH,'*.wav'), length(file_list));
    
end