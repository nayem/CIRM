function [y, fs]= wav2mat( current_path, file, save_path )
%WAV2MAT convert .wav file to .mat file
%   Return raw signal and frequency 
%
%   Nayem, Feb 13, 2018
%
    file_path = strcat(current_path,'/',file);
    [y,fs] = audioread(file_path);
%     sound(y,fs)
    
    % File name formate: 'S_01_01_16k.mat'
    wav_file = strsplit(file,'.');
    mat_file = strcat(save_path, wav_file(1),'.mat');
    save( char(mat_file) ,'y');
end

