function [y, fs]= mat2wav( current_path, file, save_path )
%MAT2WAV convert .mat file to .wav file
%   Return raw signal and frequency
%
%   Nayem, Feb 13, 2018
%
    Fs = 16e3;
    file_path = strcat(current_path,'/',file);
    y = load(file_path,'y');
    y = y.y;

    % File name formate: 'S_01_01_16k.mat'
    mat_file = strsplit(file,'.');
    wav_file = strcat(save_path, mat_file(1),'.wav');
    audiowrite( char(wav_file),y,Fs );

%     clear y Fs
%     [y, Fs] = audioread(char(wav_file));
%     sound(y,Fs)
end