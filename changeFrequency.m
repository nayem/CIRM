function changeFrequency( )
%CHANGEFREQUENCY Summary of this function goes here
%
%   Change the frequency of the .wav
%   Use it to convert into 16kHz wav files
%   1. Convert Clean Test, Train, Dev .wav files
%
%   Nayem, Sep 20, 2017

    DesiredFrequency = 16e3;
    %% Open only 1 block at a time
%     % Convert Training->Training_16k
%     inputPath = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/training/';
%     outputPath = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/training_16k/';
    
    % Convert Development->Development_16k
%     inputPath = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/development/';
%     outputPath = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/development_16k/';
    
%     % Convert Testing->Testing_16k
    inputPath = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/testing/';
    outputPath = '/data/knayem/denoising_clean_wavs_SSN_10noisespercs/testing_16k/';
    %%
    
    inputFiles     = dir(fullfile(inputPath,'*.wav'));
    
    for i=1:length(inputFiles)
        [Y,Fs] = audioread(strcat(inputPath,inputFiles(i).name));
%         [P,Q] = rat(DesiredFrequency / Fs);
%         Ynew = resample(Y,P,Q);
        Ynew = resample(Y,DesiredFrequency,Fs);
        
        fname = string(strsplit(inputFiles(i).name,'.'));
        output_filename = sprintf('%s%s_%s.%s',outputPath, fname(1),'16k',fname(2));
        audiowrite(output_filename, Ynew, DesiredFrequency);
    end

end

