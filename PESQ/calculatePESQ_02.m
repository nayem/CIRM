function calculatePESQ_02( VERSION )
%CALCULATEPESQ Summary of this function goes here
%
%   Detailed explanation goes here
%   calculate PESQ score, how [bad,good] the speech is [-0.5,4.5]
%


    SERVER = 'Eagles'; % 'BigRed2'
    CODE = 'Matlab'; % 'Python'

    if strcmpi(CODE,'Python') == 1
        VERSION = strcat(VERSION,'_py');
    end

    if strcmpi(SERVER,'BigRed2') == 1
        path_data_directory = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data';
        path_code_directory = '/N/u/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM';

    elseif strcmpi(SERVER,'Eagles') == 1
        path_data_directory = '/data/knayem';
        path_code_directory = '/home/knayem/EaglesBigred2/cIRM';
    end


    cleanPath = strcat(path_data_directory,'/denoising_clean_wavs_SSN_10noisespercs/testing_16k/');
    cleanFiles     = dir(fullfile(cleanPath,'*.wav'));

    enhancedPath = strcat(path_data_directory,'/denoise_complex_domain_wavs',VERSION,'/');
    enhancedFiles     = dir(fullfile(enhancedPath,'*.wav'));

    pesq_scores = [];
    pesq_scores_0dB = [];
    pesq_scores_3dB =[];
    pesq_scores_n3dB = [];
    pesq_scores_6dB =[];
    pesq_scores_n6dB = [];


    ten_percent = ceil(0.1*length(cleanFiles));

    for i=1:length(cleanFiles)

        for t=1:length(enhancedFiles)
            % clean file-> S_62_02_16k.wav
            % enhanced file-> S_62_02_16k_-3dB_noisyspeech_crmenh.wav

            if( strncmpi(cleanFiles(i).name,enhancedFiles(t).name, 11)==1 )
                %fprintf('\n%s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                pesq_scores = [pesq_scores, pesq(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];
            end

            if( strncmpi(cleanFiles(i).name,enhancedFiles(t).name, 11)==1 )
                suffix = enhancedFiles(t).name(12:16);
                if( strcmp(suffix,'_0dB_') )
                    %fprintf('\n0dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_0dB = [pesq_scores_0dB, pesq(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];

                elseif( strcmp(suffix,'_3dB_') )
                    %fprintf('\n3dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_3dB = [pesq_scores_3dB, pesq(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];

                elseif( strcmp(suffix,'_-3dB') )
                    %fprintf('\n-3dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_n3dB = [pesq_scores_n3dB, pesq(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];

                elseif( strcmp(suffix,'_6dB_') )
                    %fprintf('\n6dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_6dB = [pesq_scores_6dB, pesq(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];

                elseif( strcmp(suffix,'_-6dB') )
                    %fprintf('\n-6dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_n6dB = [pesq_scores_n6dB, pesq(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];
                end

            end

        end

        if(~mod(i,ten_percent))
            fprintf('%d...',(i/ten_percent)*10);
        end
    end

    fprintf('\nAverage PESQ:%f', mean(pesq_scores))
    fprintf('\nAverage 0dB-PESQ:%f, 3dB-PESQ:%f, -3dB-PESQ:%f, 6dB-PESQ:%f, -6dB-PESQ:%f \n', ...
        mean(pesq_scores_0dB),mean(pesq_scores_3dB),mean(pesq_scores_n3dB),mean(pesq_scores_6dB),mean(pesq_scores_n6dB))

    fprintf('Total Average PESQ:%f\n', (mean(pesq_scores_0dB)+mean(pesq_scores_3dB)+ ...
        mean(pesq_scores_n3dB)+mean(pesq_scores_6dB)+mean(pesq_scores_n6dB))/5.0)

end

