function calculatePESQ_mat()
%CALCULATEPESQ_MAT Summary of this function goes here
%
%   Detailed explanation goes here
%   Nayem, Feb 18, 2018
%
%   For mat files, calculate PESQ score, how [bad,good] the speech is [-0.5,4.5]
%
    cleanPath = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_clean_mats_SSN_10noisespercs/testing_16k/';
    cleanFiles     = dir(fullfile(cleanPath,'*.mat'));

    enhancedPath = '/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoise_complex_domain_mats_BR2/';
    enhancedFiles     = dir(fullfile(enhancedPath,'*.mat'));

    pesq_scores = [];
    pesq_scores_0dB = [];
    pesq_scores_3dB =[];
    pesq_scores_n3dB = [];
    pesq_scores_6dB =[];
    pesq_scores_n6dB = [];


    for i=1:length(cleanFiles)

        for t=1:length(enhancedFiles)
            % clean file-> S_62_02_16k.mat
            % enhanced file-> S_62_02_16k_-3dB_noisyspeech_crmenh.mat

            if( strncmpi(cleanFiles(i).name,enhancedFiles(t).name, 11)==1 )
                fprintf('\n%s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                pesq_scores = [pesq_scores, pesq_mat(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];
            end

            if( strncmpi(cleanFiles(i).name,enhancedFiles(t).name, 11)==1 )
                suffix = enhancedFiles(t).name(12:16);
                if( strcmp(suffix,'_0dB_') )
                    fprintf('\n0dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_0dB = [pesq_scores_0dB, pesq_mat(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];

                elseif( strcmp(suffix,'_3dB_') )
                    fprintf('\n3dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_3dB = [pesq_scores_3dB, pesq_mat(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];

                elseif( strcmp(suffix,'_-3dB') )
                    fprintf('\n-3dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_n3dB = [pesq_scores_n3dB, pesq_mat(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];

                elseif( strcmp(suffix,'_6dB_') )
                    fprintf('\n6dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_6dB = [pesq_scores_6dB, pesq_mat(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];

                elseif( strcmp(suffix,'_-6dB') )
                    fprintf('\n-6dB %s->%s,',cleanFiles(i).name,enhancedFiles(t).name);
                    pesq_scores_n6dB = [pesq_scores_n6dB, pesq_mat(strcat(cleanPath,cleanFiles(i).name), strcat(enhancedPath,enhancedFiles(t).name) )];
                end

            end

        end
%         pesq_scores = [pesq_scores, pesq_mat(strcat(cleanPath,clean_file(i)), strcat(enhancedPath,enhanced_file(i)) )];
    end

    fprintf('\nAverage PESQ:%f', mean(pesq_scores))
    fprintf('\nAverage 0dB-PESQ:%f, 3dB-PESQ:%f, -3dB-PESQ:%f, 6dB-PESQ:%f, -6dB-PESQ:%f \n', ...
        mean(pesq_scores_0dB),mean(pesq_scores_3dB),mean(pesq_scores_n3dB),mean(pesq_scores_6dB),mean(pesq_scores_n6dB))
    fprintf('Total Average PESQ:%f', (mean(pesq_scores_0dB)+mean(pesq_scores_3dB)+ ...
        mean(pesq_scores_n3dB)+mean(pesq_scores_6dB)+mean(pesq_scores_n6dB))/5.0)

end

