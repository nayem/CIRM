function wrapper_metrics(Version,noise_type,metrics)
% Calculate scores for different metrics
% GO to PESQ folder, then RUN
% Use ../scores/cIRMscores_denoising.noise<noise_type><Version>.mat
%
% Written by: Khandokar Md. Nayem, Jun 3, 2018

%     Version = '_e11v1';
%     noise_type = 'SSN';
    
    addpath('../GENERAL/')
    addpath('../scores/')

    if nargin < 3
        metrics = {'PESQ','OVRL','CCD','SRR_f','SNR','SNR_SEG','STOI','BSS_SDR','BSS_SIR','BSS_SAR','PHASE'};
    else
        metrics = {metrics};
    end

    total_scores_list = load(sprintf('../scores/cIRMscores_denoising.noise%s%s.mat',noise_type,Version));

    snr_list = [-3,-6,0,3,6];
    num_target = 109;
    num_noise = 1;
    
    %% Metric Calculate

    for m=1:length(metrics)
        scores = zeros(num_target,num_noise,length(snr_list));
        i=1;
        snr_list = [-3,-6,0,3,6];
        metric = metrics(m);
        scores_list = total_scores_list.scores_denoise_fcIRM;

        if strcmpi(metric,'PESQ')   % PESQ
            var = 'pesq_derev';
        elseif strcmpi(metric,'OVRL')   % OVRL
            var = 'Covl_derev';
        elseif strcmpi(metric,'CCD')    % CCD (Compute Ceptral Distortion)
            var = 'cd_derev';
        elseif strcmpi(metric,'SRR_f')  % SRR_f
            var = 'fwsrr_derev';
        elseif strcmpi(metric,'SNR')    % SNR
            var = 'snr_derev';
        elseif strcmpi(metric,'SNR_SEG')    % SNR_SEG
            var = 'segsnr_derev';
        elseif strcmpi(metric,'STOI')   % STOI
            var = 'stoi_derev';
        elseif strcmpi(metric,'BSS_SDR')   % BSS_SDR
            var = 'SDR_derev';
        elseif strcmpi(metric,'BSS_SIR')  % BSS_SIR
            var = 'SIR_derev';
        elseif strcmpi(metric,'BSS_SAR')  % BSS_SAR
            var = 'SAR_derev';
        elseif strcmpi(metric,'PHASE')   % PHASE
            var = 'PD';
        end

        for t=1:num_target
            for n=1:num_noise
                for s=1:length(snr_list)
                    scores(t,n,s)=scores_list{i}.(var); % careful about brackets, all are super important
                    i = i+1;
                end
            end
        end

        %% SNR List serialized, [-6,-3,0,3,6]

        mid = floor(length(snr_list)/2);
        snr_list = snr_list([flip(1:mid), (mid+1):length(snr_list)]);
        scores = scores(:,:,[flip(1:mid), (mid+1):length(snr_list)]);

        %% Avg scores
        avg_scores = computeAvgPesqBySnr(scores,snr_list,num_target,num_noise);
        
        fprintf('Metric: %s -->', metric{1});
        for s=1:length(snr_list)
            fprintf('Avg_scores[%ddB]:%f,', snr_list(s), avg_scores(s));
        end
        fprintf('\nTotal Avg_scores:%f\n\n', mean(avg_scores));

    end

end