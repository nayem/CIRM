
addpath('./Misc/')

load('./dnn_models/results_e04v2_nf.mat');
%%
% Fs = 16E3
% So sampleing freq = 8E3
% Let average timestamp length, N = 150
% So freqs = Fs*(0:N/2)/N, there will be 76 freqs and highest will be 8E3
% There are 321 size freq dimension(row-wise) in the cIRM.
% 
% Now, we are taking 8bin(each contains 1E3), so 321/8=40.125 rows will be flatten in a single row to represent the freqs.
%%

Total_bins = 8;
colors=[ [1 0 0]; [0 1 0]; [0 0 1]; [1 1 0]; [1 0 1]; [0 1 1]; [0.85 0.33 0.1]; [.93 .69 .13] ];

%% plot cIRM
for i={'overall','freqbin'}
    
    sz=25;
    figure('Name',sprintf('Plot cIRM (Type: %s)',string(i)))
    
    subplot(2,2,1);
    if strcmpi(i,'overall')
        [reals_w,imags_w] = quanized_mask(est_cirm, Total_bins, i);
        ax1=scatter(reals_w, imags_w, sz, 'b','filled');
    else
        for j=1:Total_bins
            [reals_w,imags_w] = quanized_mask(est_cirm, Total_bins, j);
            scatter(reals_w, imags_w, sz, colors(j,:),'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.1);
            hold on
        end
    end
    hold off
    grid on
    xlabel('real')
    ylabel('imaginary')
    title('wrap cIRM')

    subplot(2,2,2);
    if strcmpi(i,'overall')
        [reals_uw,imags_uw] = quanized_mask(unwrap_est_cirm, Total_bins, i);
        ax2=scatter(reals_uw, imags_uw, sz, 'b','filled');
    else
        for j=1:Total_bins
            [reals_uw,imags_uw] = quanized_mask(unwrap_est_cirm, Total_bins, j);
            scatter(reals_uw, imags_uw, sz, colors(j,:),'filled','MarkerFaceAlpha',0.3,'MarkerEdgeAlpha',0.3);
            hold on
        end
    end
    hold off
    grid on
    xlabel('real')
    ylabel('imaginary')
    title('unwrap cIRM')

    subplot(2,2,3);
    if strcmpi(i,'overall')
        [reals_w_mat,imags_w_mat] = quanized_mask(est_cirm_mat, Total_bins, i);
        ax3=scatter(reals_w_mat, imags_w_mat, sz, 'b','filled');
    else
        for j=1:Total_bins
            [reals_w_mat,imags_w_mat] = quanized_mask(est_cirm_mat, Total_bins, j);
            scatter(reals_w_mat, imags_w_mat, sz, colors(j,:),'filled','MarkerFaceAlpha',0.3,'MarkerEdgeAlpha',0.3);
            hold on
        end
    end
    hold off
    grid on
    xlabel('real')
    ylabel('imaginary')
    title('wrap cIRM (matlab)')

    subplot(2,2,4);
    if strcmpi(i,'overall')
        [reals_uw_mat,imags_uw_mat] = quanized_mask(unwrap_est_cirm_mat, Total_bins, i);
        ax4=scatter(reals_uw_mat, imags_uw_mat, sz, 'b','filled');
    else
        for j=1:Total_bins
            [reals_uw_mat,imags_uw_mat] = quanized_mask(unwrap_est_cirm_mat, Total_bins, j);
            scatter(reals_uw_mat, imags_uw_mat, sz, colors(j,:),'filled','MarkerFaceAlpha',0.2,'MarkerEdgeAlpha',0.2);
            hold on
        end
    end
    hold off
    grid on
    xlabel('real')
    ylabel('imaginary')
    title('unwrap cIRM (matlab)')

end


%% plot STFT
for i=['overall','freqbin']
    
    sz=25;
    figure('Name',sprintf('Plot cIRM (Type: %s)',string(i)))
    
    subplot(2,2,1);
    if strcmpi(i,'overall')
        [reals_w,imags_w] = quanized_mask(gen_stft, Total_bins, i);
        ax1=scatter(reals_w, imags_w, sz, 'b','filled');
    else
        for j=1:Total_bins
            [reals_w,imags_w] = quanized_mask(gen_stft, Total_bins, j);
            scatter(reals_w, imags_w, sz, colors(j,:),'filled','MarkerFaceAlpha',0.3,'MarkerEdgeAlpha',0.3);
            hold on
        end
    end
    hold off
    grid on
    xlabel('real')
    ylabel('imaginary')
    title('Generated STFT')

    subplot(2,2,2);
    if strcmpi(i,'overall')
        [reals_uw,imags_uw] = quanized_mask(gen_stft_mat, Total_bins, i);
        ax2=scatter(reals_uw, imags_uw, sz, 'b','filled');
    else
        for j=1:Total_bins
            [reals_uw,imags_uw] = quanized_mask(gen_stft_mat, Total_bins, j);
            scatter(reals_uw, imags_uw, sz, colors(j,:),'filled','MarkerFaceAlpha',0.3,'MarkerEdgeAlpha',0.3);
            hold on
        end
    end
    hold off
    grid on
    xlabel('real')
    ylabel('imaginary')
    title('Generated STFT (matlab)')

    subplot(2,2,3);
    if strcmpi(i,'overall')
        [reals_w_mat,imags_w_mat] = quanized_mask(clean_stft, Total_bins, i);
        ax3=scatter(reals_w_mat, imags_w_mat, sz, 'b','filled');
    else
        for j=1:Total_bins
            [reals_w_mat,imags_w_mat] = quanized_mask(clean_stft, Total_bins, j);
            scatter(reals_w_mat, imags_w_mat, sz, colors(j,:),'filled','MarkerFaceAlpha',0.3,'MarkerEdgeAlpha',0.3);
            hold on
        end
    end
    hold off
    grid on
    xlabel('real')
    ylabel('imaginary')
    title('Clean STFT')

    subplot(2,2,4);
    if strcmpi(i,'overall')
        [reals_uw_mat,imags_uw_mat] = quanized_mask(mixture_stft, Total_bins, i);
        ax4=scatter(reals_uw_mat, imags_uw_mat, sz, 'b','filled');
    else
        for j=1:Total_bins
            [reals_uw_mat,imags_uw_mat] = quanized_mask(mixture_stft, Total_bins, j);
            scatter(reals_uw_mat, imags_uw_mat, sz, colors(j,:),'filled','MarkerFaceAlpha',0.2,'MarkerEdgeAlpha',0.2);
            hold on
        end
    end
    hold off
    grid on
    xlabel('real')
    ylabel('imaginary')
    title('Mixture STFT')

end