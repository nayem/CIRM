function stat_analysis(MODE)
%   This function will evaluate the statistical analysis.
%
%
%
%

%% Variables
    SERVER = 'Eagles';

    if strcmpi(SERVER,'Eagles') == 1
        path_data_directory = '/data/knayem';
        path_code_directory = '/home/knayem/EaglesBigred2/cIRM';
    end
    
    switch lower(MODE)
        case 'train'
            STATS_FILE = fullfile(path_code_directory,'dnn_models','train_stats.mat');
        case 'dev'
            STATS_FILE = fullfile(path_code_directory,'dnn_models','dev_stats.mat');
        case 'test'
            STATS_FILE = fullfile(path_code_directory,'dnn_models','test_stats.mat');
        otherwise
            fprintf('Invalid MODE!\n');
            return
    end

%% load parameters
    fprintf('Loading statistic file ... ');
    load(STATS_FILE);
    fprintf('Complete! \n');
    
    for nth = 1:8
        switch nth
            case 1
                snr = clean;
                snr_struct(nth).name = 'clean';
            case 2
                snr = snr_n6;
                snr_struct(nth).name = '-6dB';
            case 3
                snr = snr_n3;
                snr_struct(nth).name = '-3dB';
            case 4
                snr = snr_0;
                snr_struct(nth).name = '0dB';
            case 5
                snr = snr_3;
                snr_struct(nth).name = '3dB';
            case 6
                snr = snr_6;
                snr_struct(nth).name = '6dB';
            case 7
                snr = snr_;
                snr_struct(nth).name = '_';
            case 8
                snr = [clean,snr_n6,snr_n3,snr_0,snr_3,snr_6,snr_];
                snr_struct(nth).name = 'all';
        end
        fprintf('Processing SNR %s ... \n',snr_struct(nth).name)
        
        r = real(snr);
        i = imag(snr);
        
        % Per-freq
        axis = 2;
        
        [snr_struct(nth).mean_r,snr_struct(nth).var_r,snr_struct(nth).median_r, ...
            snr_struct(nth).mode_r,snr_struct(nth).max_r,snr_struct(nth).min_r, ...
                snr_struct(nth).quantiles_r]=stat_functs(r,axis, snr_struct(nth).name,'Real');
        
        [snr_struct(nth).mean_i,snr_struct(nth).var_i,snr_struct(nth).median_i, ...
            snr_struct(nth).mode_i,snr_struct(nth).max_i,snr_struct(nth).min_i, ...
                snr_struct(nth).quantiles_i]=stat_functs(i,axis,snr_struct(nth).name,'Imag');
            
            
        % Overall
        axis = 0;
        
        [snr_struct(nth).oval_mean_r,snr_struct(nth).oval_var_r,snr_struct(nth).oval_median_r, ...
            snr_struct(nth).oval_mode_r,snr_struct(nth).oval_max_r,snr_struct(nth).oval_min_r, ...
                snr_struct(nth).oval_quantiles_r]=stat_functs(r,axis, snr_struct(nth).name,'Real');
        
        [snr_struct(nth).oval_mean_i,snr_struct(nth).oval_var_i,snr_struct(nth).oval_median_i, ...
            snr_struct(nth).oval_mode_i,snr_struct(nth).oval_max_i,snr_struct(nth).oval_min_i, ...
                snr_struct(nth).oval_quantiles_i]=stat_functs(i,axis,snr_struct(nth).name,'Imag');
    end

end
