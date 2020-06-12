function Script_Quantized_Scores(QUANTIZED_PAIR_FILE, TAG)

    % QUANTIZED_PAIR_FILE = '/data/knayem/Quantization_data/clean_MMquant_full_path_pair_MMS_0.015625.csv'
    % QUANTIZED_PAIR_FILE = '/data/knayem/Quantization_data/clean_quant_full_path_pair_FS_0.0078125.csv'
    % TAG = 'FS_' OR 'MMS_'
    

    %% Load training data and generate features
    addpath('./complex_mask/')
    addpath('./dnn_mixphone/')
    addpath('./dnn_mixphone/costFunc/')
    addpath('./dnn_mixphone/main/')
    addpath('./dnn_mixphone/utility')
    addpath('./dnn_mixphone/debug/')

    addpath('./FEATURE_EXTRACTION/')
    addpath('./FEATURE_EXTRACTION/ams/')
    addpath('./RASTA_TOOLBOX/')
    addpath('./COCHLEAGRAM_TOOLBOX/')
    addpath('./GENERAL/')
    addpath('./PESQ/')
    addpath('./SpeechQuality_Toolbox/')
    addpath('./OVERLAP_ADD_SYNTHESIS/')

    warning('off','all')

    noise        = 'SSN';
    globalpara   = InitParams_cIRM(noise);
    globalpara %#ok<NOPTS>

    feawin          = globalpara.feawin;
    labwin          = globalpara.labwin;
    winlen          = globalpara.winlen;
    overlap         = globalpara.overlap;
    nfft            = globalpara.nfft;
    hopsize         = globalpara.hopsize;
    arma_order      = globalpara.arma_order;
    noise           = globalpara.noise;

    Fs       = 16e3;
    
    %% Format file name to store scores
    if nargin < 2
        TAG =   'FS_';
    end

    T = readtable(QUANTIZED_PAIR_FILE,'ReadVariableNames',false,'Format','%s%s');
    [r, c] = size(T);
    
    suffix = strsplit(QUANTIZED_PAIR_FILE, TAG);
    step_sz = strsplit(char(suffix(2)),'.csv');
    n = char(step_sz(1));
    SCORE_FILE = sprintf('./scores/quantizedScores_%s%s.mat',char(TAG),char(n))
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Process Reverberant/Clean Training Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    params.winlen = winlen; params.useGPU = 0; params.scalebywindow = 0;
    scores_denoise_fcIRM       = cell(r,1);

    tot_scores_denoise_fcIRM       = cell(1*r,1);
    
    ten_percent = ceil(0.1*r);
    fprintf('Processing %d Files ...\n\t',r )
    
    total_file_count = 1;
    
    for fileNum = 1:r

        cln_file = char(T{fileNum,1});
        rev_file = char(T{fileNum,1});
        derev_file = char(T{fileNum,2});
        
        clean_sig = audioread(cln_file);
        mix_sig = audioread(cln_file); % quantized signal
        denoise_sig = audioread(derev_file); % quantized signal
        
        scores_denoise_fcIRM{fileNum} = comp_dereverb_metrics(clean_sig,mix_sig,denoise_sig,Fs,rev_file,derev_file);
        tot_scores_denoise_fcIRM{total_file_count} = scores_denoise_fcIRM{fileNum};

        if(~mod(fileNum,ten_percent))
            fprintf('%d...',(fileNum/ten_percent)*10)
        end
        
        total_file_count = total_file_count + 1;
    end
    fprintf('\n')
    
    
    %save(sprintf('./scores/quantizedScores_FS_0.015625.mat'), 'scores_*')
    save(SCORE_FILE, 'scores_*')

end