function mixture_target_masker = generateMixture_v1_1(target,masker,snr, TASK,NUMS_OF_CUTS)

%
% Description: This function generates a mixture (target + noise) at a
% particular snr
%
% Inputs:
%   - target: clean speech signal
%
%   - masker: noise signal
%
%   - snr: signal-to-noise ratio
%
% Outputs:
%   - mixture: noisy speech at desired snr level
%
% Written by Donald S. Williamson, 10/31/2012
% Modified by Khandokar Nayem, Apr 7, 2018
%


%% Random Cut
% Nayem edit, Sep 14, 2017

% For Train, Dev -> isFromFirst_2min= true
% For Test -> isFromFirst_2min= false

% 3D matrix, [mixture,target,masker] are concated in 3rd dimention
mixture_target_masker = [;;];


cut_list = [];

% Make the target and masker files the same length
lt = length(target); lm = length(masker);

% fprintf('lt:%d, lm:%d\n', lt,lm);



if strcmpi(TASK,'TEST')
    % Take cuts from last 2 mins of the Noise file
    head = floor(lm/2)+1;
    tail = lm;
    len = tail-head+1;

    possible_cut_numbers = floor(len/lt);
    selected_cut_number = randperm(possible_cut_numbers,NUMS_OF_CUTS);

    for (sc = selected_cut_number)

        s = head + (sc-1)*lt + 1;
        t = head + sc*lt;
        % fprintf('%s target-Len:%d, mask-Len:%d, s:%d, t:%d, len:%d\n', TASK, lt,lm,s,t,t-s+1);

        cut_list = [cut_list; [s,t] ];
    end


elseif strcmpi(TASK,'TRAIN') || strcmpi(TASK, 'DEV')
    % Take cuts from first 2 mins of the Noise file
    head = 1;
    tail = floor(lm/2);
    len = tail-head+1;

    possible_cut_numbers = floor(len/lt);
    selected_cut_number = randperm(possible_cut_numbers,NUMS_OF_CUTS);


    for (sc = selected_cut_number)

        s = head + (sc-1)*lt;
        t = head + sc*lt - 1;
        % fprintf('%s target-Len:%d, mask-Len:%d, s:%d, t:%d, len:%d\n', TASK,lt,lm,s,t,t-s+1);

        cut_list = [cut_list; [s,t] ];
    end

end



%%


for c = [1:length(selected_cut_number)]
    s_t = cut_list(c,:);
    s = s_t(1);
    t = s_t(2);

    % fprintf('s: %d, t:%d\n', s,t);

    if (lt >= lm)       % equalize the lengths of the two files
        target_temp = target(1:lm);
    else
        masker_temp = masker(s:t);
        target_temp = target;
    end


    % Scale the masker so the mixture has the appropriate SNR

    change = 20*log10(std(target_temp)/std(masker_temp)) - snr;
    masker_temp = masker_temp*10^(change/20);     % scale masker to specified input SNR

    mixture = target_temp + masker_temp;

    mixture_target_masker=cat(3,mixture_target_masker, [mixture,target_temp,masker_temp]);
    fprintf('%d...',c );

end

fprintf('mixture_target_masker: ' );
disp(size(mixture_target_masker) );

end






