function scores = comp_dereverb_metrics(anech_sig,revb_sig,derevb_sig,Fs,rev_file,derev_file,normflag)

if nargin < 7
    normflag = 0;
end
    
%% -------- Initialize variables ----------- %
sil_sec           = zeros(1.5*Fs,1);
min_speech_length = 8; % minimum length of signal for PESQ (in seconds)
num_times_2_cat   = ceil((min_speech_length*Fs)/length(revb_sig));


%% ---------- Process signals for PESQ calculation ------------ %
cs  = [repmat([sil_sec;anech_sig],num_times_2_cat,1);sil_sec];
rs  = [repmat([sil_sec;revb_sig],num_times_2_cat,1);sil_sec];
drs = [repmat([sil_sec;derevb_sig],num_times_2_cat,1);sil_sec];


%% --------- Compute SRMR ------------------ %
try
    scores.srmr_rev = SRMR_main(rev_file); % SRMR
catch Error
    scores.srmr_rev = [];
end

try
    scores.srmr_derev = SRMR_main(derev_file); % SRMR
catch Error
    scores.srmr_derev = [];
end


%% ------------- Compute PESQ ----------------- %
scores.pesq_rev  = pesq_dat(cs,rs,Fs);  % PESQ
if isnan(scores.pesq_rev)
    scores.pesq_rev = [];
end
scores.pesq_derev  = pesq_dat(cs,drs,Fs); % PESQ
if isnan(scores.pesq_derev)
    scores.pesq_derev = [];
end

scores.pesqdir_rev  = pesq_dat(anech_sig,revb_sig,Fs);  % PESQ
if isnan(scores.pesqdir_rev)
    scores.pesqdir_rev = [];
end
scores.pesqdir_derev  = pesq_dat(anech_sig,derevb_sig,Fs); % PESQ
if isnan(scores.pesqdir_derev)
    scores.pesqdir_derev = [];
end

%% ------------------ Compute OVRL ------------------------ %
[Csig,Cbak,Covl]= composite(anech_sig, revb_sig, Fs);
scores.Covl_rev = Covl;
scores.Cbak_rev = Cbak;
scores.Csig_rev = Csig;

[Csig,Cbak,Covl]= composite(anech_sig, derevb_sig, Fs);
scores.Covl_derev = Covl;
scores.Cbak_derev = Cbak;
scores.Csig_derev = Csig;

%% ------------- Compute Cepstral Distortion ------------- %
scores.cd_rev   = comp_cep(anech_sig,revb_sig,Fs); % Cepstral distance
scores.cd_derev = comp_cep(anech_sig,derevb_sig,Fs); % Cepstral distance


%% ------------- Compute Frequency-weighted SRR -------------- %
scores.fwsrr_rev   = comp_fwseg(anech_sig, revb_sig,Fs);
scores.fwsrr_derev = comp_fwseg(anech_sig, derevb_sig,Fs);


%% -------------- Compute SNR and Seg. SNR -------------------- %
[scores.snr_rev,scores.segsnr_rev]     = comp_snr(anech_sig, revb_sig,Fs,0);
[scores.snr_derev,scores.segsnr_derev] = comp_snr(anech_sig, derevb_sig,Fs,normflag);

% scores.snralt_rev   = computeSNR(anech_sig,revb_sig);
% scores.snralt_derev = computeSNR(anech_sig,derevb_sig);

%% -------------- Compute STOI ------------------- %
scores.stoi_rev   = stoi(anech_sig,revb_sig,Fs);
scores.stoi_derev = stoi(anech_sig,derevb_sig,Fs);

% %% -------------- Compute MSE of STFT ---------------- %
winlen = 32e-3*Fs;
ovrlen = 24e-3*Fs;
nfft   = 32e-3*Fs;

anh_stft   = spectrogram(anech_sig,winlen,ovrlen,nfft,Fs);
rev_stft   = spectrogram(revb_sig,winlen,ovrlen,nfft,Fs);
derev_stft = spectrogram(derevb_sig,winlen,ovrlen,nfft,Fs);

% differ         = anh_stft - rev_stft;
% scores.msestft_rev = mean(mean(real(differ.*conj(differ)),2));
% 
% differ           = anh_stft - derev_stft;
% scores.msestft_derev = mean(mean(real(differ.*conj(differ)),2));

%% -------------- Compute MSE Phase Difference ---------------- %
anh_angle   = angle(anh_stft);
rev_angle   = angle(rev_stft);
derev_angle = angle(derev_stft);

% scores.msephase_rev   = mean(mean((anh_angle-rev_angle).^2));
% scores.msephase_derev = mean(mean((anh_angle-derev_angle).^2));

% %% -------------- Compute Phase Difference ---------------- %
% scores.diffphase_rev   = mean(mean(abs(anh_angle-rev_angle)));
% scores.diffphase_derev = mean(mean(abs(anh_angle-derev_angle)));

%% -------------- BSS Toolkit ------------------------------ %
%[xTarget, xeInterf, xeArtif]                     = bss_decomp_gain(revb_sig.', 1, [anech_sig (revb_sig-anech_sig)].');
%[scores.SDR_rev, scores.SIR_rev, scores.SAR_rev] = bss_crit(xTarget, xeInterf, xeArtif);

%[yTarget, yeInterf, yeArtif]                           = bss_decomp_gain(derevb_sig.', 1, [anech_sig (revb_sig-anech_sig)].');
%[scores.SDR_derev, scores.SIR_derev, scores.SAR_derev] = bss_crit(yTarget, yeInterf, yeArtif);


%[xeTarget, xeInterf, xeArtif]                     = bss_decomp_gain_SI(anech_sig, revb_sig.', 1, [anech_sig (revb_sig-anech_sig)].');
%[scores.SI_SDR_rev, scores.SI_SIR_rev, scores.SI_SAR_rev] = bss_crit_SI(xeTarget, xeInterf, xeArtif);

%[yeTarget, yeInterf, yeArtif]                           = bss_decomp_gain_SI(anech_sig, derevb_sig.', 1, [anech_sig (revb_sig-anech_sig)].');
%[scores.SI_SDR_derev, scores.SI_SIR_derev, scores.SI_SAR_derev] = bss_crit_SI(yeTarget, yeInterf, yeArtif);


%% ---------------- Phase-based Metrics ---------------------- %

phase_dev_rev    = rev_angle - anh_angle;
phase_dev_derev  = rev_angle - derev_angle;
scores.PD = mean(sum((cos(phase_dev_rev) - cos(phase_dev_derev)).^2,1));