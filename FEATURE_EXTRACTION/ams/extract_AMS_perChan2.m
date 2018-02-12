function [ns_ams] = extract_AMS_perChan2(sub_gf, nChnl, Srate, Nframes,winlen,overlaplen)

%% 
% Level Adjustment
% [x ratio]= LTLAdjust(x, Srate);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the envelope
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mix_env = env_extraction_gmt_chan3(sub_gf, 'abs',Srate); %time domain envelope in subbands

%% 
s_frame_len = (winlen/Srate)*1000; % 20 ms
hop_size    = winlen - overlaplen;

nFFT_env = winlen;%128; % Number of frames
nFFT_ams = winlen*2;%256; % FFT size

nFFT_speech = (s_frame_len/1000)*Srate; % in samples


ns_ams = zeros(1*15,Nframes);

parameters = AMS_init_FFT2(nFFT_env,nFFT_speech,nFFT_ams,nChnl,Srate);

win_ams    = window(@hann,winlen);
repwin_ams = repmat(win_ams,1,1);
start_samp = 1;
for frameNum=1:Nframes
    
    stop_samp   = start_samp + winlen - 1;
    mix_env_frm = mix_env(:,start_samp:stop_samp);
    ams         = abs(fft(mix_env_frm'.*repwin_ams,nFFT_ams));
    start_samp  = start_samp + hop_size;

	ams = parameters.MF_T*ams(1:nFFT_ams/2,:);
	ams = ams';
	ns_ams(:,frameNum) = reshape(ams,[],1);	
end

return;

