function [ns_ams] = extract_AMS_perChan(sub_gf, nChnl, freq)

if nargin < 3
    freq = 16000;
end

Srate = freq;
%% 
% Level Adjustment
% [x ratio]= LTLAdjust(x, Srate);

%% 
len = floor(6*Srate/1000); % 6ms, frame size in samples, envelope length
if rem(len,2)==1
    len = len+1; 
end
env_step    = 0.25; % 250 microseconds
len2        = floor(env_step*Srate/1000); 
Nframes     = floor(length(sub_gf)/len2);% -len/len2+1;
s_frame_len = 20; % 20 ms


nFFT_env = 128; % Number of frames
nFFT_ams = 256; % FFT size

nFFT_speech    = (s_frame_len/1000)*Srate; % in samples
AMS_frame_len  = s_frame_len/env_step; % 128 frames of envelope corresponding to 128*0.25 = 32ms
AMS_frame_step = AMS_frame_len/2; % step size

KK = floor(Nframes/AMS_frame_step);
ns_ams = zeros(1*15,KK);

parameters = AMS_init_FFT2(nFFT_env,nFFT_speech,nFFT_ams,nChnl,Srate);
% parameters_FB = AMS_init(nFFT_speech,64,nChnl,Srate);

mix_env = env_extraction_gmt_chan2(sub_gf, 'abs',Srate); %time domain envelope in subbands

win_ams    = window(@hann,AMS_frame_len);
repwin_ams = repmat(win_ams,1,1);
for frameNum=1:KK
    if frameNum == 1 %special treatment to the 1st frame, making it consistent with cochleagram and IBM calculation 
        mix_env_frm = mix_env(:,(1:AMS_frame_step)+(AMS_frame_step*(frameNum-1)));
        ams = abs(fft(mix_env_frm'.*repwin_ams((Srate/len2*s_frame_len/1000/2+1):end,:),nFFT_ams));        
    else
        mix_env_frm = mix_env(:,(1:AMS_frame_len)+(AMS_frame_step*(frameNum-2)));    
        ams = abs(fft(mix_env_frm'.*repwin_ams,nFFT_ams));
        
    end
	ams = parameters.MF_T*ams(1:nFFT_ams/2,:);
	ams = ams';
	ns_ams(:,frameNum) = reshape(ams,[],1);	
end

return;

