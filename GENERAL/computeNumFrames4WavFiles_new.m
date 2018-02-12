function [totalFrames,numFramesPerSig] = computeNumFrames4WavFiles_new(wav_files,overlaplen,winlen)

data_len = length(wav_files);
lenData  = zeros(data_len,1);

totalFrames = 0;
numFramesPerSig = zeros(data_len,1);

for wavNum = 1:data_len
    
    lenData(wavNum) = length(wav_files{wavNum});
    T = single(fix((lenData(wavNum)-overlaplen)/(winlen-overlaplen))); % Number of time frames
    totalFrames = totalFrames + T;
    numFramesPerSig(wavNum) = T;
end

