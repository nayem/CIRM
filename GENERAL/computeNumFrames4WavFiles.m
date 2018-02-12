function [totalFrames,numFramesPerSig] = computeNumFrames4WavFiles(wav_files,hop_size)

data_len = length(wav_files);
lenData  = zeros(data_len,1);

for wavNum = 1:data_len
    
    lenData(wavNum) = length(wav_files{wavNum});
end

numFramesPerSig  = floor(lenData./hop_size)-1;
totalFrames      = sum(numFramesPerSig);