function ssn = CreateSSN( ctl )

fid = fopen(ctl);

idx = randperm(152);

topIdx = idx(1:100);
[white, f] = wavread('Noise/white.wav', 'native');
white = double(white);
white = resample(white, 16000, f);

ssn = zeros(size(white));

for i = 1 : 152
    fileRoot = fgetl(fid);
    if any(i == topIdx)
        sig = dlmread(fileRoot);
        partSSN = fftfilt(sig, white);
        ssn = ssn + partSSN;
    end
end

ssn = ssn / 100;

fclose(fid);