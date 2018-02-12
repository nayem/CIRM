% Description: test the speech quality assessment toolbox

%%
root_path   = '//fs2/users/dwilliamson/Reverberation/REVERB Data/LDC2013E110/LDC2013E110/REVERB_WSJCAM0_et/data';
num_files   = 3;
reverb_file = cell(num_files,1);
clean_file  = cell(num_files,1);

reverb_file{1} = '/far_test/primary_microphone/si_et_1/c30/c30c0203_ch1.wav'; % REVERB Challenge 2014- Sim data, evaluation set, 1ch, room3, far
clean_file{1}  = '/cln_test/primary_microphone/si_et_1/c30/c30c0203.wav';

reverb_file{2} = '/far_test/primary_microphone/si_et_1/c30/c30c020a_ch1.wav'; % REVERB Challenge 2014- Sim data, evaluation set, 1ch, room3, far
clean_file{2}  = '/cln_test/primary_microphone/si_et_1/c30/c30c020a.wav';

reverb_file{3} = '/far_test/primary_microphone/si_et_1/c30/c30c020c_ch1.wav'; % REVERB Challenge 2014- Sim data, evaluation set, 1ch, room3, far
clean_file{3}  = '/cln_test/primary_microphone/si_et_1/c30/c30c020c.wav';

ref_data   = cell(length(clean_file),1);
mix_data   = cell(length(clean_file),1);
noise_data = cell(length(clean_file),1);
proc       = cell(length(clean_file),1);

filtOrder   = 40;
ripple      = 30;
beta_kaiser = 4;
deltaHz     = 200;
fsWB        = 16e3;
fc_low      = fsWB/4 + deltaHz;
Wn_low      = (2/fsWB)*fc_low;
b_low       = fir1(filtOrder,Wn_low,'low',kaiser(filtOrder+1,beta_kaiser));

for i = 1:length(clean_file)
    
    [reverb_data,fs] = audioread(sprintf('%s%s',root_path,reverb_file{i}));
    clean_data       = audioread(sprintf('%s%s',root_path,clean_file{i}));
    
    if length(reverb_data) >= length(clean_data)
        reverb_data = reverb_data(1 : length(clean_data));
    elseif length(reverb_data) < length(clean_data)
        clean_data = clean_data(1 : length(reverb_data));
    end
    
    ref_data{i}   = clean_data;
    mix_data{i}   = reverb_data;
    noise_data{i} = reverb_data - clean_data;
    proc{i}       = filter(b_low,1,reverb_data);
    
end

%%  --------------------------- Compute Scores ----------------------------

resultsDir  = '\\fs\Public\dwilliamson';
uttLen      = [];
preambleLen = [];
labelName   = [];
bandWidth   = 'WB';

scores = sqaTool(ref_data,noise_data,mix_data,proc,fs,resultsDir,bandWidth);
