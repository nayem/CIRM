function [ scores ] = sqaTool(clean,noise_data,mix,processed,Fs,resultsDir,bandWidth,uttLen,preamble,labelname )
%Description: This function performs speech quality assessment of a
%             processed signal using various speech quality measures. The
%             speech quality measures include:
%               - PESQ, POLQA, segmental SNR, weighted spectral slope (WSS)
%               - log-likelihood ratio (LLR), Itakura-Saito distance (IS)
%               - cepstrum distance (CD), frequency-weighted
%                 segmental SNR (fwSegSnr), speech-to-reverberation
%                 modulation energy ratio (SRMR)
%               - TOSQA, 3QUEST, G.160, source-to-distortion ratio (SDR),
%                 source-to-inteferences ratio (SIR), sources-to-noise ratio
%                 (SNR), sources-to-artifacts ratio (SAR)
%               - short-time objective intelligibilty (STOI), composite
%
% Inputs:
%   - clean:      vector or cell array (with length L)of clean speech examples
%   - noise_data: vector or cell array (with length L) of noise examples
%   - mix:        vector or cell array (with length L) of noisy speech examples
%   - processed:  vector or cell array (with length L) of processed mixtures
%   - Fs:         sampling rate (integer)
%   - resultsDir: path of the folder where the scores will be stored. Cannot be a
%                   local path (would save the file on the server machine) or a file on \\fs2\users\<username> (server
%                   does not have write rights there) for 3Quest/TOSQA.
%   - bandWidth: string for the band flag ('NB','WB','Sw','FB')
%   - uttLen:    duration of each utterance. Typically for G160 mixes, 4sec (1 sec of noise only , 2 sec of
%                   speech+noise and 1 sec of noise only).
%   - preamble:  duration in seconds of the preamble of the output files to be discarded before the score computation
%   - labelname:   only for BRT use...
%
% Outputs:
%   - scores: structure containing the objective scores
%

if nargin < 8
    uttLen    = 4.0;
    preamble  = 16.0;
    labelname = getusername;
end

% Initialize variables
nsInd    = 2; % primary mic, noisy speech
refInd   = 5; % primary mix, clean speech
numChans = 7; % Number of channels in file

%% ------------------- Get reference signal data ------------------- %
if iscell(clean)
    
    num_elements = length(clean);
    ref   = cell(num_elements,1);
    refFs = zeros(num_elements,1);
    
    for i = 1:num_elements
        
        if (isa(clean{i},'char'))
            [ref{i},refFs(i)] = audioread(clean{i});
        elseif (isa(clean{i},'double') || isa(clean{i},'single'))
            ref{i}   = double(clean{i});
            refFs(i) = Fs;
        else
            error('Unsupported format for reference signal parameter');
        end
    end
    
elseif (isa(clean,'char'))
    
    [ref,refFs] = audioread(clean);
    
elseif (isa(clean,'double') || isa(clean,'single'))
    
    ref   = clean;
    refFs = Fs;
else
    error('Unsupported format for reference signal parameter');
end

%% ------------------- Get processed signal data ------------------- %
if iscell(processed)
    
    num_elements = length(processed);
    proc   = cell(num_elements,1);
    procFs = zeros(num_elements,1);
    
    for i = 1:num_elements
        
        if (isa(processed{i},'char'))
            [proc{i},procFs(i)] = audioread(processed{i});
        elseif (isa(processed{i},'double') || isa(processed{i},'single'))
            proc{i}   = double(processed{i});
            procFs(i) = Fs;
        else
            error('Unsupported format for reference signal parameter');
        end
    end
    
elseif (isa(processed,'char'))
    
    [proc, procFs] = audioread(processed);
    
elseif (isa(processed,'double') || isa(processed,'single'))
    
    proc   = processed;
    procFs = Fs;
else
    error('Unsupported format for processed signal parameter');
end


%% ------------------- Get noisy speech signal data ------------------- %

if iscell(mix)
    
    num_elements = length(mix);
    noisyspeech  = cell(num_elements,1);
    nsFs         = zeros(num_elements,1);
    
    for i = 1:num_elements
        
        if (isa(mix{i},'char'))
            [noisyspeech{i},nsFs(i)] = audioread(mix{i});
        elseif (isa(mix{i},'double') || isa(mix{i},'single'))
            noisyspeech{i} = double(mix{i});
            nsFs(i)        = Fs;
        else
            error('Unsupported format for reference signal parameter');
        end
    end
    
elseif (isa(mix,'char'))
    
    [noisyspeech,nsFs] = audioread(mix);
    
elseif (isa(mix,'double') || isa(mix,'single'))
    
    noisyspeech = mix;
    nsFs = Fs;
else
    error('Unsupported format for output parameter');
end

%% ------------------- Get noise signal data ------------------- %

if iscell(noise_data)
    
    num_elements = length(noise_data);
    noise        = cell(num_elements,1);
    noiseFs      = zeros(num_elements,1);
    
    for i = 1:num_elements
        
        if (isa(noise_data{i},'char'))
            [noise{i},noiseFs(i)] = audioread(noise_data{i});
        elseif (isa(noise_data{i},'double') || isa(noise_data{i},'single'))
            noise{i}   = double(noise_data{i});
            noiseFs(i) = Fs;
        else
            error('Unsupported format for reference signal parameter');
        end
    end
    
elseif (isa(noise_data,'char'))
    [noise,noiseFs] = audioread(noise_data);
elseif (isa(noise_data,'double') || isa(noise_data,'single'))
    noise   = noise_data;
    noiseFs = Fs;
else
    error('Unsupported format for output parameter');
end


%% -------------------- Account for signal mismatches ------------------- %

if ~iscell(ref) && ~iscell(proc) && ~iscell(noisyspeech) && ~iscell(noise)
    if (numel(unique([refFs,procFs,nsFs,noiseFs])) > 1)
        error('Files must have the same sampling rate');
    end
    
    len  = min([length(ref), length(proc),length(noisyspeech),length(noise)]);
    ref         = ref(1:len);
    proc        = proc(1:len);
    noisyspeech = noisyspeech(1:len);
    noise       = noise(1:len);
    
else
    num_elems = length(ref);
    
    for i = 1:num_elems
    
        ref1  = ref{i};
        proc1 = proc{i};
        ns    = noisyspeech{i};
        n     = noise{i};
        
        len  = min([length(ref1), length(proc1),length(ns),length(n)]);
        
        ref{i}         = ref1(1:len);
        proc{i}        = proc1(1:len);
        noisyspeech{i} = ns(1:len);
        noise{i}       = n(1:len);
    
    end
end

% ------------------- Perform Speech Quality Assessment ----------------- %


% %% Telecommunications objective speech quality assessment (TOSQA)
% if iscell(ref)
%     num_elems = length(ref);
%     structure = repmat(struct,num_elems,1);
%     
%     filepath = sprintf('%s\\temp_files',resultsDir);
%     if ~exist(filepath,'dir')
%         mkdir(filepath)
%     end
%     
%     for i = 1:num_elems
%         structure(i).cleanFile  = sprintf('%s\\inpath_3quest%d.wav',filepath,i);
%         audiowrite(structure(i).cleanFile, ref{i}, refFs(i));
%         
%         structure(i).procFile = sprintf('%s\\outpath_3quest%d.wav',filepath,i);
%         audiowrite(structure(i).procFile, proc{i}, procFs(i));
%     end
% else
%     filepath = sprintf('%s\\temp_files',resultsDir);
%     if ~exist(filepath,'dir')
%         mkdir(filepath)
%     end
%     
%     structure(1).cleanFile  = sprintf('%s\\inpath_3quest.wav',filepath);
%     audiowrite(structure(1).cleanFile ,ref,refFs);
%     structure(1).procFile = sprintf('%s\\outpath_3quest.wav',filepath);
%     audiowrite(structure(1).procFile,proc,procFs);
% end
% computeTosqa(structure,bandWidth,resultsDir)


% %% 3-fold Quality Evaluation of Speech in Telecommunications (3QUEST)
% 
% if iscell(ref)
%     num_elems = length(ref);
%     structure = repmat(struct,num_elems,1);
%     
%     filepath = sprintf('%s\\temp_files',resultsDir);
%     if ~exist(filepath,'dir')
%         mkdir(filepath)
%     end
%     
%     for i = 1:num_elems
%         len  = length(ref{i});
%         data = zeros(len,numChans);
%         data(:,nsInd) = noisyspeech{i};
%         data(:,refInd) = ref{i};
%         
%         structure(i).inPath  = sprintf('%s\\inpath_3quest%d.wav',filepath,i);
%         audiowrite(structure(i).inPath, data, refFs(i));
%         
%         structure(i).outPath = sprintf('%s\\outpath_3quest%d.wav',filepath,i);
%         audiowrite(structure(i).outPath, proc{i}, procFs(i));
%     end
% else
%     data           = zeros(len,numChans);
%     data(:,nsInd)  = noisyspeech;
%     data(:,refInd) = ref;
%     
%     filepath = sprintf('%s\\temp_files',resultsDir);
%     if ~exist(filepath,'dir')
%         mkdir(filepath)
%     end
%     
%     structure(1).inPath  = sprintf('%s\\inpath_3quest.wav',filepath);
%     audiowrite(structure(1).inPath ,data,refFs);
%     structure(1).outPath = sprintf('%s\\outpath_3quest.wav',filepath);
%     audiowrite(structure(1).outPath,proc,procFs);
% end
% compute3Quest(structure,bandWidth,resultsDir,uttLen,preamble,labelname);


%% PESQ
% if iscell(proc)
%     num_elems = length(proc);
%     scores.pesq = zeros(num_elems,1);
%
%     for i = 1:num_elems
%         scores.pesq(i) = tstMeasurePesq2(ref{i}, refFs(i), proc{i}, procFs(i));
%     end
% else
%     scores.pesq = tstMeasurePesq2(ref, refFs, proc, procFs);
% end

%% POLQA



%% Segmental SNR
[scores.snr, scores.segsnr] = comp_snr(ref, proc,refFs);


%% Weighted Spectral Slope (WSS)
scores.wss = comp_wss(ref,proc,refFs);



%% Log-likelihood ratio (LLR)
scores.llr = comp_llr(ref,proc,refFs);



%% Itakura-Saito distance (IS)
scores.is = comp_is(ref,proc,refFs);



%% Cepstrum distance (CD)
scores.cd = comp_cep(ref,proc,refFs);



%% Frequency-weighted Segmental SNR (fwSegSNR)
scores.fwSegSnr = comp_fwseg(ref,proc,refFs);



%% Speech-to-reverberation modulation energy ratio (SRMR)
if iscell(proc)
    num_elems = length(proc);
    srmr = zeros(num_elems,1);
    
    filepath = sprintf('%s\\temp_files',resultsDir);
    if ~exist(filepath,'dir')
        mkdir(filepath)
    end
    
    for i = 1:num_elems
        tempname    = sprintf('%s\\temp_srmr.wav',filepath);
        audiowrite(tempname,proc{i},procFs(i));
        srmr(i) = SRMR_main(tempname);
        delete(tempname)
    end
    scores.srmr = srmr;
else
    filepath = sprintf('%s\\temp_files',resultsDir);
    if ~exist(filepath,'dir')
        mkdir(filepath)
    end
    
    tempname = sprintf('%s\\temp_srmr.wav',filepath);
    audiowrite(tempname,proc,procFs);
    scores.srmr = SRMR_main(tempname);
    delete(tempname)
end

% %% G.160 Metrics
% if iscell(proc)
%     num_elems = length(proc);
%     scores.g160  = repmat(struct,num_elems,1);
%     
%     for i = 1:num_elems
%         scores.g160(i).r = tstMeasureAudNs2(ref{i}, noisyspeech{i}, proc{i}, refFs(i));
%     end
%     
% else
%     scores.g160.r = tstMeasureAudNs2(ref, noisyspeech, proc, refFs);
% end


%% Bark Spectral Distortion (BSD)



%% Kullback-Leibler Divergence



%% Log-Area Ratio (LAR)




%% BSS Evaluation Toolbox
if iscell(proc)
    num_elems = length(proc);
    bss_toolkit = repmat(struct,num_elems,1);
    
    for i = 1:num_elems
        S      = zeros(1,length(ref{i}));
        S(1,:) = ref{i}';
        N      = zeros(1,length(ref{i}));
        N(1,:) = noise{i}';
        [s_tar,e_int,e_nse,e_art] = bss_decomp_gain(proc{i}',1,S,N);
        [bss_toolkit(i).SDR,bss_toolkit(i).SIR,bss_toolkit(i).SNR,bss_toolkit(i).SAR] = bss_crit(s_tar,e_int,e_nse,e_art);
    end
    scores.bss_toolkit = bss_toolkit;
else
    S      = zeros(1,length(ref));
    S(1,:) = ref';
    N      = zeros(1,length(ref));
    N(1,:) = noise';
    [s_tar,e_int,e_nse,e_art] = bss_decomp_gain(proc',1,S,N);
    [scores.bss_toolkit.SDR,scores.bss_toolkit.SIR,scores.bss_toolkit.SNR,scores.bss_toolkit.SAR] = bss_crit(s_tar,e_int,e_nse,e_art);
end

%% Short-time objective intelligibility (STOI)
if iscell(proc)
    num_elems = length(proc);
    scores.stoi = zeros(num_elems,1);
    
    for i = 1:num_elems
        scores.stoi(i) = stoi(ref{i},proc{i},refFs(i));
    end
    
else
    scores.stoi = stoi(ref,proc,refFs);
end


%% Hearing Aid Speech Quality Index (HASQI)




%% Composite Measure (Hu and Loizou)

% if iscell(proc)
%     num_elems = length(proc);
%     scores.composite = repmat(struct,num_elems,1);
%
%     for i = 1:num_elems
%         [scores.composite(i).Csig,scores.composite(i).Cbak,scores.composite(i).Covl]= composite(ref{i},proc{i},refFs(i));
%     end
% else
%     [scores.composite.Csig,scores.composite.Cbak,scores.composite.Covl]= composite(ref,proc,refFs);
% end


%% Generate Excel file
excelFile = sprintf('%s\\speechqualityresults.xlsx',resultsDir);
obj_metrics = fieldnames(scores);
num_metrics = length(obj_metrics);
if iscell(proc)
    num_elems = length(proc);
    excel_data =  cell(1+num_elems,num_metrics+1);
else
    excel_data =  cell(1,num_metrics+1);
end
index = 2;
for metNum = 1:num_metrics
    metric = getfield(scores,obj_metrics{metNum});
    
    if ~isstruct(metric)
        excel_data(1,index) = obj_metrics(metNum);
        index = index + 1;
    else
        names = fieldnames(metric);
        num_fields = length(names);
        if strcmp(obj_metrics(metNum),'bss_toolkit')
            for fNum = 1:num_fields
                excel_data(1,index) = strcat('bss.',names(fNum));
                index = index + 1;
            end
        elseif strcmp(obj_metrics(metNum),'g160')
            metric = getfield(metric,'r');
            names = fieldnames(metric);
            num_fields = length(names);
            for fNum = 1:num_fields
                excel_data(1,index) = strcat('g160.',names(fNum));
                index = index + 1;
            end
        end
    end
end


if iscell(proc)
    
    num_elems = length(proc);
    
    for i = 1:num_elems
        index = 1;
        if isa(processed{i},'char')
            [~,file,~] = fileparts(processed{i});
            excel_data{1+i,index} = file;
            index = index + 1;
        else
            excel_data{1+i,index} = sprintf('Signal_%d',i);
            index = index + 1;
        end
        
        % Output the scores
        
        for metNum = 1:num_metrics
            metric = getfield(scores,obj_metrics{metNum});
            
            if ~isstruct(metric)
                excel_data{1+i,index} = metric(i);
                index = index + 1;
            else
                names = fieldnames(metric);
                num_fields = length(names);
                if strcmp(obj_metrics(metNum),'bss_toolkit')
                    names = fieldnames(metric(i));
                    num_fields = length(names);
                    for fNum = 1:num_fields
                        excel_data{1+i,index} = getfield(metric(i),names{fNum});
                        index = index + 1;
                    end
                elseif strcmp(obj_metrics(metNum),'g160')
                    metric = getfield(metric(i),'r');
                    names = fieldnames(metric);
                    num_fields = length(names);
                    for fNum = 1:num_fields
                        excel_data{1+i,index} = getfield(metric,names{fNum});
                        index = index + 1;
                    end
                end
            end
        end
    end
    
else
    
    index = 1;
    if isa(processed,'char')
        [~,file,~] = fileparts(processed);
        excel_data{2,index} = file;
        index = index + 1;
    else
        excel_data{2,index} = sprintf('Signal_%d',1);
        index = index + 1;
    end
    
    % Output the scores
    
    for metNum = 1:num_metrics
        metric = getfield(scores,obj_metrics{metNum});
        
        if ~isstruct(metric)
            excel_data{2,index} = metric(1);
            index = index + 1;
        else
            names = fieldnames(metric);
            num_fields = length(names);
            if strcmp(obj_metrics(metNum),'bss_toolkit')
                names = fieldnames(metric(1));
                num_fields = length(names);
                for fNum = 1:num_fields
                    excel_data{2,index} = getfield(metric(1),names{fNum});
                    index = index + 1;
                end
            elseif strcmp(obj_metrics(metNum),'g160')
                metric = getfield(metric(1),'r');
                names = fieldnames(metric);
                num_fields = length(names);
                for fNum = 1:num_fields
                    excel_data{2,index} = getfield(metric,names{fNum});
                    index = index + 1;
                end
            end
        end
    end
    
end

xlswrite(excelFile,excel_data)

end

