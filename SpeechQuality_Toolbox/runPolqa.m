function runPolqa(inDir,inFs)
% This will calcualte POLQA for all of the 3GPP listening tests
% inDir should be the directory contain files to be measured
% inFs is sample rate the test files (eg. 8000, 16000)
%
% Error codes:
% -1; //Unspecified error, e.g. for access violations and other exceptions
% -2; //Input WAV files are not mono
% -3; //Input WAV files have not 8, 16 or 32 bit format per sample
% -4; //Pre-gain for deg signal has invalid value
% -5; //Unrecognized application type passed as argument
% -7; //Premature end of file reached, e.g. WAV header only without actual audio, or zero length file
% -8; //Error while reading file (usually file not found)
% -9; //Error while trying to write output file
% -10; //Received signal too quiet / silent
% -11; //Could not write delay profile to file
% -18; //Input files have sampling frequency below 8kHz or above 96kHz
% -19; //Heavily corrupted files (perhaps wrong files)
%  
% -21; //Error in Intel Integrated Performace Primitives Library function
% -24; //Internal error while reading input files
% -25; //Internal error in preprocessing module
% -26; //Internal error in SQuad08 algorithm time alignment module
% -27; //Internal error in SQuad08 algorithm code module
% -28; //Internal error in postprocessing module
% -29; //Internal error in POLQA algorithm
%  
% -30; //Less than 1 second of speech inside the file
% -31; //Less than 1 second of speech inside the file
% -32; //File length greater than 40 seconds
% -33; //File length greater than 40 seconds
% -35; //Internal resampling operation failed
%  
% -60; //Internal error in quality code reading/writing
%  
% -80; //Registry key check failed
% -99; //Execution speed too slow, premature end of calculations. Intermediate results may still be available in result messages. 
%

%%
tmpDir       = 'C:\tmp\polqa';
addDir(tmpDir);
procOutName  = [tmpDir '\procTmp.wav'];
refOutName   = [tmpDir '\refTmp.wav'];
finalOutName = [inDir '\polqaResults.xlsx'];
exe          = '"C:\Program Files\SwissQual\SQuadAnalyzer\SQuadAnalyzer.exe"';
%dLength=64;
%convTime=0;
testFs=48e3;

%Load reference file
refSeg       = [43 59];
refName      = 'P:\share\Test\TestSessions\aec\source\3GPP_DT_FB-NearEnd_Uplink.wav';
[ref, refFs] = audioread(refName);
ref = ref(1+refSeg(1)*refFs:refSeg(2)*refFs);

%Filter according to what band we're in
if inFs==8000
    mode = 0;
    ref  = applyMSIN(ref,refFs);
    ref  = applyLP35(ref,refFs);    
else
    mode = 1;
    ref  = applyLP7(ref,refFs);
    ref  = applyP341(ref,refFs);    
end

%Calculate score for each file
files        = dir([inDir '\*.wav']);
outRaw{1,1}  = 'Filename';
outRaw{1,2}  = 'Utterance';
outRaw{1,3}  = 'POLQA MOS';
outRaw{1,4}  = 'ASLR';
outRaw{1,5}  = 'NOIR';
outRaw{1,6}  = 'ASLD';
outRaw{1,7}  = 'NOID';
outRaw{1,8}  = 'startDel(ms)';
outRaw{1,9}  = 'delSpread(ms)';
outRaw{1,10} = 'delDeviation(ms)';
numPairs     = 2;
pairLength   = 8; %s
%convTime=16;

for kk=1:length(files)
    
    fprintf('\nFile: %s\n',files(kk).name);
    [proc, fs] = audioread([inDir '\' files(kk).name]);
    %proc= proc(:,1);
    %proc=proc(1+convTime*fs:(convTime+dLength)*fs);
    proc    = resample(proc,testFs,fs,24);
    outFile = [tmpDir '\resultsPOLQA.csv'];
    
    for jj=1:numPairs
        fprintf('Utterance: %d',jj);
        refTmp=ref(1+testFs*(jj-1)*pairLength:testFs*jj*pairLength);
        procTmp=proc(1+testFs*(jj-1)*pairLength:testFs*jj*pairLength);

        %write outputs and run polqa
        wavwrite(refTmp,testFs,refOutName);
        wavwrite(procTmp,testFs,procOutName);
        if kk==1
            cmd = [exe ' ' refOutName ' ' procOutName ' 1 ' num2str(mode) ' > ' outFile]; %First one is always one.  Second changes on WB/NB
        else
            cmd = [exe ' ' refOutName ' ' procOutName ' 1 ' num2str(mode) ' >> ' outFile]; %First one is always one.  Second changes on WB/NB
        end
        
        [status, res]=system(cmd); %status = -80 when no license
        
        if status ~= 0
            %error('Check SwissQual license');
            outRaw{kk+1,1}=files(kk).name;
            outRaw{kk+1,2}=['ErrorCode ' num2str(status)];     
            fprintf('  ErrorCode : %d\n',status);
            delete C:\tmp\polqa\resultsPOLQA.csv
        else    
            %read back in and select MOS value
            outRaw{1+(kk-1)*numPairs+jj,1}=files(kk).name;
            outRaw{1+(kk-1)*numPairs+jj,2}=num2str(jj);
            [~, textCell]=xlsread(outFile);
            s=textCell{1};

            %Find MOS
            ind=strfind(s,'MOS');
            mos=s(ind+6:ind+9);
            fprintf('  MOS: %s\n',mos);
            outRaw{1+(kk-1)*numPairs+jj,3}=str2num(mos);

            %Find ASLR speech level ref
            ind=strfind(s,'ASLR');
            aslr=s(ind+6:ind+11);
            outRaw{1+(kk-1)*numPairs+jj,4}=str2num(aslr);

            %Find NOIR noise level ref
            ind=strfind(s,'NOIR');
            noir=s(ind+6:ind+11);
            outRaw{1+(kk-1)*numPairs+jj,5}=str2num(noir);        

            %Find ASLD speech level device
            ind=strfind(s,'ASLD');
            asld=s(ind+6:ind+11);
            outRaw{1+(kk-1)*numPairs+jj,6}=str2num(asld);

            %Find NOID noise level device
            ind=strfind(s,'NOID');
            noid=s(ind+6:ind+11);
            outRaw{1+(kk-1)*numPairs+jj,7}=str2num(noid);  

            %Find Start Delay (ms)
            ind=strfind(s,'OFFS');
            isNum=0;
            indL=10;
            while isNum==0
                offsT=s(ind+6:ind+indL);
                offs=str2num(offsT);
                if ~isempty(offs)
                    isNum=1;
                else
                    indL=indL-1;
                    if indL==0
                        isNum=1;
                    end
                end
            end
            outRaw{1+(kk-1)*numPairs+jj,8}=offs;              
            
            %Find Delay Spread (ms)
            ind=strfind(s,'DSPR');
            dspr=s(ind+6:ind+10);
            isNum=0;
            indL=10;
            while isNum==0
                dsprT=s(ind+6:ind+indL);
                dspr=str2num(dsprT);
                if ~isempty(dspr)
                    isNum=1;
                else
                    indL=indL-1;
                    if indL==0
                        isNum=1;
                    end
                end
            end
            outRaw{1+(kk-1)*numPairs+jj,9}=dspr;
            
            %Find Delay Deviation (ms)
            ind=strfind(s,'DDEV');
            ddev=s(ind+6:ind+10); 
            isNum=0;
            indL=10;
            while isNum==0
                ddevT=s(ind+6:ind+indL);
                ddev=str2num(ddevT);
                if ~isempty(ddev)
                    isNum=1;
                else
                    indL=indL-1;
                    if indL==0
                        isNum=1;
                    end
                end
            end
            outRaw{1+(kk-1)*numPairs+jj,10}=ddev; 
            
            
            delete C:\tmp\polqa\resultsPOLQA.csv
        end    
    end
end
xlswrite(finalOutName,outRaw);

end %End runPolqaFor3GPP


%% Apply filters

% Apply MSIN filter (16k sample rate process)
function outFile = applyMSIN(inFile,fs)
    filename1 = tempname;
    filename2 = tempname;
    inFile = resample(inFile, 16000, fs, 24);
    tstUtilPcmWrite(inFile, filename1);
    cmd = 'P:\share\ITU\G.191\Software\stl2009\projects\Filter\Debug\Filter.exe';
    cmdParams = [' MSIN ' filename1 ' ' filename2]; 
    [stat,res] = system([cmd cmdParams]);
    if (stat ~= 0)
        error('Error encountered while running console app');
    end
    outFile = tstUtilPcmRead(filename2);
    outFile=resample(outFile,fs,16000,24);
end

% Apply P.341 filter (16k sample rate process)
function outFile = applyP341(inFile,fs)
    filename1 = tempname;
    filename2 = tempname;
    inFile = resample(inFile, 16000, fs, 24);
    tstUtilPcmWrite(inFile, filename1);
    cmd = 'P:\share\ITU\G.191\Software\stl2009\projects\Filter\Debug\Filter.exe';
    cmdParams = [' P341 ' filename1 ' ' filename2]; 
    [stat,res] = system([cmd cmdParams]);
    if (stat ~= 0)
        error('Error encountered while running console app');
    end
    outFile = tstUtilPcmRead(filename2);
    outFile=resample(outFile,fs,16000,24);
end

% Apply LP35 filter (48k sample rate process)
function outFile = applyLP35(inFile,fs)
    filename1 = tempname;
    filename2 = tempname;
    lpfilter=' LP35 ';
    inFile = resample(inFile, 48000, fs, 24);
    tstUtilPcmWrite(inFile, filename1);
    cmd = 'P:\share\ITU\G.191\Software\stl2009\projects\Filter\Debug\Filter.exe';
    cmdParams = [lpfilter filename1 ' ' filename2]; 
    [stat,res] = system([cmd cmdParams]);
    if (stat ~= 0)
        error('Error encountered while running console app');
    end
    outFile = tstUtilPcmRead(filename2);
end

% Apply LP75 filter (48k sample rate process)
function outFile = applyLP7(inFile,fs)
    filename1 = tempname;
    filename2 = tempname;
    lpfilter=' LP7 ';
    inFile = resample(inFile, 48000, fs, 24);
    tstUtilPcmWrite(inFile, filename1);
    cmd = 'P:\share\ITU\G.191\Software\stl2009\projects\Filter\Debug\Filter.exe';
    cmdParams = [lpfilter filename1 ' ' filename2]; 
    [stat,res] = system([cmd cmdParams]);
    if (stat ~= 0)
        error('Error encountered while running console app');
    end
    outFile = tstUtilPcmRead(filename2);
end

