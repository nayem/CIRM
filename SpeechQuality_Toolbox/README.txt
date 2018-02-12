
    ************************************************************
									Speech Quality Assessment
                                               Version 1.0
   
                                              13 August, 2014   
                                       Donald S. Williamson
    ************************************************************


This document describes the speech quality assessment toolbox. The toolbox
computes objective measures on a processed speech signal. The toolbox includes
the following metrics: TOSQA, 3QUEST, PESQ, segmental SNR, weighted spectral slope (WSS),
log-likelihood ratio (LLR), Itakura-Saito distance (IS), cepstrum distance (CD), frequency-
weighted segmental SNR (fwSegSNR), speech-to-reverberation modulation energy ratio (SRMR), G.160,
BSS evaluation toolbox, short-time objective intelligibility (STOI), and a composite measure. The 
toolbox returns the scores in a structure and also generates an Excel file.

********************
Preliminaries

Before starting, 
  (1) Make sure that you have access to the Share_encrypted_bug11010 branch in Perforce
  (2) Files are located in the following location: 	  
		 P:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\
		

********************
Usage

The usage syntax is below: 

  [ scores ] = sqaTool(clean,noise_data,mix,processed,Fs,resultsDir,bandWidth,uttLen,preamble,labelname)

Inputs
	(1) clean:      vector or cell array (with length L) of clean speech examples
	(2) noise_data: vector or cell array (with length L) of noise examples
	(3) mix:        vector or cell array (with length L) of noisy speech examples
	(4) processed:  vector or cell array (with length L) of processed mixtures
	(5) Fs:         integer or array (with length L) of sampling rates
	(6) resultsDir: path of the folder where the scores will be stored. Cannot be a
                   local path (would save the file on the server machine) or a file on \\fs2\users\<username> (server
                   does not have write rights there) for 3Quest/TOSQA.
	(7) bandWidth: string for the band flag ('NB','WB','Sw','FB')
	(8) uttLen:    duration of each utterance. Typically for G160 mixes, 4sec (1 sec of noise only , 2 sec of
                   speech+noise and 1 sec of noise only).
	(9) preamble:  duration in seconds of the preamble of the output files to be discarded before the score computation
	(10) labelname:   only for BRT use...
            
Output
	(1) scores: structure containing the objective scores

If the data inputs (clean, noise_data,mix,processed) are cell arrays, then they must all have the same length, and the indices should correspond. Using cell arrays allows the computation of metrics for multiple files. The actual entries in the cell array may be the actual data or a path to the wavfile.

********************
Necessary Files

p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\README.txt
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\SRMR_main.p
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\bss_crit.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\bss_decomp_filt.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\bss_decomp_gain.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\bss_energy_ratios.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\bss_make_lags.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\bss_proj.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\cepsdist.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\cepsdist_unsync.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\comp_cep.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\comp_fwseg.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\comp_is.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\comp_llr.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\comp_snr.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\comp_wss.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\composite.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\realceps.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\sqaTool.m
p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\stoi.m

You must also have access to the 3QUEST and TOSQA code, located at:

P:\Share\MatlabUtils\Tools\Acqua\compute3Quest.m
P:\Share\MatlabUtils\Tools\Acqua\computeTosqa.m

********************
Example File

p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\SpeechQuality_Toolbox\tstSQAtoolbox.m

I also created a file p:\Share_encrypted_bug11010\Devel\Atlas\Matlab\tst\casaGrande\dwilliamson\processResults_3Quest.m, that creates an excel file from the 3Quest scores. This cannot be run until the 3Quest scores have been computed.
