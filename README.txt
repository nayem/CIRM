This is a readme for the cIRM toolbox. This toolbox contains files to train and test cIRM estimation
for speech denoising. The list of important files is below:

- scriptTrainDNN_cIRM_denoise.m: Script to train the DNN to estimate the cIRM. The input to
								 this file is a character string that specifies the noise. A path
								 to noisy and clean speech wav files is embedded in the code. This
								 will need to be changed for your own testing purposes. This file saves
								 a model for the DNN in a predefined location.
								 Ex. scriptTrainDNN_cIRM_denoise('SSN')
								 
								 
								 
- prepareTrainingData_cIRM_denoise.m: Called from scriptTrainDNN_cIRM_denoise. This function generates
								the features and labels (cIRM) for the training data.
								
- prepareDevData_cIRM_denoise.m: Called from scriptTrainDNN_cIRM_denoise. This function generates
								the features and labels (cIRM) for the testing data.
								
- InitParams_cIRM.m: Called from scriptTrainDNN_cIRM_denoise. Specifies the parameters for the features
					and labels
					
- InitializeNN_cIRM.m: Called from scriptTrainDNN_cIRM_denoise.  This file specifies the parameters for 
					the deep neural network (DNN).
					
- scriptTestDNN_cIRM_denoise.m: Script to test the DNN that estimates the cIRM. A string for the model that
					is saved in scriptTrainDNN_cIRM_denoise.m is hardcoded in this file. A path to noisy and clean speech
					wav files are embedded in the code. This will need to be changed for your own testing purposes. 
					This file also computes and saves PESQ, STOI, and etc. scores. USE THIS VERSION TO COMPUTE PESQ, IN ORDER
					TO MATCH PUBLISHED RESULTS. An output wav file for each testing	case is also generated. 
					
- denoising_clean_wavs_SSN_10noisespercs: folder that contains the clean wav files used for the development,
				testing, and training data sets.

- denoising_mix_wavs_SSN_10noisespercs: folder that contains the noisy speech wav files used for the development,
				testing, and training data sets. This data is for SSN at SNRs of -6, -3, 0, 3, and 6 dB.
		
- scores: folder where objective scores will be stored in *.mat file

- denoise_complex_domain_wavs: location where output processed wav files are stored


	