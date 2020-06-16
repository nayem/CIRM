# Code Base (cIRM, Quantized)
cIRM = Complex Ideal Ratio Mask
Quantized = Quantization Spec

## Data
All the necessary data are in `\data\knayem`

### Backup
Backup are in `carbonate.uits.iu.edu`. 
Full path is `knayem@carbonate.uits.iu.edu:/N/u/knayem/Carbonate/Eagles_Backup/Data`

**File transfer**

To transfer a file (`file.txt`) to another server. You should be logged in to the console of sending server (from-server).
```shell
scp file.txt remote_username@10.10.0.2:/remote/directory/newfilename.txt
```

To transfer folder (`IEEE_DataFiles`) from a server (e.g. `eagles`) to another server (e.g. `carbonate`)
```shell
scp IEEE_DataFiles knayem@carbonate.uits.iu.edu:/N/u/knayem/Carbonate/Eagles_Backup/Data
```

### GPU check
```shell
nvidia-smi
```

## Eagles
```shell
ssh -Y knayem@eagles.soic.indiana.edu

bash
cd EaglesBigred2/cIRM
```

**For Matlab**
```shell
matlab
```

```matlab
scriptTrainDNN_cIRM_denoise_08('SSN')         
# SERVER = 'Eagles'; % 'BigRed2'
# VERSION = '_e10v1';

scriptTestDNN_cIRM_denoise_02()
# VERSION = '_e10v1';
# SERVER = 'Eagles'; % 'BigRed2'
# CODE = 'Matlab'; % 'Python'

calculatePESQ_02( VERSION )
SERVER = 'Eagles'; % 'BigRed2'
CODE = 'Matlab'; % 'Python'
```

**For Python**
```shell
matlab
```

```matlab
scriptTrainDNN_cIRM_denoise_08('SSN')         
# SERVER = 'Eagles'; % 'BigRed2'
# VERSION = '_e10v1';
```
```python
python DNN_01_8_v2.py
```
```matlab
cd dnn_models/
cIRM_Net_Change(VERSION)

scriptTestDNN_cIRM_denoise_02()
# VERSION = '_e10v1';
# SERVER = 'Eagles'; % 'BigRed2'
# CODE = 'Python'; % 'Matlab'

calculatePESQ_02( VERSION )
SERVER = 'Eagles'; % 'BigRed2'
CODE = 'Python'; % 'Matlab'
```



***Jupyter Notebook***
```shell
ssh -Y -L localhost:8896:localhost:8888 knayem@eagles.soic.indiana.edu

jupyter notebook

```
***Copy Server to Server***
To copy from BigRed2 to Eagles, from BigRed2,

```scp -r <BigRed2_files>/ knayem@eagles.soic.indiana.edu:<Eagle_folder>/```

Example:

```scp -r denoising_mix_wavs_SSN_15000noisespercs/ knayem@eagles.soic.indiana.edu:/data/knayem/```


### Generate Noisy Audio
Go to ```/home/knayem/EaglesBigRed2/cIRM/GENERAL/``` folder. Run ```createNoisySpeech_v2_1()```.

```shell
/home/knayem/EaglesBigRed2/cIRM/
      |---> GENERAL
```

Function ```createNoisySpeech_v2_1()``` prototype,
```
createNoisySpeech_v2_1(NOISE,TASK,SNR)

::parameter:: 
      NOISE : <str> case-insensitive; one of these 4 noises types 'SSN', 'CAFE', 'BABBLE', 'FACTORY'.
      TASK  : <str> case-insensitive; one of these 3 tasks types 'TRAIN', 'DEV', 'TEST'.
      SNR   : <int>; snr levels depends on task type. Train (-3,0,3), Dev (-3,0,3) and Test (-6,-3,0,3,-6).
      
::return::
      <none>
      Saved file name pattern 
      <filename1>_16k_<filename2>_<SNR>dB_<NOISE>_noisyspeech.wav
      S_72_09_16k_0_-3dB_FACTORY_noisyspeech.wav
      <filename1>=actual file name, <filename2>=serial of file name
```
By default, 10 audio (different cuts by adding various parts of a noise type) generated of an audio.

**Note:** Check the ```Clean_Wav_Save_Path``` and ```Noisy_Wav_Save_Path``` carefully. (Line 45-59) 









## BigRed 2
```bigred2.uits.iu.edu```

### Directory 
**Home Directory**
```shell
/gpfs/home/k/n/knayem/BigRed2
```
(home directory on BigRed2 that has a maximum storage area of 100GB.)

**Code Directory**
```shell
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM
```
**Data Directory**
```shell
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data
```

### Matlab Commands to run cIRM codes
_change files' path of `scriptTestDNN_cIRM_denoise_02()` and `calculatePESQ_02()` accordingly (matlab/python)_
```shell
qsub -I -l walltime=10:00:00 -l nodes=1:ppn=4 -l gres=ccm -q gpu
module add ccm
ccmlogin
cd /gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM
module add matlab/2016a
matlab

>> scriptTrainDNN_cIRM_denoise_02('SSN')

>> scriptTestDNN_cIRM_denoise_02()

>> cd PESQ
>> calculatePESQ_02()
```

### Python Commands to run cIRM codes
_change files' path of `scriptTestDNN_cIRM_denoise_02()` and `calculatePESQ_02()` accordingly (matlab/python)_
```shell
qsub -I -l walltime=10:00:00 -l nodes=1:ppn=4 -l gres=ccm -q gpu
module add ccm
ccmlogin
cd /gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM
module add tensorflow
module add anaconda2

python DNN_01.py

module add matlab/2016a
matlab

>> cd dnn_models
>> cIRM_Net_Change()

>> cd ..
>> scriptTestDNN_cIRM_denoise_02()

>> cd PESQ
>> calculatePESQ_02()
```

### File Annotation
Clean Speech (SSN Noise), for Train+Development+Test
```shell
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_clean_wavs_SSN_10noisespercs
      |---> training_16k
            |---> S_01_01_16k.wav
            |
            |---> ... (total 500 files)
      |---> development_16k
            |---> S_51_01_16k.wav
            |
            |---> ... (total 110 files)
      |---> testing_16k
            |---> S_62_02_16k.wav
            |
            |---> ... (total 109 files)
```

Mix/Noisy Speech (SSN Noise), for Train+Development+Test
```shell
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_mix_wavs_SSN_10noisespercs
      |---> training_16k
            |---> S_01_01_16k_-3dB_noisyspeech.wav
            |---> S_01_01_16k_0dB_noisyspeech.wav
            |---> S_01_01_16k_3dB_noisyspeech.wav
            |
            |---> ... (total 500x3=1500 files, [-3dB,0dB,3dB] noise level)
      |---> development_16k
            |---> S_51_01_16k_-3dB_noisyspeech.wav
            |---> S_51_01_16k_0dB_noisyspeech.wav
            |---> S_51_01_16k_3dB_noisyspeech.wav
            |
            |---> ... (total 110x3=330 files, [-3dB,0dB,3dB] noise level)
      |---> testing_matched
            |---> S_62_02_16k_-3dB_noisyspeech.wav
            |---> S_62_02_16k_-6dB_noisyspeech.wav
            |---> S_62_02_16k_0dB_noisyspeech.wav
            |---> S_62_02_16k_3dB_noisyspeech.wav
            |---> S_62_02_16k_6dB_noisyspeech.wav
            |
            |---> ... (total 109x3= 545 files, [-6dB,-3dB,0dB,3dB,6dB] noise level)
```

Enhanced/Generated Speech (SSN Noise), from Testing
```shell
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoise_complex_domain_wavs
      |---> S_62_02_16k_-3dB_noisyspeech_crmenh.wav
      |---> S_62_02_16k_-6dB_noisyspeech_crmenh.wav
      |---> S_62_02_16k_0dB_noisyspeech_crmenh.wav
      |---> S_62_02_16k_3dB_noisyspeech_crmenh.wav
      |---> S_62_02_16k_6dB_noisyspeech_crmenh.wav
      |
      |---> ... (total 109x5= 545 files, [-6dB,-3dB,0dB,3dB,6dB] noise level)
```

**Model files**
```shell
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM/dnn_models
      |---> dnncirm.noiseSSN.mat (Trained matlab Model)
            (MATLAB: scriptTrainDNN_cIRM_denoise('SSN')-> write )
      |---> dnncirm.noiseSSN_02.mat 
      |
      |---> DNN_datas.mat 
            (MATLAB: scriptTrainDNN_cIRM_denoise_mat('SSN')-> write)
            (PYTHON: DNN_01.py -> read)
            (MATLAB: cIRM_Net_Change()-> read)
      |---> DNN_params.mat
            (MATLAB: scriptTrainDNN_cIRM_denoise_mat('SSN')-> write)
            (PYTHON: DNN_01.py -> read)
            (MATLAB: cIRM_Net_Change()-> read)
      |
      |---> DNN_net.mat (Trained python Model [intermediate])
            (PYTHON: DNN_01.py -> write)
            (MATLAB: cIRM_Net_Change()-> read)
      |---> DNN_net_02.mat
      |
      |---> DNN_CIRM_net.mat (Trained python Model [final])
            (MATLAB: cIRM_Net_Change()-> write)
            (MATLAB: scriptTestDNN_cIRM_denoise()-> read)
      |---> DNN_CIRM_net_02.mat
      
```



### scratch directory
```/N/dc2/scratch/knayem```
(scratch space is automatically deleted after 60 days so make sure to move files you care about to your home directory.)

```cd /N/dc2/scratch/knayem```


### Jupyter at BigRed2
Check at your pc if ```tcp:8895``` is free or not. If not free (e.g. process ```<PID1>``` is running), then kill it.
```shell
lsof -i tcp:8895
kill -9 <PID1>
ssh -N -f -L localhost:8895:localhost:8895 knayem@bigred2.uits.iu.edu
ssh knayem@bigred2.uits.iu.edu
```
Run at the server,
```
lsof -i tcp:8895
kill -9 <PID2>
jupyter notebook --no-browser --port=8895
```


## Module Commands
http://modules.sourceforge.net/man/module.html

**Tip:** GitHub README Basic writing and formatting syntax, https://help.github.com/articles/basic-writing-and-formatting-syntax/
