# cIRM
Complex Ideal Ratio Mask

## Eagles
```shell
ssh -Y knayem@eagles.soic.indiana.edu

bash

cd EaglesBigred2/cIRM
```

***Jupyter Notebook***
```shell
ssh -Y -L localhost:8896:localhost:8888 knayem@eagles.soic.indiana.edu

jupyter notebook

```
***Copy Server to Server***
To copy from BigRed2 to Eagles, from BigRed2
```scp -r <BigRed2_files>/ knayem@eagles.soic.indiana.edu:<Eagle_folder>/```

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
