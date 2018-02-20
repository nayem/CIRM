# cIRM
Complex Ideal Ratio Mask

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

### File Annotation
Clean Speech (SSN Noise), for Train
```shell
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_clean_wavs_SSN_10noisespercs
      |---> training_16k
            |---> S_01_01_16k.wav
            |---> ... (total 500 files)
      |---> development_16k
            |---> S_51_01_16k.wav
            |---> ... (total 110 files)
      |---> testing_16k
            |---> S_62_02_16k.wav
            |---> ... (total 109 files)
```

Noisy Speech (SSN Noise), for Train
```shell
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoising_mix_wavs_SSN_10noisespercs
      |---> training_16k
            |---> S_01_01_16k_-3dB_noisyspeech.wav
            |---> S_01_01_16k_0dB_noisyspeech.wav
            |---> S_01_01_16k_3dB_noisyspeech.wav
            |---> ... (total 500x3=1500 files, [-3dB,0dB,3dB] noise level)
      |---> development_16k
            |---> S_51_01_16k_-3dB_noisyspeech.wav
            |---> S_51_01_16k_0dB_noisyspeech.wav
            |---> S_51_01_16k_3dB_noisyspeech.wav
            |---> ... (total 110x3=330 files, [-3dB,0dB,3dB] noise level)
      |---> testing_matched
            |---> S_62_02_16k_-3dB_noisyspeech.wav
            |---> S_62_02_16k_-6dB_noisyspeech.wav
            |---> S_62_02_16k_0dB_noisyspeech.wav
            |---> S_62_02_16k_3dB_noisyspeech.wav
            |---> S_62_02_16k_6dB_noisyspeech.wav
            |---> ... (total 109x3= 545 files, [-6dB,-3dB,0dB,3dB,6dB] noise level)
```

Generated Speech (SSN Noise), after Testing
```shell
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Data/denoise_complex_domain_wavs
      |---> S_62_02_16k_-3dB_noisyspeech_crmenh.wav
      |---> S_62_02_16k_-6dB_noisyspeech_crmenh.wav
      |---> S_62_02_16k_0dB_noisyspeech_crmenh.wav
      |---> S_62_02_16k_3dB_noisyspeech_crmenh.wav
      |---> S_62_02_16k_6dB_noisyspeech_crmenh.wav
      |---> ... (total 109x5= 545 files, [-6dB,-3dB,0dB,3dB,6dB] noise level)
```

**Model files**
```
/gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM/dnn_models
      |---> DNN_datas.mat (matlab->write, python->read)
      |---> DNN_params.mat (matlab->write, python->read)
      |---> dnncirm.noiseSSN.mat (matlab->write, Trained matlab Model-train)
      |---> DNN_net.mat (python->write, matlab->read, Trained python Model (intermediate)-train)
      |---> DNN_CIRM_net.mat (matlab->write, Trained Model (final)-train)
      
```



### scratch directory
```/N/dc2/scratch/knayem```
(scratch space is automatically deleted after 60 days so make sure to move files you care about to your home directory.)

```cd /N/dc2/scratch/knayem```

### Matlab Commands to run cIRM codes
```shell
qsub -I -l walltime=01:00:00 -l nodes=1:ppn=4 -l gres=ccm -q gpu
module add ccm
ccmlogin
cd /gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM
module add matlab/2016a
matlab

>> scriptTrainDNN_cIRM_denoise_mat('SSN')

>> scriptTestDNN_cIRM_denoise_mat()

>> cd PESQ
>> calculatePESQ_mat()
```

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
