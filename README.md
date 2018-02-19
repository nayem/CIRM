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
