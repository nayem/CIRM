# cIRM
Complex Ideal Ratio Mask

## BigRed 2
```bigred2.uits.iu.edu```

### Home directory 
```/gpfs/home/k/n/knayem/BigRed2```
(home directory on BigRed2 that has a maximum storage area of 100GB.)

```cd /gpfs/home/k/n/knayem/BigRed2```

### cIRM file directory 
```cd /gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM```

### scratch directory
```/N/dc2/scratch/knayem```
(scratch space is automatically deleted after 60 days so make sure to move files you care about to your home directory.)

```cd /N/dc2/scratch/knayem```

### Matlab Commands cIRM
```
qsub -I -l walltime=01:00:00 -l nodes=1:ppn=4 -l gres=ccm -q gpu
module add ccm
ccmlogin
cd /gpfs/home/k/n/knayem/BigRed2/Eagles_Backup/Code/cIRM/cIRM
module add matlab
matlab
```




## Module Commands
http://modules.sourceforge.net/man/module.html
