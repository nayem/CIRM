function para = InitParams_cIRM(noise)
 
    para.feawin        = 2; % Length of windows, 5 frames in each side
    para.labwin        = 1;
    para.Fs            = 16e3;
    
    para.winlen        = 40e-3*para.Fs;%32e-3*para.Fs;
    para.overlap       = 20e-3*para.Fs;%26e-3*para.Fs;%24e-3*para.Fs;
    para.win_len       = para.winlen;
    para.overlap_len   = para.overlap;
    para.nfft          = para.winlen;%32e-3*para.Fs;
    para.hopsize       = para.winlen - para.overlap;
    para.hop_size      = para.hopsize;
    para.labeltype     = 'cIRM'; %'realimag', 'complexIRM_realimag', 'fullcomplexIRM_realimag'
    para.numGammatoneChans = 64;
    
    para.fRange         = [50, 8000];
    para.arma_order     = 2;
    para.noise          = noise;
    para.logistic_max   = 10;
    para.logistic_steep = 0.1;  
    para.clip_level     = 10;
    para.labcompress    = 'logistic';

end