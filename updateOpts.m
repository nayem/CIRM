function opts = updatesOpts(opts, para)
 
    opts.feawin = para.feawin; % Length of windows, 5 frames in each side
    opts.labwin = para.labwin;
    opts.Fs = para.Fs;
    
    opts.winlen = para.winlen;%32e-3*opts.Fs;
    opts.overlap = para.overlap;%26e-3*para.Fs;%24e-3*para.Fs;
    opts.win_len = para.win_len;
    opts.overlap_len = para.overlap_len;
    opts.nfft = para.nfft;%32e-3*para.Fs;
    opts.hopsize = para.hopsize;
    opts.hop_size = para.hop_size;
    opts.labeltype = para.labeltype; %'realimag', 'complexIRM_realimag', 'fullcomplexIRM_realimag'
    opts.numGammatoneChans = para.numGammatoneChans;
    
    opts.fRange = para.fRange;
    opts.arma_order = para.arma_order;
    opts.noise  = para.noise ;
    opts.logistic_max = para.logistic_max;
    opts.logistic_steep = para.logistic_steep;  
    opts.clip_level = para.clip_level;
    opts.labcompress = para.labcompress;

end