function W = initRandW_simple(nhid,nvis)
% 
% r  = 6 / sqrt(nhid+nvis+1);  
% W  = rand(nhid, nvis,'single') * 8 * r - 4*r;

W = 0.001*randn(nhid, nvis, 'single');
