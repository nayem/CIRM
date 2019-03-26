function [cIRM_r, cIRM_i, uncompress_cIRM_r, uncompress_cIRM_i] = cIRM(clean_speech,noisy_speech, cliplevel,labcompress,c1,c2)

if nargin < 3
    labcompress = 'logistic';
    c1 = 10;
    c2 = 0.1;
end

Y_r = real(noisy_speech);
Y_i = imag(noisy_speech);

S_r = real(clean_speech);
S_i = imag(clean_speech);

ns_pwspec = Y_r.^2 + Y_i.^2;

YrSr = Y_r.*S_r;
YiSi = Y_i.*S_i;
YrSi = Y_r.*S_i;
YiSr = Y_i.*S_r;

cIRM_r = (YrSr + YiSi)./ns_pwspec;
cIRM_i = (YrSi - YiSr)./ns_pwspec;

cIRM_r(ns_pwspec == 0) = 0;
cIRM_i(ns_pwspec == 0) = 0;

% nayem -> return uncompressed labels
uncompress_cIRM_r = cIRM_r;
uncompress_cIRM_i = cIRM_i;
%------------------------------------%

if strcmp(labcompress,'clip') == 1
    cIRM_r(cIRM_r > cliplevel) = cliplevel; cIRM_r(cIRM_r < -cliplevel) = -cliplevel;
    cIRM_i(cIRM_i > cliplevel) = cliplevel; cIRM_i(cIRM_i < -cliplevel) = -cliplevel;
elseif strcmp(labcompress,'logistic') == 1
    cIRM_r = logisticFunction(cIRM_r,c1,c2);
    cIRM_i = logisticFunction(cIRM_i,c1,c2);
end

