function cep_mean= comp_cep(cleanData, enhdData, Fs)

% ----------------------------------------------------------------------
%          Cepstrum Distance Objective Speech Quality Measure
%
%   This function implements the cepstrum distance measure used
%   in [1]
%
%   Usage:  CEP=comp_cep(cleanData, enhdData, Fs)
%           
%         cleanData - clean data
%         enhdData  - enhanced data
%         Fs        - sampling rate
%         cep_mean  - computed cepstrum distance measure
% 
%         Note that the cepstrum measure is limited in the range [0, 10].
%
%  Example call:  CEP =comp_cep(cleanData, enhdData, Fs)
%
%  
%  References:
%
%     [1]	Kitawaki, N., Nagabuchi, H., and Itoh, K. (1988). Objective quality
%           evaluation for low bit-rate speech coding systems. IEEE J. Select.
%           Areas in Comm., 6(2), 262-273.
%
%  Author: Philipos C. Loizou 
%  (LPC routines were written by Bryan Pellom & John Hansen)
%  Modified by: Donald S. Williamson (July 2014)
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: #1 $  $Date: 2014/08/08 $

% ----------------------------------------------------------------------
if nargin~=3
    fprintf('USAGE: CEP=comp_cep(cleanData, enhdData, Fs)\n');
    fprintf('For more help, type: help comp_cep\n\n');
    return;
end

alpha=0.95;

if iscell(cleanData)
    num_elems = length(cleanData);
    cep_mean  = zeros(num_elems,1);
    
    for i = 1:num_elems
        IS_dist = cepstrum( cleanData{i} + eps, enhdData{i} + eps,Fs(i));
        IS_len  = round( length( IS_dist)* alpha);
        IS      = sort( IS_dist);
        
        cep_mean(i) = mean( IS( 1: IS_len)); 
    end
else
    len       = min( length( cleanData), length( enhdData));
    cleanData = cleanData( 1: len)+eps;
    enhdData  = enhdData( 1: len)+eps;
    
    IS_dist= cepstrum( cleanData, enhdData,Fs);
    
    IS_len= round( length( IS_dist)* alpha);
    IS= sort( IS_dist);
    
    cep_mean= mean( IS( 1: IS_len));
end



function distortion = cepstrum(clean_speech, processed_speech,sample_rate)


% ----------------------------------------------------------------------
% Check the length of the clean and processed speech.  Must be the same.
% ----------------------------------------------------------------------

clean_length      = length(clean_speech);
processed_length  = length(processed_speech);

if (clean_length ~= processed_length)
  disp('Error: Both Speech Files must be same length.');
  return
end

% ----------------------------------------------------------------------
% Scale both clean speech and processed speech to have same dynamic
% range.  Also remove DC component from each signal
% ----------------------------------------------------------------------

%clean_speech     = clean_speech     - mean(clean_speech);
%processed_speech = processed_speech - mean(processed_speech);

%processed_speech = processed_speech.*(max(abs(clean_speech))/ max(abs(processed_speech)));

% ----------------------------------------------------------------------
% Global Variables
% ----------------------------------------------------------------------

winlength   = round(30*sample_rate/1000); %240;		   % window length in samples
skiprate    = floor(winlength/4);		   % window skip in samples
if sample_rate<10000
   P           = 10;		   % LPC Analysis Order
else
    P=16;     % this could vary depending on sampling frequency.
end
C=10*sqrt(2)/log(10);
% ----------------------------------------------------------------------
% For each frame of input speech, calculate the Itakura-Saito Measure
% ----------------------------------------------------------------------

num_frames = clean_length/skiprate-(winlength/skiprate); % number of frames
start      = 1;					% starting sample
window     = 0.5*(1 - cos(2*pi*(1:winlength)'/(winlength+1)));

for frame_count = 1:num_frames

   % ----------------------------------------------------------
   % (1) Get the Frames for the test and reference speech. 
   %     Multiply by Hanning Window.
   % ----------------------------------------------------------

   clean_frame = clean_speech(start:start+winlength-1);
   processed_frame = processed_speech(start:start+winlength-1);
   clean_frame = clean_frame.*window;
   processed_frame = processed_frame.*window;

   % ----------------------------------------------------------
   % (2) Get the autocorrelation lags and LPC parameters used
   %     to compute the IS measure.
   % ----------------------------------------------------------

   [R_clean, Ref_clean, A_clean] = ...
      lpcoeff(clean_frame, P);
   [R_processed, Ref_processed, A_processed] = ...
      lpcoeff(processed_frame, P);

  C_clean=lpc2cep(A_clean);
  C_processed=lpc2cep(A_processed);
  
   % ----------------------------------------------------------
   % (3) Compute the cepstrum-distance measure
   % ----------------------------------------------------------

  
   distortion(frame_count) = min(10,C*norm(C_clean-C_processed,2)); 
   

   start = start + skiprate;

end



function [acorr, refcoeff, lpparams] = lpcoeff(speech_frame, model_order)

   % ----------------------------------------------------------
   % (1) Compute Autocorrelation Lags
   % ----------------------------------------------------------

   winlength = max(size(speech_frame));
   for k=1:model_order+1
      R(k) = sum(speech_frame(1:winlength-k+1) ...
		     .*speech_frame(k:winlength));
   end

   % ----------------------------------------------------------
   % (2) Levinson-Durbin
   % ----------------------------------------------------------

   a = ones(1,model_order);
   E(1)=R(1);
   for i=1:model_order
      a_past(1:i-1) = a(1:i-1);
      sum_term = sum(a_past(1:i-1).*R(i:-1:2));
      rcoeff(i)=(R(i+1) - sum_term) / E(i);
      a(i)=rcoeff(i);
      a(1:i-1) = a_past(1:i-1) - rcoeff(i).*a_past(i-1:-1:1);
      E(i+1)=(1-rcoeff(i)*rcoeff(i))*E(i);
   end

   acorr    = R;
   refcoeff = rcoeff;
   lpparams = [1 -a];

%----------------------------------------------
function [cep]=lpc2cep(a)
%
% converts prediction to cepstrum coefficients
%
% Author: Philipos C. Loizou

M=length(a);
cep=zeros(1,M-1);

cep(1)=-a(2);

for k=2:M-1
    ix=1:k-1;
    vec1=cep(ix).*a(k-1+1:-1:2).*ix;
    cep(k)=-(a(k+1)+sum(vec1)/k);
    
end



 

    

