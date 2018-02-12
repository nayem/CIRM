function llr_mean= comp_llr(cleanData, enhancedData,Fs)

% ----------------------------------------------------------------------
%
%      Log Likelihood Ratio (LLR) Objective Speech Quality Measure
%
%
%     This function implements the Log Likelihood Ratio Measure
%     defined on page 48 of [1] (see Equation 2.18).
%
%   Usage:  llr=comp_llr(cleanData, enhancedData, Fs)
%           
%         cleanData    - clean input data
%         enhancedData - enhanced output data
%         Fs           - sampling rate
%         llr          - computed likelihood ratio
%
%         Note that the LLR measure is limited in the range [0, 2].
%
%  Example call:  llr =comp_llr(sp04,enhanced,16000)
%
%
%  References:
%
%     [1] S. R. Quackenbush, T. P. Barnwell, and M. A. Clements,
%	    Objective Measures of Speech Quality.  Prentice Hall
%	    Advanced Reference Series, Englewood Cliffs, NJ, 1988,
%	    ISBN: 0-13-629056-6.
%
%  Authors: Bryan L. Pellom and John H. L. Hansen (July 1998)
%  Modified by: Philipos C. Loizou  (Oct 2006) - limited LLR to be in [0,2]
%               Donald S. Williamson (July 2014)
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: #1 $  $Date: 2014/08/08 $
% ----------------------------------------------------------------------

if nargin~=3
    fprintf('USAGE: LLR=comp_llr(cleanData, enhancedData, Fs)\n');
    fprintf('For more help, type: help comp_llr\n\n');
    return;
end

alpha=0.95;

if iscell(cleanData)
    num_elems = length(cleanData);
    llr_mean  = zeros(num_elems,1);
    
    for i = 1:num_elems
        
        IS_dist = llr( cleanData{i} + eps, enhancedData{i} + eps,Fs(i));
        IS_len  = round( length( IS_dist)* alpha);
        IS      = sort( IS_dist);
        
        llr_mean(i) = mean( IS( 1: IS_len));
    end
else
    len          = min( length( cleanData), length( enhancedData));
    cleanData    = cleanData( 1: len)+eps;
    enhancedData = enhancedData( 1: len)+eps;
    
    IS_dist= llr( cleanData, enhancedData,Fs);
    
    IS_len= round( length( IS_dist)* alpha);
    IS= sort( IS_dist);
    
    llr_mean= mean( IS( 1: IS_len));
end


function distortion = llr(clean_speech, processed_speech,sample_rate)


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
% Global Variables
% ----------------------------------------------------------------------

winlength   = round(30*sample_rate/1000); %240;		   % window length in samples
skiprate    = floor(winlength/4);		   % window skip in samples
if sample_rate<10000
   P           = 10;		   % LPC Analysis Order
else
    P=16;     % this could vary depending on sampling frequency.
end
% ----------------------------------------------------------------------
% For each frame of input speech, calculate the Log Likelihood Ratio 
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
   %     to compute the LLR measure.
   % ----------------------------------------------------------

   [R_clean, Ref_clean, A_clean] = ...
      lpcoeff(clean_frame, P);
   [R_processed, Ref_processed, A_processed] = ...
      lpcoeff(processed_frame, P);

   % ----------------------------------------------------------
   % (3) Compute the LLR measure
   % ----------------------------------------------------------

   numerator   = A_processed*toeplitz(R_clean)*A_processed';
   denominator = A_clean*toeplitz(R_clean)*A_clean';
   distortion(frame_count) = min(2,log(numerator/denominator));
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



