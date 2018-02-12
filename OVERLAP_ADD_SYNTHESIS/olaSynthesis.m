function x = olaSynthesis(STFT,FFT_LEN,HOP_SIZE,WINTYPE,mix,WIN_SIZE)
%
% Description: This function performs synthesis using the overlap and add
% approach
%
% Inputs:
% - STFT: Short-Time Fourier Transform
%
% - FFT_LEN: The SIZE of the FFT that was used to produce STFT
%               
% - HOP_SIZE: The HOP Size that was used to produce STFT
%
% - WINTYPE: Specifies the analysis window type (0-hamming,1-rect)
%
% - mix: original mixture
%
% - WIN_SIZE: Length of the window
%
% Outputs:
% - x: the synthesized signal
%
% References:
% Quatieri, Thomas. Discrete-Time Speech Signal Processing: Principles and
%       Practice. Prentice Hall PTR. 2002
%

% Determine if both the postive and negate frequencies are included in STFT
[numRows,numWindows] = size(STFT);

if(numRows ~= FFT_LEN)
    
    % Compute the complex conjugate of the STFT
    stft_conj = conj(STFT);
    
    % Flip the complex conjugate of the stft
    stft_conj_flip = stft_conj(FFT_LEN/2:-1:2,:);
    
    % Then the negative frequencies have not been included
    stft = [STFT;stft_conj_flip];
   
end

% Perform overlap and add method for each time window
x = zeros(FFT_LEN + (numWindows-1)*HOP_SIZE,1);
for startSamp = 0:HOP_SIZE:(numWindows-1)*HOP_SIZE
    
    winNum = 1+startSamp/HOP_SIZE;
    
    % Compute the Inverse FFT of each time window
    I_STFT = real(ifft(stft(:,winNum)));
    x(startSamp+1:(startSamp+FFT_LEN)) =...
        x(startSamp+1:(startSamp+FFT_LEN)) + I_STFT;
end

% Normalize by the type of analysis window
if(WINTYPE == 0)
    if nargin < 6
        win = hamming(HOP_SIZE*2); % Assumes 50% overlap was used
    else
        win = hamming(WIN_SIZE);
    end
    
else
    if(WINTYPE == 1)
        if nargin < 6
            win = hann(HOP_SIZE*2);
        else
            win = hamming(WIN_SIZE);
        end
    end
end

WIN = fft(win,FFT_LEN);
SCALING_FACT = HOP_SIZE/WIN(1);
x = SCALING_FACT*x;

if nargin >= 5
    
    y = zeros(length(mix),1);
    
    if(length(x) < length(mix))
        y(1:length(x)) = x;
    else
        y(:) = x(1:length(mix));
    end
    
    x = y;
    
end