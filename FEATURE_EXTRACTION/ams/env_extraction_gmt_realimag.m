function env = env_extraction_gmt_realimag(SIG,choice,fs)

if nargin < 3
    fs = 16e3;
end
%%
if fs == 12e3
    R = 3;
else
    R = 4; %decimation factor, R times shorter
end


dSIG = SIG;

if strcmp(choice, 'abs')
    ENV = abs(dSIG);
elseif strcmp(choice, 'square')
%     ENV = abs(dSIG.^2);
    ENV = abs(dSIG.^2);
    
else
    printf('Unkownm envelope detection strategy\n');
    exit;
end

env = ENV;%decimate(ENV,R,'fir');



