function [x, HDR, byteOrder] = readSphere(filename, beflag)
% Use beflag to specify that data is encoded in big endian format
if nargin < 2
    beflag = 0;
end

fp = fopen(filename,'r');
[tmp1,tmp2,byteOrder]  = fopen(fp);

ftype = 'short';

fseek(fp,0,'bof');
% HDR = char(zeros(54,25));
% for i=1:54
%     X = (fscanf(fp,'%s',1));
%     HDR(i,1:length(X)) = X;
% end

i=1;
while (1)
    X = (fscanf(fp,'%s',1));
    
    if(ftell(fp)>1024)
        break;
    end
    
    HDR{i} = X;
    i=i+1;
end

% HDR = fread(fp,1024,'short');

fseek(fp,1024,'bof'); % skip the header
if ~beflag
    x  = fread(fp,inf,ftype);
else
    x = fread(fp,inf,ftype,'ieee-be');
end
fclose(fp); 