function [ newmag ] = unwrapAugmentedSTFT(impt_mag,T,d,m)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

avgmag = zeros(d,m);

for k =T+1:m-T
    tmp               = impt_mag(:,k);
    chunck            = reshape(tmp,d,2*T+1);
    avgmag(:,k-T:k+T) = avgmag(:,k-T:k+T) + chunck;
end

avgmag = avgmag/(2*T+1);
newmag = avgmag;
newmag(newmag<0) = 0;

end

