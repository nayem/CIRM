function [x_r,x_i] = quanized_mask( mask, total_bin, type )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    if strcmpi(type, 'overall')
        x = mask(:);
    else
        bin = type;
        [r, c] = size(mask);
        n = round(r/total_bin);
        
        start_index = (bin-1)*n + 1;
        end_index = bin*n;
        if end_index>r
            end_index = r;
        end

        temp = mask(start_index:end_index,:);
        x = temp(:);  
    end
    
    x_r = real(x);
    x_i = imag(x);
end

