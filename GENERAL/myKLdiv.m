function [ D_A_B ] = myKLdiv(A,B,type)
%
% Description: This function performs KL divergence
%


if(strcmp(type,'gen') == 1) % Generalized KL divergence
    
    output = A.*(log(A) - log(B)) - A + B;
    
    if(~isvector(A) && ismatrix(A))
        D_A_B = sum(sum(output,1),2);
    else
        D_A_B = sum(output);
    end
    
elseif(strcmp(type,'sym') == 1) % Symmetric KL divergence
    
    D_A_B = myKLdiv(A,B,'gen');
    D_B_A = myKLdiv(B,A,'gen');
    
    D_A_B = 0.5*(D_A_B + D_B_A);
    
end



end

