function y=logsum3(x,d1,d2,d3,k)
%LOGSUM logsum2(x,d,k)=log(sum(sum(k.*exp(x),d)))
%
% Inputs:  X(M,N,...) data matrix to sum
%          D          optional dimension to sum along [1st non-singular dimension]
%          K(M,N,...) optional scaling matrix. It must either be idential
%                     in size to X, or else be a vector whose length is
%                     equal to the size of dimension D of X
%
% Outputs: Y(1,N,...) = log(sum(sum(exp(X).*K,D)))
%
% This routine evaluates the given expression for Y but takes care to avoid
% overflow or underflow.
%
%      Copyright (C) Mike Brookes 1998
%      Version: $Id: logsum.m 3227 2013-07-04 15:42:04Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Modified by Donald S. Williamson - 9/10/2013
%

if nargin==1 || ~numel(d1)
    d1=[find(size(x)-1) 1];
    d1=d1(1);
end
n1=size(x,d1);
n2=size(x,d2);
n3=size(x,d3);
if n1<=1,            % use efficient computation if only one term in the sum
    if n2 <=1
        if n3 <=1
            if nargin<4
                y=x;
            else
                y=x+log(k);
            end
            return;
        end
    end
end
s=size(x);
p=[d1,d2,d3:ndims(x),1:d1-1];
z=reshape(permute(x,p),n1,n2,n3,prod(s)/(n1*n2*n3));
q=max(max(max(z,[],1),[],2),[],3);              % we subtract y from each row to avoid dynamic range problems
a=(q==Inf)|(q==-Inf);       % check for infinities
if nargin<5
    y=q+log(sum(sum(sum(exp(z-q(ones(n1,1),ones(1,n2),ones(1,n3),:)),1),2),3));
elseif numel(k)==n1 % Need to update for logsum2 and logsum3
    y=q+log(sum(exp(z-q(ones(n1,1),:)).*repmat(k(:),1,prod(s)/n1),1));
else % Need to update for logsum2 and logsum3
    y=q+log(sum(exp(z-q(ones(n1,1),:)).*reshape(permute(k,p),n1,prod(s)/n1),1));
end
y(a)=q(a);                  % correct any column whose max is +-Inf
s(d1)=1;                     % update the dimension of the summed component
s(d2)=2;
s(d3)=3;
% if(numel(y) ~= 1) % Need to update for logsum2
%     y=ipermute(reshape(y,s(p)),p);
% end