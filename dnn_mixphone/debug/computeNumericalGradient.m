function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
% theta = gather(theta);
numgrad = zeros(size(theta));

%%
EPSILON = 1e-5;
dim = length(theta);

for d = 1:dim
   ebasis = zeros(size(theta)); ebasis(d) =1;
   numgrad(d) = (J(theta+EPSILON*ebasis) - J(theta-EPSILON*ebasis)) / (2*EPSILON);    
end
end
