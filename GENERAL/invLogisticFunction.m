function output = invLogisticFunction(input,c1,c2)
% Inverse logistic function
% c1 - functions maximum value (-c1 is the minimum value)
% c2 - steepness of the curve

output = real(-log(complex((2*c1./(input + c1)) - 1))/c2);

output(isnan(output)) = 0;
output(isinf(output)) = 0;