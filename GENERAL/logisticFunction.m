function output = logisticFunction(input,c1,c2)
% logistic function
% c1 - functions maximum value (-c1 is the minimum value)
% c2 - steepness of the curve
output = 2*c1./(1 + exp(-c2*input)) - c1;
