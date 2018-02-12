function mva_data = meanVarArmaNormalize_Test(data,order,mu,std)
% mean/var norm + ARMA filtering

mv_data = meanVarNormalize_Test(data,mu,std);
mva_data = doARMA(mv_data,order);

end
