function [mean_x,var_x,median_x,mode_x,max_x,min_x,quantiles_x]=stat_functs(x, axis, name, tag)
%   This function will evaluate the statistical functions.

    if axis == 0
        x=x(:);
        axis = 1;
    end
    
    mean_x=mean(x, axis);
    var_x=var(x, 0, axis);
    median_x=median(x, axis);
    mode_x=mode(x, axis);
    max_x=max(x, [], axis);
    min_x=min(x, [], axis);
    quantiles_x=quantile(x, 0:0.05:1 ,axis);

%     figure(1)
%     boxplot(x)
%     xlabel('Freq(Hz)')
%     ylabel('Time(secs)')
%     title(sprintf('BoxPlot of %s: %s part',name,tag ))
%         
%     figure(2)
%     histogram(x)
%     xlabel('Freq(Hz)')
%     ylabel('Time(secs)')
%     title(sprintf('Histogram of %s: %s part',name,tag ))
end
