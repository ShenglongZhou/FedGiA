function  PlotObj(obj1)
    figure('Renderer', 'painters', 'Position',[1100 400 370 320]);
    axes('Position', [0.16 0.14 0.81 0.8] ); 
    h1 = plot(1:length(obj1),obj1(1:end)); 
    hold on, grid on
    h1.LineWidth  = 1.5;        
    h1.LineStyle  = '-';  
    h1.Color      = '#3caea3';  
    xlabel('Iterations'); ylabel('Objective');  
    legend('FedGiA')
end

