clc; clear; close all;
addpath(genpath(pwd));
 
t     = 3;
Prob  = {'LinReg','LogReg','NCLogReg'};
m     = 128;
prob  = Prob{t};
k0    = [20 15 10 5 1];

switch prob 
    case 'LinReg' 
        n       = 100;
        var1    = 1/3;
        var2    = 1/3; 
    case {'LogReg','NCLogReg'}
        var1    = load('toxicity.mat').X; 
        var2    = load('toxicityclass.mat').y; 
        n       = size(var1,2);  
end  

[A,b,dim,n] = DataGeneration(prob,m,n,var1,var2); 
 
pars.optH   = 'gram'; 
for i       = 1:nnz(k0)   
    out0{i} = FedGiA(dim,n,A,b,k0(i),'LinReg',pars);
end
 
pars.optH   = 'diag';
for i       = 1:nnz(k0)
    out1{i} = FedGiA(dim,n,A,b,k0(i),'LinReg',pars);
end


% present the results
figure('Renderer', 'painters', 'Position',[800 400 700 330]);
axes('Position', [0.09 0.13 0.91 0.82] ); 
colors = {'#173f5f','#20639b','#3caea3','#f6d55c','#ed553b'}; 
for t = 1 : 2
    sub = subplot(1,2,t);               
    for i = 1:nnz(k0) 
        if t==1
           y = out0{i}.objx; plt=@plot;
        else
           y = out0{i}.errx; plt=@semilogy;
        end  
        plt(1:length(y),y, 'DisplayName',strcat('k_0=',num2str(k0(i))),...
            'LineWidth', 1.5, 'LineStyle', '-', 'Color',colors{i}); hold on 
    end
    for i = 1:nnz(k0)   
        if t==1
           y = out1{i}.objx(1:end-1); plt=@plot;
        else
           y = out1{i}.errx; plt=@semilogy;
        end  
        plt(1:length(y),y, 'DisplayName',strcat('k_0=',num2str(k0(i))),...
            'LineWidth', 1.5, 'LineStyle', ':', 'Color',colors{i}); hold on 
    end
    grid on, xlabel('Iterations'); 
    legend('NumColumns',2,'Location','NorthEast')
    if t==1
        ylabel('Objective') 
        set(sub, 'Position',[0.08,0.12,0.4,0.85] );
    else
       ylabel('Error')
       set(sub, 'Position',[0.58,0.12,0.4,0.85] );
    end
   
end
