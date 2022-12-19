clc; clear; close all;
addpath(genpath(pwd));

t     = 1;
Prob  = {'LinReg','LogReg','NCLogReg'};
m     = 128;
prob  = Prob{t};

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
 
k0          = 10;
pars.r0     = 1;      % increase this value if the solver diverges  
out         = FedGiA(dim,n,A,b,k0,prob,pars);   
PlotObj(out.objx); 
