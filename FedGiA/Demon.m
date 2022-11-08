clc; clear; close all;
addpath(genpath(pwd));

t     = 1;
prob  = {'LinReg','LogReg','NCLogReg'};
m     = 128;
n     = 100;   
switch prob{t}
    case 'LinReg'
        dim   = randi([50 150],1,m);
        d     = sum(dim);    
        [A,b] = GenerateData(m,n,dim,1/3,1/3); 
    otherwise
        dat   = load('toxicity.mat'); 
        lab   = load('toxicityclass.mat'); 
        lab.y(lab.y==-1)= 0;
        A     = Normalization(dat.X,3); 
        b     = lab.y;
        [d,n] = size(A);  
        I     = randperm(d);
        A     = A(I,:);  % randomize samples
        b     = b(I,:);  
        while 1
            idx = unique([randperm(d-2,m-1)+1 d]);
            dim  = idx-[0 idx(1:end-1)]; 
            if min(dim)>0.01*d/m; break; end
        end 
end

k0        = 10;
pars.r0   = 1; % increase this value if the solver diverges 

pars.optH = 'diag';
out1      = FedGiA(dim,n,A,b,k0,prob{t},pars);   
pt1       = PlotObj(out1.objx);
 
pars.optH = 'gram';
out2      = FedGiA(dim,n,A,b,k0,prob{t},pars);   
pt2       = PlotObj(out2.objx);