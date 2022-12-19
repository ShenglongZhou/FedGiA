function out = FedGiA(dim,n,A,b,k0,prob,pars)
% This solver solves an optimization problem in the following form:
%
%         min_{xi,x}  (1/m) sum_{i=1}^m fi(xi) 
%            s.t.       xi = x, i=1,2,...,m
%
% where xi\in\R^n and x\in\R^
%
% ------------------------- Linear regression (LinReg)---------------------
% For every i = 1,2,...,m
%
%           fi(xi) = (1/2/di)||Ai*xi-bi||^2   
%
% where (Ai,bi) is the data for node/client i
%       Ai\in\R^{di-by-n} the measurement matrix
%       bi\in\R^{di-by-1} the observation vector 
%
% -------------------- Logistic regression (LogReg)------------------------
% For every i = 1,2,...,m
%
%     fi(xi) = (1/di)*sum_{j=1}^{di}[ ln(1+exp(<ai_j,xi>)-bi_j*<ai_j,xi> ]
%            + (mu/2/di)*||xi||^2 
%
% where (ai_1,...,ai_{di}),(bi_1,...,bi_{di}) are the data for client i
%       ai_j\in\R^{n-by-1}, j=1,...,di
%       bi_j\in\R^{1},      j=1,...,di
%       mu>0, (default value: 0.001)
%
% ---------- Non-convex Regularized Logistic regression (NCLogReg) --------
% For every i = 1,2,...,m
%
%      fi(xi) = (1/di)*sum_{j=1}^{di}[ ln(1+exp(<ai_j,xi>)-bi_j*<ai_j,xi> ]
%             + (mu/2/di)*sum_{t=1}^n (xi_t^2/(1+xi_t^2))  
%
% where (ai_1,...,ai_{di}),(bi_1,...,bi_{di}) are the data for client i
%       ai_j\in\R^{n-by-1}, j=1,...,di
%       bi_j\in\R^{1},      j=1,...,di
%       mu>0, (default value: 0.1)
% =========================================================================
% Inputs:
%   dim     : A 1-by-m row vector, dim = (d1, d2, ..., dm)        (REQUIRED)
%             di is the number of rows of Ai, i=1,2,...,m
%             Let d = d1 + d2 + ... + dm
%   n       : Dimension of solution x                             (REQUIRED)
%   A       : A=[A1; A2; ...; Am]\in\R^{d-by-n}                   (REQUIRED)
%   b       : b=[b1; b2; ...; bm]\in\R^{d-by-1}                   (REQUIRED)
%   k0      : A positive integer controlling communication rounds (REQUIRED)
%             The larger k0 is the fewer communication rounds are
%   prob    : must be one of {'LinReg','LogReg','NCLogReg'}       (REQUIRED)
%   pars  :   All parameters are OPTIONAL                                                     
%             pars.r0    --  A positive scalar (default: `1) 
%                            NOTE: Increase this value if you find the solver diverges   
%             pars.optH  -- 'gram', a gram matrix will be used for H
%                           'diag', a diagonal matrix will be used for H, (default)
%             pars.tol   --  Tolerance of the halting condition (default,1e-7)
%             pars.maxit --  Maximum number of iterations (default,1000*k0) 
% =========================================================================
% Outputs:
%     out.sol:      The solution x
%     out.obj:      Objective function value at out.sol
%     out.time:     CPU time
%     out.iter:     Number of iterations 
%     out.cr:       Total number of communication rounds
% =========================================================================
% Written by Shenglong Zhou on 08Nov2022 based on the algorithm proposed in
%     Shenglong Zhou,  Geoffrey Ye Li,
%     An Efficient Hybrid Algorithm for Federated Learning,
%     arXiv:2205.01438, 2022. 		
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% =========================================================================
warning off; rng('shuffle'); 

t0  = tic;
if  nargin < 6
    fprintf(' No enough inputs !!!\n No problems will be solved!!!\n'); 
    return;
elseif nargin<7 
    pars   = [];
end

d = sum(dim);
if size(A,1)~=d  
    fprintf(' Dimensions are not consistent !!! \n No problems will be solved!!!\n'); 
    return;
end

[m,optH,rate,maxit,tol,fun,sigma,wrsig,Ax] = set_parameters(A,b,dim,n,prob,pars); 
Fnorm   = @(x)norm(x,'fro')^2;
objx    = zeros(1,maxit);
errx    = zeros(1,maxit); 
X       = zeros(n,m); 
PI      = X;
Z       = X;         
m0      = ceil(rate*m);

fprintf(' Start to run the solver -- FedGiA \n');
fprintf(' -------------------------------------------------------------\n');
fprintf('                          Iter    f(y)      Error      Time  \n');  
fprintf(' -------------------------------------------------------------\n');

% main body --------------------------------------------------------------- 
for iter = 0 : maxit
       
    if  mod(iter, k0)==0              
        x       = mean(Z,2);  
        [fx,gx] = fun(x);  
        err     = Fnorm(mean(gx,2)/m);      
        M       = randperm(m);
        M       = sort(M(1:m0)); 
    end
    
    if iter == 0
       tol = min(tol,0.1*err);
    end 
     
    objx(iter+1) = fx; 
    errx(iter+1) = err;      
    if mod(iter, k0)==0    
    fprintf(' Communication at iter = %4d %9.4f   %9.3e  %6.3fsec\n',...
              iter, fx, err, toc(t0)); 
    end     
    if err <= tol && mod(iter,1)==0; break;  end
        
    for i   = 1 : m 
        if  ismember(i,M)           
            if isequal(optH, 'gram') 
                rhs = gx(:,i)+PI(:,i); 
                if min(dim(i),n) > 500
                    X(:,i)  = x - my_cg(Ax{i},rhs,1e-8,50,zeros(n,1));
                else
                    X(:,i)  = x - Ax{i}(rhs);
                end
            else
                X(:,i)  = x-(gx(:,i)+PI(:,i))/wrsig(i);
            end
            PI(:,i) = PI(:,i) + (X(:,i)-x)*sigma;
            Z(:,i)  = X(:,i)+ PI(:,i)/sigma;  
        else
            X(:,i)  = x;    
            PI(:,i) = -gx(:,i);
            Z(:,i)  = x + PI(:,i)/sigma;    
        end
    end 
end

% Resuts output -----------------------------------------------------------
out.sol    = x;
out.obj    = fx;
out.objx   = objx(1:iter+1);
out.errx   = errx(1:iter+1);  
out.iter   = iter;
out.time   = toc(t0);  
out.cr     = ceil(iter/k0);
fprintf(' -------------------------------------------------------------\n');
fprintf(' Objective:     %10.3f\n',fx); 
fprintf(' Iteration:     %10d\n',iter);
fprintf(' Error:         %10.2e\n',err);
fprintf(' Time:          %7.3fsec\n',out.time);
fprintf(' CR:            %10d\n',out.cr);
fprintf(' -------------------------------------------------------------\n');

end

% Set parameters ---------------------------------------------------------- 
function [m,optH,rate,maxit,tol,fun,sigma,wrsig,Ax] = set_parameters(A,b,dim,n,prob,pars) 
    m       = length(dim);
    maxit   = 1e4;   
    optH    = 'diag';
    rate    = 0.5;
    r0      = 1;
    if isfield(pars,'optH');  optH  = pars.optH;  end
    if isfield(pars,'maxit'); maxit = pars.maxit; end
    if isfield(pars,'rate');  rate  = pars.rate;  end 
    if isfield(pars,'r0');    r0    = pars.r0;    end 
  
    I      = zeros(m+1,1);
    I(1)   = 0;
    for i  = 1 : m  
        I(i+1) = I(i)+dim(i);
    end
    ri      = zeros(1,m);
    Ai      = cell(1,m);
    bi      = cell(1,m);
    AAi     = cell(1,m);
    Ati     = cell(1,m);
    for i   = 1 : m 
        indi   = I(i)+1:I(i+1); 
        Ai{i}  = A(indi,:);  
        bi{i}  = b(indi);
        Ati{i} = Ai{i}';
        if dim(i) >= n
           AAi{i} = Ati{i}*Ai{i};
           ri(i)  = eigs(AAi{i},1)/dim(i); 
        else
           AAi{i} = Ai{i}*Ati{i};
           ri(i)  = eigs(AAi{i}',1)/dim(i);
        end     
    end    
    Ax    = cell(1,m);
    switch prob
        case 'LinReg'    
            tol    = 1e-7;
            sigma  = r0*0.15*max(ri);   
            wrsig  = ri+sigma;  
            fun    = @(X)funcLinear(X,Ai,bi,m,n,dim); 
            if isequal(optH, 'gram') 
                w1 = 1./dim;
                w2 = sigma*ones(size(dim));
            end
        case 'LogReg'
            tol    = 5e-6/sum(dim);
            ri     = ri/4+1e-3./dim; 
            sigma  = r0*max(0.025,4*log(sum(dim))/n)*max(ri);         
            wrsig  = ri+sigma;         
            fun    = @(x)funcLogist(x,Ai,bi,m,n,dim,1e-3);  
            if isequal(optH, 'gram')
                w1 = 0.25./dim;
                w2 = (1e-3./dim+sigma);
            end       
        case 'NCLogReg' 
            tol    = 5e-6/sum(dim); 
            mu     = 0.01;
            ri     = ri/4 + mu./dim;   
            sigma  = r0*max(0.025,4*log(sum(dim))/n)*max(ri);   
            wrsig  = ri + sigma;    
            fun    = @(x)funcNCLogist(x,Ai,bi,m,n,dim,0.01);
            if isequal(optH, 'gram')
                w1 = 0.25./dim;
                w2 = (mu./dim+sigma);
            end
        otherwise
            fprintf( ' ''prob'' is incorrect !!!\n ''porb'' must be one of {''LinReg'',''LogReg'',''NCLogReg''}\n')
    end
    
     if isequal(optH, 'gram')
        for i = 1 : m      
            if  n    <= min(dim(i), 500)
                inA   = inv(AAi{i}+(w2(i)/w1(i))*eye(n));
                Ax{i} = @(x)(inA*x)/w1(i);   
            elseif dim(i) <= min(n,500)
                inA   = inv( AAi{i}+ (w2(i)/w1(i))*eye(dim(i)) );
                Ax{i} = @(x) (x-Ati{i}*(inA*(Ai{i}*x)))/w2(i);  
            else
                Ax{i} = @(x)( w1(i)*((Ai{i}*x)'*Ai{i})'+ w2(i)*x);  
            end
        end
     end
    clear AAi Ati
end

%--------------------------------------------------------------------------
function  [objx,gradx] = funcLinear(x,Ai,bi,m,n,d)      
    objx     = 0; 
    gradx    = zeros(n,m);
    for i    = 1:m 
        Aij  = Ai{i};
        tmp  = Aij*x-bi{i};
        objx = objx  + norm( tmp )^2/d(i); 
        gradx(:,i) =  (tmp'* Aij )'/d(i);
    end
    objx = objx/2/m;
end

%--------------------------------------------------------------------------
function  [objx,gradx]  = funcLogist(x,Ai,bi,m,n,d,mu)     
    objx   = 0; 
    gradx  = zeros(n,m);
    for i  = 1:m
        Ax   = Ai{i}*x;  
        eAx  = 1 + exp(Ax);
        objx = objx + ( sum( log(eAx)-bi{i}.*Ax ) + (mu/2)*norm(x)^2 )/d(i); 
        gradx(:,i) = ( ((1-bi{i}-1./eAx)'*Ai{i})'+ mu*x )/d(i);
    end
    objx = objx/m; 
end

%--------------------------------------------------------------------------
function  [objx,gradx]  = funcNCLogist(x,Ai,bi,m,n,d,mu)      
    objx   = 0; 
    gradx  = zeros(n,m);
    nmu    = n*mu/2;
    for i  = 1:m
        Ax   = Ai{i}*x;  
        eAx  = 1 + exp(Ax);
        x2   = 1./(1+x.*x);
        objx = objx + ( sum( log(eAx)-bi{i}.*Ax ) + nmu - (mu/2)*sum(x2) )/d(i); 
        gradx(:,i) = ( ((1-bi{i}-1./eAx)'*Ai{i})'+ mu*x.*(x2.^2) )/d(i);
    end
    objx = objx/m; 
end

% Conjugate gradient method-------------------------------------------------
function x = my_cg(fx,b,cgtol,cgit,x)
    r = b;
    e = sum(r.*r);
    t = e;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        if  i == 1  
            p = r;
        else
            p = r + (e/e0)*p;
        end
        w  = fx(p); 
        a  = e/sum(p.*w);
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = sum(r.*r);
    end
   
end

% inverse lemma -----------------------------------------------------------
function invA = inverseAAtI(A,t)

    [row,col] = size(A);
    
    if     row  <= col && row <= 1000
           invA = inv(A*A'+t*eye(row));
    elseif col   <= row && col <= 1000
           invA = inv(A'*A+t*eye(col));
    else
        
    end
end
