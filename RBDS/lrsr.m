function [Z,E] = lrsr(X,D,param,display)
% This routine solves the following optimization problem,
% min_{Z,E} |Z|_*+beta*|Z|_1+lambda*|E|_1
% s.t., X = DZ+E
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
%        D -- D*M matrix of a dictionary, M is the size of the dictionary
lambda=param.lambda;
beta=param.beta;
tol = 1e-6;
maxIter = 5000;
[d,n] = size(X);
m = size(D,2);
rho = 1.1;
max_mu = 1e8;
mu = 1e-5;
if nargin<4
    display = true;
end

%% Initializing optimization variables
% intialize
Z = zeros(m,n);
J = zeros(m,n);
E = zeros(d,n);
L = J;

Y1 = zeros(d,n);
Y2 = zeros(m,n);
Y3= Y2;
%% Start main loop
iter = 0;
if display
    disp(['initial,rank=' num2str(rank(Z))]);
end
while iter<maxIter
    iter = iter + 1;
    
    % update J
    temp = Z + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
    % udpate Z
    Z_left = D'*D+2*eye(size(D'*D));
    Z = Z_left \ ( (D)'*(X-E)+J +L +( (D)'*Y1-Y2 -Y3 )/mu );
    
    % update L
    L_temp = Z+Y3/mu;
    L=max(0,L_temp-beta/mu)+min(0,L_temp+beta/mu);
    
    % update E
    xmaz = X-D*Z;
    temp = xmaz+Y1/mu;
    E=max(0,temp-lambda/mu)+min(0,temp+lambda/mu);
    
    leq1 = xmaz-E;
    leq2 = Z-J;
    leq3=Z-L;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC = max( stopC, max(max(abs(leq3))) );
    if display && (iter==1 || mod(iter, 20 )==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol
        disp('ALM done.');
        break;
    else
        %update lagrange multipliers
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        mu = min(max_mu,mu*rho);
    end
end