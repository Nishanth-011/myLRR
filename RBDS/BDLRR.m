function [Z,D,E] = BDLRR(X, D, param)
% numPerCls = para.numPerCls;

alpha = param.alpha;
lambda = param.lambda;
beta = param.beta;
gamma = param.gamma;

trainSamPerCls = ones(1, param.clsNum)*(param.trainNumPerCls);

%------------------------------------------------
% Parameters initialization
%------------------------------------------------
[d,n] = size(X);
[~,m] = size(D);
tol = 1e-6;
maxIter = 1e4;
% rho = 1.15;
rho = 1.1;
mu= 1e-5;
% mu = sqrt(max(d,m))\1;
max_mu = 1e8;
Z = zeros(m,n);
E = zeros(size(X));
J = zeros(m,n);
L = J;
R = zeros(m,n);
Y1 = zeros(d,n);
Y2 = zeros(m,n);
Y3 = Y2;
%% Start main loop
iter = 0;
while iter<maxIter
    iter = iter + 1;
    temp = Z + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    
    %------------------------------------------------
    % update J
    %------------------------------------------------
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    clear U V sigma svp temp;
    
    %------------------------------------------------
    % update Z
    %------------------------------------------------
    Z_left = (D)'*(D)+( alpha/mu+2)*eye(m);
    Z = Z_left \ ( D'*(X-E)+J+L+(D'*Y1 - Y2 -Y3 + alpha*R)/mu);
    clear Z_left;
    
    %------------------------------------------------
    % update L
    %------------------------------------------------
    temp = Z+Y3/mu;
    L = solve_l1_norm(temp,beta/mu);
    clear B temp;
    
    %------------------------------------------------
    % update E
    %------------------------------------------------
    temp = X-D*Z+Y1/mu;
    E = solve_l1_norm(temp,lambda/mu);
    clear temp;
    
    % update D
    D_trans= ( Z*Z'+gamma/mu*eye(m) ) \ ( Y1*Z'/mu - (E-X)*Z' )';
    D = D_trans';
    
    %------------------------------------------------
    % Extract Block-Diagonal Components (R)
    %------------------------------------------------
    Z_block = cell(numel(trainSamPerCls),1);
    for k = 1:numel(trainSamPerCls)
        Z_block{k} = Z ( (k-1)*param.dicNumPerCls+1: k*param.dicNumPerCls,...
            sum(trainSamPerCls(1:k-1))+1:sum( trainSamPerCls(1:k) ) );
    end
    
    R = blkdiag(Z_block{:});
    clear Z_block;
    
    %------------------------------------------------
    % Convergence Validation
    %------------------------------------------------
    leq1 = X-D*Z-E;
    leq2 = Z-J;
    leq3 = Z-L;
    stopC1 = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC = max(stopC1,max(max(abs(leq3))));
    
    if stopC<tol || iter>=maxIter
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        mu = min(max_mu,mu*rho);
    end
    if (iter==1 || mod(iter, 20 )==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    obj(iter) = norm(leq1,'fro')/norm(X,'fro');
end
end

% Soft thresholding
function [E] = solve_l1_norm(x,varepsilon)
E = max(x- varepsilon, 0);
E = E+min( x+ varepsilon, 0);
end
