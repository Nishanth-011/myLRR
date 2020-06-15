% min_{A>=0, A*1=1, F'*F=I}  trace(D'*A) + r*||A||^2 + 2*lambda*trace(F'*L*F)
% written by Lunke Fei on 16/07/2015
function [A obj dis] = LRRHWAPfastini(X, distX, Z_ini, g1, g2, k, time)
% y: num*1 clbetaHter indicator vector
% A: num*num learned Hymmetric Himilarity matrix
addpath(genpath('.\YALL1_v1.3'));

NITER = 100;
% NITER = 80;%orl
% NITER = 120;%umist yaleb
% NITER = 200;%coil20

dim = size(X,1);
num = size(X,2);

if nargin < 3
 g2 = 0.1;
end
if nargin < 2
 g1 = 0.1;
end
A0 = distX;
Z = Z_ini;%eye(num);
E = sparse(dim, num);
H = Z;

gama_1 = g1;
gama_2 = g2;

beta = 25;
%beta2 = beta;
Y1 = zeros(dim, num);
Y2 = zeros(num);
normX = norm(X,2);
Ita_a = 2*normX;

%other parameters setting.
ro = 1.1;
svp = 5;
tol = 1e-6;
max_beta = 1e5;

for iter = 1:NITER
    %iter
    
    Z_old = Z;
    E_old = E;
    
    Q1 = Z + (X'*(X-X*Z-E+Y1/beta) - (Z - H + Y2/beta))/Ita_a;
    [U D V] = svd(Q1,'econ');
    D = diag(D);
    svp = length(find(D>1/(Ita_a*beta)));
    if svp >= 1
        D = D(1:svp)-1/(Ita_a*beta);
    else
        D = 0;
        svp = 1;
    end
    Z = U(:,1:svp)*diag(D)*V(:,1:svp)';
    dD = diag(D);
    
    E1 = X-X*Z+Y1/beta;
    theta = gama_1/beta;
    E = max(0,E1-theta)+min(0,E1+theta);
%     H = (gama_2*A0-beta*Z-Y2)/(1-beta);
   H = Calc_Hf(A0, Z, Y2, gama_2, beta);
%    H = Z+Y2/beta-gama_2*A0/beta;
%     H = Z-

    errH = norm(Z-Z_old,2)/normX;
    errE = norm(E-E_old,2)/normX;
    convergence = (errE < tol)&&(errH < tol);
    if convergence
        break;
    end    
    
    obj(iter) = rank(Z) + gama_1*sum(sum(abs(E))) + gama_2*sum(sum(distX.*Z));
   % imshow(H,[]);
    
    Y1 = Y1 + beta*(X-X*Z-E);
    Y2 = Y2 + beta*(Z-H);
    beta = min(max_beta, ro*beta);
end

dis=distX;
A = (H+H')/2;

function [A] = Calc_H(disX,Z,Y,gama, miu)
ss = size(disX, 1);
T = 0.0005;
for i=1:ss
    idxa0 = 1:ss;
    
    dz = Z(i,idxa0);
    dy = Y(i,idxa0);
    dx = disX(i, idxa0);
    ad = dz+dy/miu-gama*dx/miu;
    A(i,idxa0) = EProjSimplex_new(ad);
    %A(i,i) = 0;
    %A = (A>T).*A;
end
function [A] = Calc_Hf(disX,Z,Y,gama, miu)
ss = size(disX, 1);
% T = 0.0005;
idxa0 = 1:ss;
a = fix(ss/10);
b = mod(ss,10);
A = zeros(ss,ss);
dz = Z;
dy = Y;
dx = disX;
ad = dz+dy/miu-gama*dx/miu;

parfor i=1:ss
    A(i,:) = EProjSimplex_new(ad(i,:));
end

function [A] = Calc_Hfast(disX,Z,Y,gama, miu)
ss = size(disX, 1);
idxa0 = 1:ss;
dz = Z;
dy = Y;
dx = disX;
ad = dz+dy/miu-gama*dx/miu;
A = zeros(ss,ss);
A = EProjSimplex_fast(ad);