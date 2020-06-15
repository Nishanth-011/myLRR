%% 1
clear;
addpath('Datasets/');
addpath('Functions/');
addpath('Functions/Measures/');
addpath('Functions/Measures/ami');
addpath('Functions/Measures/munkres');
addpath('LRR/');
addpath('FastESC-master');
addpath('FastESC-master/EBMM_Release');
addpath('SparsifiedKMeans-master');
addpath('SparsifiedKMeans-master/private');
addpath('NN_LRR_AGR_final_code/');
addpath('IBDLR/');



% IBDLR and RKLRR for LRR problem

clear;
close all;
addpath(genpath(cd));
load ORL_32x32
fprintf('\n\n**************************************   %s   *************************************\n' , datestr(now) );
fea=NormalizeFea(fea);

% exp 1
k = 3;    % number of subspaces
n = 100;     % samples in each subspace
d = 150;    % dimension
r = 25;      % rank

% [X_cln, X,gnd] = generate_data(n,r,d,k);
X = fea';
k = 40;
[d, n]=size(X);
lambda = 3;  
gamma = 5;
%% IBDLR
tic
[Z, ~,objs] = IBDLR(X'*X,k,lambda,gamma,1);
%[Z] = RKLRR(0.21,X'*X);
time = toc;
%Z(1:N+1:end) = 0;
Z_n = cnormalize(Z, Inf);
A = abs(Z_n) + abs(Z_n)';
% groups = SpectralClustering1(A, 3, 'Eig_Solver', 'eigs');
% imshow(A)
% evalAccuracy(gnd',groups)
% 
% E = X-X*Z;
% obj = lambda*nuclearnorm(Z)+sum(sqrt(sum(E.*E)));
% iter = length(objs);
% figure;
% plot(real(objs))
% 
% fprintf('Minimum \t Time \t Iter.\n' );
% fprintf('%f \t %f \t %d\n', obj, time, 0);
kk = 40;
for i = 1:10
    label_gnd{i} = gnd;
    %% kmeans++
    c = NJW(A,kk);
    [aARI(i),aAMI(i),aNMI(i),aACC(i)] = evaluate(c,kk,gnd,kk);
    aAUC(i) = AUC(gnd,c);
    label_kmeans{i} = c;
    [aTPR(i),aFPR(i),aPrecision(i),aRecall(i),aF1(i)] = performanceIndexs(gnd,c);
end
amARI = mean(aARI);
amAMI = mean(aAMI);
amNMI = mean(aNMI);
amACC = mean(aACC);
amAUC = mean(aAUC);
amTPR = mean(aTPR);
amFPR = mean(aTPR);
amPrecision = mean(aPrecision);
amRecall = mean(aRecall);
amF1 = mean(aF1);
aaresult = [amARI,amAMI,amNMI,amACC,amAUC,amTPR,amFPR,amPrecision,amRecall,amF1,t];






data_num = 1;
run_time = 10;
dataset_name = {'ORL_32x32','Umist','YaleB_323','COIL20','mnist_all','CCC40','USPSfu','PD100'};
load(strcat(dataset_name {data_num},'.mat'))
if data_num == 1
    interval = 40;%è¿?ä¸????»æ?·æ???°ç???´é??
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 2
    fea=X';
    interval = 20;%è¿?ä¸????»æ?·æ???°ç???´é??
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 3
    interval = 20;%è¿?ä¸????»æ?·æ???°ç???´é??
    ttime = 1;
elseif data_num == 4
    interval = 20;%è¿?ä¸????»æ?·æ???°ç???´é??
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 5
    interval = 10;%è¿?ä¸????»æ?·æ???°ç???´é??
    num = 200;
    fea = double([train0(1:num,:);train1(1:num,:);train2(1:num,:);train3(1:num,:);train4(1:num,:);train5(1:num,:);train6(1:num,:);train7(1:num,:);train8(1:num,:);train9(1:num,:)]);
    gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 6
    interval = 50;%è¿?ä¸????»æ?·æ???°ç???´é??
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 7
    dnum = 20;
    num = dnum;
    fea = [data0(:,1:dnum)';data1(:,1:dnum)';data2(:,1:dnum)';data3(:,1:dnum)';data4(:,1:dnum)';data5(:,1:dnum)';data6(:,1:dnum)';data7(:,1:dnum)';data8(:,1:dnum)';data9(:,1:dnum)'];
    fea = double(fea);
    gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
    interval = 2;%è¿?ä¸????»æ?·æ???°ç???´é??
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 8
    interval = 10;
    ttime = floor(length(unique(gnd))/interval);
end
data = fea;
datal = double(gnd);
label_kmeans = cell(ttime,run_time);
label_gnd = cell(ttime,run_time);
c = [];

kk = interval;%ç±»å????
number = find(datal<=kk);%æ¯?ä¸?ç±»ç??ä¸???
%number=max(number);
fea=data(number,:);%kk *number,:);  
gnd=datal(number);%kk*number);
fea=NormalizeFea(fea);
lambda = 0.1;
tic
% X = fea';
% lambda1 = 1e-4;
% lambda2 = 1e-3;
% lambda3 = 1e-1;
% options = [];
% options.NeighborMode = 'KNN';
% options.k = 10;
% options.WeightMode = 'Binary';      % Binary  HeatKernel
% Z = constructW(X',options);
% Z = full(Z);
% Z1 = Z-diag(diag(Z));         
% Z = (Z1+Z1')/2;
% DZ= diag(sum(Z));
% LZ = DZ - Z;                
% [F_ini, ~, evs]=eig1(LZ, c, 0);
% Z_ini = Z;
% clear LZ DZ Z fea Z1
% max_iter= 80;
% Ctg = inv(X'*X+2*eye(size(X,2)));
% [Z,S,U,F,E] = LRR_AGR(X,F_ini,Z_ini,c,lambda1,lambda2,lambda3,max_iter,Ctg);
X = fea';
k = 40;
[d, n]=size(X);
lambda = 3;  
gamma = 5;
%% IBDLR
tic
[Z, ~,objs] = IBDLR(X'*X,k,lambda,gamma,1);
%[Z] = RKLRR(0.21,X'*X);
time = toc;
%Z(1:N+1:end) = 0;
Z_n = cnormalize(Z, Inf);
A = abs(Z_n) + abs(Z_n)';
toc
t = toc;
for i = 1:run_time
    label_gnd{i} = gnd;
    %% kmeans++
    c = NJW(Z,kk);
    [aARI(i),aAMI(i),aNMI(i),aACC(i)] = evaluate(c,kk,gnd,kk);
    aAUC(i) = AUC(gnd,c);
    label_kmeans{i} = c;
    [aTPR(i),aFPR(i),aPrecision(i),aRecall(i),aF1(i)] = performanceIndexs(gnd,c);
end
amARI = mean(aARI);
amAMI = mean(aAMI);
amNMI = mean(aNMI);
amACC = mean(aACC);
amAUC = mean(aAUC);
amTPR = mean(aTPR);
amFPR = mean(aTPR);
amPrecision = mean(aPrecision);
amRecall = mean(aRecall);
amF1 = mean(aF1);
aaresult = [amARI,amAMI,amNMI,amACC,amAUC,amTPR,amFPR,amPrecision,amRecall,amF1,t];
save(strcat('Results/newrevise/',strcat('s',dataset_name{data_num},'lrragr')))