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
addpath('PR_code_AWNLRR/');
addpath('PR_code_AWNLRR/Ncut_9');
for all = 8:8
data_num = all;
run_time = 10;
dataset_name = {'ORL_32x32','Umist','YaleB_3232','COIL20','mnist_all','CCC40','PD100','Yale_32x32'};
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
    interval = 10;
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 8
    interval = 15;
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
% tic
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
fea = fea';
% fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
n = length(gnd);
nnClass = length(unique(gnd));  

options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Binary';
Z = constructW(fea',options);
Z_ini = full(Z);
clear LZ Z Z1 options

% % if you only have cpu do this 
% Ctg = inv(fea'*fea+eye(size(fea,2)));

% % -------- if you have gpu you can accelerate the inverse operation as follows:  ---------- % %
Xg = gpuArray(single(fea));
Ctg = inv(Xg'*Xg+eye(n));
Ctg = double(gather(Ctg));
clear Xg;
aACC = zeros(9,9);
for i = 1:9
    for j = 1:9
        lambda1 = 0.01;
        lambda2 = 0.0001*10^(i-1);
        lambda3 = 0.0001*10^(j-1);
        % lambda1 = 0.01
        % lambda2 = 1e-4
        % lambda3 = 1e-4
        miu = 1e-2;
        rho = 1.1;
        max_iter = 80;
%         [Z,S,obj] = AWLRR(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
        [Z,E,B, err, obj] =  Jlrs_slover(fea');
%         toc
%         t = toc;
        Z_out = Z;
        A = Z_out;
        A = A - diag(diag(A));
        A = abs(A);
        A = (A+A')/2;  
        % [Z b4 dis4] = LRRHWAPin(fea',0.1,0.1,5,99);
        % A = Z;

        % result = ClusteringMeasure(gnd, result_label); 
%         for i = 1:run_time
            label_gnd{i} = gnd;
        %     [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(A,nnClass);
        %     result_label = zeros(size(fea,2),1);
        %     for j = 1:nnClass
        %         id = find(NcutDiscrete(:,j));
        %         result_label(id) = j;
        %     end
            %% kmeans++
            c = NJW(A,kk);
        %     c = result_label;
            [~,~,~,aACC(i,j)] = evaluate(c,kk,gnd,kk);
%             aAUC(i) = AUC(gnd,c);
%             label_kmeans{i} = c;
%             [aTPR(i,j),aFPR(i,j),aPrecision(i,j),aRecall(i,j),aF1(i,j)] = performanceIndexs(gnd,c);
    end
end
save(strcat('Results/newrevise/',strcat('aaa',dataset_name{data_num},'xuncan')))
end
%         amARI = mean(aARI);
%         amAMI = mean(aAMI);
%         amNMI = mean(aNMI);
%         amACC = mean(aACC);
%         amAUC = mean(aAUC);
%         amTPR = mean(aTPR);
%         amFPR = mean(aTPR);
%         amPrecision = mean(aPrecision);
%         amRecall = mean(aRecall);
%         amF1 = mean(aF1);
%         aaresult = [amARI,amAMI,amNMI,amACC,amAUC,amTPR,amFPR,amPrecision,amRecall,amF1,t];
% save(strcat('Results/newrevise/',strcat('s',dataset_name{data_num},'lrragr')))