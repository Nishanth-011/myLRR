%%%%这个是用来复现聚类结果的代码
%%%%
%0----kmeans 
%1----NJW
%2----NSLLR
%3----LRRADP
%4----LRRHWAP
%需要哪个数据集将某一个数据集前面的%去掉即可
%这个代码只跑整个数据集
clear;
addpath('Datasets/');
addpath('Functions/');
addpath('Functions/Measures/');
addpath('Functions/Measures/ami');
addpath('Functions/Measures/munkres');
addpath('LRR');
addpath('FastESC-master');
addpath('FastESC-master/EBMM_Release');
addpath('SparsifiedKMeans-master');
addpath('SparsifiedKMeans-master/private');
data_num =7;
run_time = 10;
dataset_name = {'ORL_32x32','Umist','YaleB_3232','COIL20','mnist_all','CCC40','Yale_32x32','PD200'};
load(strcat(dataset_name {data_num},'.mat'))
if data_num == 1
    interval = 40;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 2
    fea=X';
    interval = 20;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 3
    interval = 20;%这个是去样本数的间隔
    ttime = 1;
elseif data_num == 4
    interval = 20;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 5
    interval = 10;%这个是去样本数的间隔
    num = 200;
    fea = double([train0(1:num,:);train1(1:num,:);train2(1:num,:);train3(1:num,:);train4(1:num,:);train5(1:num,:);train6(1:num,:);train7(1:num,:);train8(1:num,:);train9(1:num,:)]);
    gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 6
    interval = 50;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 7
    interval = 15;%这个是去样本数的间隔
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

kk = interval;%类别数
number = find(datal<=kk);%每一类的个数
%number=max(number);
fea=data(number,:);%kk *number,:);  
gnd=datal(number);%kk*number);
fea=NormalizeFea(fea); 
for i = 1:run_time
    label_gnd{i} = gnd;
    %% kmeans++
    tic
    c = kmeans(fea,kk);
    toc
    t = toc;
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
save(strcat('Results/newrevise/',strcat('s',dataset_name{data_num},'kmeans')))