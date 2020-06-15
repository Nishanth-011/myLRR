%%%%这个是用来复现聚类结果的代码

%需要哪个数据集将某一个数据集前面的%去掉即可
%这个代码只跑整个数据集
clear;
addpath('../Datasets/');
addpath('../Functions/');
addpath('../Functions/Measures/');
addpath('../Functions/Measures/ami');
addpath('../Functions/Measures/munkres');
data_num =8;
run_time = 10;
dataset_name = {'ORL_32x32','Umist','YaleB_3232','COIL20','mnist_all','CCC40','Yale_64x64','PD100'};
load(strcat(dataset_name {data_num},'.mat'))
if data_num == 1
    load orl_y
    kk =40;
elseif data_num == 2
    load umist_y
    kk = 20;%这个是去样本数的间隔
elseif data_num == 3
    kk = 20;%这个是去样本数的间隔
    ttime = 1;
elseif data_num == 4
    kk = 20;%这个是去样本数的间隔
    load coil20_y
    y = label;
elseif data_num == 5
    kk = 10;%这个是去样本数的间隔
    num = 200;
    fea = double([train0(1:num,:);train1(1:num,:);train2(1:num,:);train3(1:num,:);train4(1:num,:);train5(1:num,:);train6(1:num,:);train7(1:num,:);train8(1:num,:);train9(1:num,:)]);
    gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
    load smnist_y
elseif data_num == 6
    kk = 50;%这个是去样本数的间隔
    load ccc_y;
elseif data_num == 7
    kk = 15;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/kk);
elseif data_num == 8
    kk = 10;
    load pd_y
end
c = double(y'+1);
[aARI,aAMI,aNMI,aACC] = evaluate(c,kk,gnd,kk);
[aTPR,aFPR,aPrecision,aRecall,aF1] = performanceIndexs(gnd,c);
%ari NMI	ACC	Precision	F-score
result = [aARI,aNMI,aACC,aPrecision,aF1];