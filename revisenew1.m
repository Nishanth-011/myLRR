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
data_num = 1;
run_time = 10;
dataset_name = {'ORL_32x32','Umist','YaleB_3232','COIL20','mnist_all','CCC80','USPSfu','PD100'};
load(strcat(dataset_name {data_num},'.mat'))
if data_num == 1
    interval = 2;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 2
    fea=X';
    interval = 2;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 3
    interval = 2;%这个是去样本数的间隔
    ttime = 10;
elseif data_num == 4
    interval = 2;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 5
    interval = 2;%这个是去样本数的间隔
    num = 400;
    fea = double([train0(1:num,:);train1(1:num,:);train2(1:num,:);train3(1:num,:);train4(1:num,:);train5(1:num,:);train6(1:num,:);train7(1:num,:);train8(1:num,:);train9(1:num,:)]);
    gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 6
    interval = 5;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 7
    dnum = 20;
    num = dnum;
    fea = [data0(:,1:dnum)';data1(:,1:dnum)';data2(:,1:dnum)';data3(:,1:dnum)';data4(:,1:dnum)';data5(:,1:dnum)';data6(:,1:dnum)';data7(:,1:dnum)';data8(:,1:dnum)';data9(:,1:dnum)'];
    fea = double(fea);
    gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
    interval = 2;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 8
    interval = 2;
    ttime = floor(length(unique(gnd))/interval);
end
data = fea;
datal = double(gnd);
label_0kmeans = cell(ttime,run_time);
label_1njw = cell(ttime,run_time);
label_2fast = cell(ttime,run_time);
label_3lrradp = cell(ttime,run_time);
label_4nsllrr = cell(ttime,run_time);
label_5ours = cell(ttime,run_time);
label_6gnd = cell(ttime,run_time);
ACC0 = zeros(ttime,run_time);
ACC1 = zeros(ttime,run_time);
ACC2 = zeros(ttime,run_time);
ACC3 = zeros(ttime,run_time);
ACC4 = zeros(ttime,run_time);
ACC5 = zeros(ttime,run_time);
AUC0 = zeros(ttime,run_time);
AUC1 = zeros(ttime,run_time);
AUC2 = zeros(ttime,run_time);
AUC3 = zeros(ttime,run_time);
AUC4 = zeros(ttime,run_time);
AUC5 = zeros(ttime,run_time);
ARI0 = zeros(ttime,run_time);
ARI1 = zeros(ttime,run_time);
ARI2 = zeros(ttime,run_time);
ARI3 = zeros(ttime,run_time);
ARI4 = zeros(ttime,run_time);
ARI5 = zeros(ttime,run_time);
AMI0 = zeros(ttime,run_time);
AMI1 = zeros(ttime,run_time);
AMI2 = zeros(ttime,run_time);
AMI3 = zeros(ttime,run_time);
AMI4 = zeros(ttime,run_time);
AMI5 = zeros(ttime,run_time);
NMI0 = zeros(ttime,run_time);
NMI1 = zeros(ttime,run_time);
NMI2 = zeros(ttime,run_time);
NMI3 = zeros(ttime,run_time);
NMI4 = zeros(ttime,run_time);
NMI5 = zeros(ttime,run_time);
for time =1:ttime
    kk = interval*time;%类别数
    number = find(datal<=kk);%每一类的个数
    %number=max(number);
    fea=data(number,:);%kk *number,:);  
    gnd=datal(number);%kk*number);
    fea=NormalizeFea(fea); 
    [LRRADP b1 dis1] = LRSA(fea');
    [LRRHWAP b4 dis4] = LRRHWAPin(fea',0.1,0.1,5,99);%yale数据集用的的是k=3
    k=5;%knn的k
    Y = fea';
    Yg = fea';
    a=fkNN(Yg,k);
    b=constractmap(a);
    c = transmit(b,0);
    d = (c+c')/2;
    d(find(d>0))=1;
   [NSLLRR,OBJ] =  sparse_graph_LRR(Yg,d);
   
   % set sigma
    dis = pdist2_my(fea,fea);
    dis(dis<0) = 0;
    sigma = mean(sqrt(dis(:)));

    % demension of EFM, see [1] for details
    D = 30;

    %% 1. FastESC
    vESC = FastESC(fea, sigma, D, kk);
    c0 = [];
    c1 = [];
    c2 = [];
    c3 = [];
    c4 = [];
    c5 = [];
    
    for i = 1:run_time
        label_6gnd{time,i} = gnd;
        %% kmeans++
        c0 = kmeans(fea,kk);
        [ARI0(time,i),AMI0(time,i),NMI0(time,i),ACC0(time,i)] = evaluate(c0,kk,gnd,kk);
        AUC0(time,i) = AUC(gnd,c0);
        label_0kmeans{time,i} = c0;
        [TPR,FPR,Precision,Recall,F1] = performanceIndexs(gnd,c0);
        %% NJW
        c1 =  NJW(fea,kk); 
        [ARI1(time,i),AMI1(time,i),NMI1(time,i),ACC1(time,i)] = evaluate(c1,kk,gnd,kk);
        AUC1(time,i) = AUC(gnd,c1);
        label_1njw{time,i} = c1;
        %% FastESC
        [~,c2] = sort(vESC, 2, 'descend');
        [ARI2(time,i),AMI2(time,i),NMI2(time,i),ACC2(time,i)] = evaluate(c2(:,1),kk,gnd,kk);
        AUC2(time,i) = AUC(gnd,c2);
        label_2fast{time,i} = c2(:,1);
        %% LRRADP
        c3 =  NJW(LRRADP,kk);
        [ARI3(time,i),AMI3(time,i),NMI3(time,i),ACC3(time,i)] = evaluate(c3,kk,gnd,kk);
        AUC3(time,i) = AUC(gnd,c3);
        label_3lrradp{time,i} = c3;
        %% NSLLRR
        c4 =  NJW(NSLLRR,kk);
        [ARI4(time,i),AMI4(time,i),NMI4(time,i),ACC4(time,i)] = evaluate(c4,kk,gnd,kk);
        AUC4(time,i) = AUC(gnd,c4);
        label_4nsllrr{time,i} = c4;
        %% OURS
        c5 =  NJW(LRRHWAP,kk);
        [ARI5(time,i),AMI5(time,i),NMI5(time,i),ACC5(time,i)] = evaluate(c5,kk,gnd,kk);
        AUC5(time,i) = AUC(gnd,c5);
        label_5ours{time,i} = c5;
    end
%     %% kmeans
%     m_ARI0(time) = mean(ARI0(time,1:run_time));
%     m_AMI0(time) = mean(AMI0(time,:));
%     m_NMI0(time) = mean(NMI0(time,:));
%     m_ACC0(time) = mean(ACC0(time,:));
%     m_AUC0(time) = mean(AUC0(time,:));
%     %% njw
%     m_ARI1(time) = mean(ARI1(time,:));
%     m_AMI1(time) = mean(AMI1(time,:));
%     m_NMI1(time) = mean(NMI1(time,:));
%     m_ACC1(time) = mean(ACC1(time,:));
%     m_AUC1(time) = mean(AUC1(time,:));
%     %% fast
%     m_ARI2(time) = mean(ARI2(time,:));
%     m_AMI2(time) = mean(AMI2(time,:));
%     m_NMI2(time) = mean(NMI2(time,:));
%     m_ACC2(time) = mean(ACC2(time,:));
%     m_AUC2(time) = mean(AUC2(time,:));
%     %% lrradp
%     m_ARI3(time) = mean(ARI3(time,:));
%     m_AMI3(time) = mean(AMI3(time,:));
%     m_NMI3(time) = mean(NMI3(time,:));
%     m_ACC3(time) = mean(ACC3(time,:));
%     m_AUC3(time) = mean(AUC3(time,:));
%     %% nsllrr
%     m_ARI4(time) = mean(ARI4(time,:));
%     m_AMI4(time) = mean(AMI4(time,:));
%     m_NMI4(time) = mean(NMI4(time,:));
%     m_ACC4(time) = mean(ACC4(time,:));
%     m_AUC4(time) = mean(AUC4(time,:));
%     %% ours
%     m_ARI5(time) = mean(ARI5(time,:));
%     m_AMI5(time) = mean(AMI5(time,:));
%     m_NMI5(time) = mean(NMI5(time,:));
%     m_ACC5(time) = mean(ACC5(time,:));
%     m_AUC5(time) = mean(AUC5(time,:));
end
    %% kmeans
    m_ARI0 = mean(ARI0');
    m_AMI0= mean(AMI0');
    m_NMI0= mean(NMI0');
    m_ACC0= mean(ACC0');
    m_AUC0= mean(AUC0');
    %% njw
    m_ARI1= mean(ARI1');
    m_AMI1= mean(AMI1');
    m_NMI1= mean(NMI1');
    m_ACC1= mean(ACC1');
    m_AUC1= mean(AUC1');
    %% fast
    m_ARI2= mean(ARI2');
    m_AMI2= mean(AMI2');
    m_NMI2= mean(NMI2');
    m_ACC2= mean(ACC2');
    m_AUC2= mean(AUC2');
    %% lrradp
    m_ARI3= mean(ARI3');
    m_AMI3= mean(AMI3');
    m_NMI3= mean(NMI3');
    m_ACC3= mean(ACC3');
    m_AUC3= mean(AUC3');
    %% nsllrr
    m_ARI4= mean(ARI4');
    m_AMI4= mean(AMI4');
    m_NMI4= mean(NMI4');
    m_ACC4= mean(ACC4');
    m_AUC4= mean(AUC4');
    %% ours
    m_ARI5= mean(ARI5');
    m_AMI5= mean(AMI5');
    m_NMI5= mean(NMI5');
    m_ACC5= mean(ACC5');
    m_AUC5= mean(AUC5');

 save(strcat('Results/newrevise/',strcat(dataset_name{data_num},'g_2p')))