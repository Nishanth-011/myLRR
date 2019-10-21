%%%%这个是用来复现聚类结果的代码
%%%%
%0----kmeans 
%1----NJW
%2----NSLLR
%3----LRRADP
%4----LRRHWAP
%需要哪个数据集将某一个数据集前面的%去掉即可
%比如orl进行8次每次间隔是5，则ttime = 8，interval = 5
%最后的结果是accuracy0_m—accuracy4_m和NMI0-NMI4
clear;
addpath('Datasets/');
addpath('Functions/');
addpath('LRR');
addpath('FastESC-master');
addpath('FastESC-master/EBMM_Release');
data_num = 5;
dataset_name = {'ORL_32x32','Umist','YaleB_3232','COIL20','mnist_all','CCC50'};
load(strcat(dataset_name {data_num},'.mat'))
if data_num == 1
    interval = 5;%这个是去样本数的间隔
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
    interval = 2;%这个是去样本数的间隔
    ttime = floor(length(unique(gnd))/interval);
end
data = fea;
datal = double(gnd);
for time =1:ttime
    kk = interval*time;%类别数
    number = find(datal<=kk);%每一类的个数
    %number=max(number);
    fea=data(number,:);%kk *number,:);  
    gnd=datal(number);%kk*number);
    fea=NormalizeFea(fea); 
    [new1 b1 dis1] = LRSA(fea');
    [new2 b4 dis4] = LRRHWAP(fea',0.1,0.1,5,99);%yale数据集用的的是k=3
    k=5;%knn的k
    Y = fea';
    Yg = fea';
    a=fkNN(Yg,k);
    b=constractmap(a);
    c = transmit(b,0);
    d = (c+c')/2;
    d(find(d>0))=1;
   [new,OBJ] =  sparse_graph_LRR(Yg,d);
   accuracy0 = zeros(1,10);
   accuracy1 = zeros(1,10);
   accuracy2 = zeros(1,10);
   accuracy3 = zeros(1,10);
   accuracy4 = zeros(1,10);
   % set sigma
    dis = pdist2_my(fea,fea);
    dis(dis<0) = 0;
    sigma = mean(sqrt(dis(:)));

    % demension of EFM, see [1] for details
    D = 30;

    %% 1. FastESC
    vESC = FastESC(fea, sigma, D, kk);
    for i = 1:10
        c0 =  kmeans(fea,kk,'Start','plus');
        idx=bestMap(gnd,c0); % 匹配
        accuracy0(i) = length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
        c1 =  NJW(fea,kk); 
        idx=bestMap(gnd,c1); % 匹配
        accuracy1(i) = length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
        c2 =  NJW(new,kk);
        idx=bestMap(gnd,c2); % 匹配
        accuracy2(i) = length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
        c3 =  NJW(new1,kk);
        idx = bestMap(gnd,c3); % 匹配
        accuracy3(i) = length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
        c4 =  NJW(new2,kk);
        idx=bestMap(gnd,c4); % 匹配
        accuracy4(i)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
    end 
    accuracy0_m(time)=mean(accuracy0);
    NMI0(time) = NormalizedMutualInformation(gnd,c0,length(gnd),kk); 
    accuracy1_m(time)=mean(accuracy1);
    NMI1(time) = NormalizedMutualInformation(gnd,c1,length(gnd),kk); 
    accuracy2_m(time)=mean(accuracy2);
    NMI2(time) = NormalizedMutualInformation(gnd,c2,length(gnd),kk); 
    accuracy3_m(time)=mean(accuracy3);
    NMI3(time) = NormalizedMutualInformation(gnd,c3,length(gnd),kk); 
    accuracy4_m(time)=mean(accuracy4);
    NMI4(time) =   NormalizedMutualInformation(gnd,c4,length(gnd),kk);
    [~,c5] = sort(vESC, 2, 'descend');
    idx=bestMap(gnd,c5(:,1)); % 匹配
    accuracy5_m(time)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
    NMI5(time) = NormalizedMutualInformation(gnd,c5(:,1),length(gnd),kk); 
end
 save(strcat('result/','revise',strcat(dataset_name{data_num},'g2p')))
