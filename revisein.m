%%%%������������־������Ĵ���
%%%%
%0----kmeans 
%1----NJW
%2----NSLLR
%3----LRRADP
%4----LRRHWAP
%��Ҫ�ĸ����ݼ���ĳһ�����ݼ�ǰ���%ȥ������
%����orl����8��ÿ�μ����5����ttime = 8��interval = 5
%���Ľ����accuracy0_m��accuracy4_m��NMI0-NMI4
clear;
addpath('Datasets/');
addpath('Functions/');
addpath('LRR');
addpath('FastESC-master');
addpath('FastESC-master/EBMM_Release');
addpath('SparsifiedKMeans-master');
addpath('SparsifiedKMeans-master/private');
data_num = 1;
dataset_name = {'40f','Umist','YaleB_3232','COIL20','mnist_all','CCC80','USPSfu'};
load(strcat(dataset_name {data_num},'.mat'))
if data_num == 1
    fea = double(data);
    interval = 5;%�����ȥ�������ļ��
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 2
    fea=X';
    interval = 2;%�����ȥ�������ļ��
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 3
    interval = 2;%�����ȥ�������ļ��
    ttime = 10;
elseif data_num == 4
    interval = 2;%�����ȥ�������ļ��
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 5
    interval = 2;%�����ȥ�������ļ��
    num = 400;
    fea = double([train0(1:num,:);train1(1:num,:);train2(1:num,:);train3(1:num,:);train4(1:num,:);train5(1:num,:);train6(1:num,:);train7(1:num,:);train8(1:num,:);train9(1:num,:)]);
    gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 6
    interval = 5;%�����ȥ�������ļ��
    ttime = floor(length(unique(gnd))/interval);
elseif data_num == 7
    dnum = 20;
    num = dnum;
    fea = [data0(:,1:dnum)';data1(:,1:dnum)';data2(:,1:dnum)';data3(:,1:dnum)';data4(:,1:dnum)';data5(:,1:dnum)';data6(:,1:dnum)';data7(:,1:dnum)';data8(:,1:dnum)';data9(:,1:dnum)'];
    fea = double(fea);
    gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
    interval = 2;%�����ȥ�������ļ��
    ttime = floor(length(unique(gnd))/interval);
end
data = fea;
datal = double(gnd);
for time =1:ttime
    kk = interval*time;%�����
    number = find(datal<=kk);%ÿһ��ĸ���
    %number=max(number);
    fea=data(number,:);%kk *number,:);  
    gnd=datal(number);%kk*number);
    fea=NormalizeFea(fea); 
%     [new1 b1 dis1] = LRSA(fea');
    [new2 b4 dis4] = LRRHWAP(fea',0.1,0.1,5,99);%yale���ݼ��õĵ���k=3
%     k=5;%knn��k
%     Y = fea';
%     Yg = fea';
%     a=fkNN(Yg,k);
%     b=constractmap(a);
%     c = transmit(b,0);
%     d = (c+c')/2;
%     d(find(d>0))=1;
%    [new,OBJ] =  sparse_graph_LRR(Yg,d);
   accuracy0 = zeros(1,10);
   accuracy1 = zeros(1,10);
   accuracy2 = zeros(1,10);
   accuracy3 = zeros(1,10);
   accuracy4 = zeros(1,10);
   accuracy5 = zeros(1,10);
   NMI0 = zeros(1,10);
   NMI1 = zeros(1,10);
   NMI2 = zeros(1,10);
   NMI3 = zeros(1,10);
   NMI4 = zeros(1,10);
   NMI5 = zeros(1,10);
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
    
    for i = 1:10
        c0 =  kmeans(fea,kk);
%         [c0,~] = kmeans_sparsified(fea, 2);
        idx=bestMap(gnd,c0); % ƥ��
        accuracy0(i) = length(find(gnd == idx))/length(gnd);% �ҵ�gnd���Ѿ�����ŵ�idx����ƥ���ֵ����������
        NMI0(i) = NormalizedMutualInformation(gnd,c0,length(gnd),kk); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        c1 =  NJW(fea,kk); 
        idx=bestMap(gnd,c1); % ƥ��
        accuracy1(i) = length(find(gnd == idx))/length(gnd);% �ҵ�gnd���Ѿ�����ŵ�idx����ƥ���ֵ����������
        NMI1(i) = NormalizedMutualInformation(gnd,c1,length(gnd),kk); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         c2 =  NJW(new,kk);
%         idx=bestMap(gnd,c2); % ƥ��
%         accuracy2(i) = length(find(gnd == idx))/length(gnd);% �ҵ�gnd���Ѿ�����ŵ�idx����ƥ���ֵ����������
%         NMI2(i) = NormalizedMutualInformation(gnd,c2,length(gnd),kk); 
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         c3 =  NJW(new1,kk);
%         idx = bestMap(gnd,c3); % ƥ��
%         accuracy3(i) = length(find(gnd == idx))/length(gnd);% �ҵ�gnd���Ѿ�����ŵ�idx����ƥ���ֵ����������
%         NMI3(i) = NormalizedMutualInformation(gnd,c3,length(gnd),kk); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        c4 =  NJW(new2,kk);
        idx=bestMap(gnd,c4); % ƥ��
        accuracy4(i)=length(find(gnd == idx))/length(gnd);% �ҵ�gnd���Ѿ�����ŵ�idx����ƥ���ֵ����������
        NMI4(i) = NormalizedMutualInformation(gnd,c4,length(gnd),kk);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         [~,c5] = sort(vESC, 2, 'descend');
%         idx=bestMap(gnd,c5(:,1)); % ƥ��
%         accuracy5(i)=length(find(gnd == idx))/length(gnd);
%         NMI5(i) = NormalizedMutualInformation(gnd,c5(:,1),length(gnd),kk); 
    end 
    accuracy0_m(time)=mean(accuracy0); 
    NMI0_m(time) = mean(NMI0); 
    accuracy1_m(time)=mean(accuracy1); 
    NMI1_m(time) = mean(NMI1);  
%     accuracy2_m(time)=mean(accuracy2);
%     NMI2_m(time) = mean(NMI2); 
%     accuracy3_m(time)=mean(accuracy3); 
%     NMI3_m(time) = mean(NMI3);  
    accuracy4_m(time)=mean(accuracy4);
    NMI4_m(time) = mean(NMI4);
%     accuracy5_m(time)=mean(accuracy5); 
%     NMI5_m(time) = mean(NMI5); 
end
 save(strcat('result/','revise',strcat(dataset_name{data_num},'g_in10p')))