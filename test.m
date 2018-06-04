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
ttime = 1;%这个是实验次数
interval = 5;%这个是去样本数的间隔
addpath('Datasets/');
addpath('Functions/');
addpath('LRR');
for time1 = 1:1
    for time2 =1:ttime
        %load COIL20;
        %load YaleBext_3232;
        load ORL_32x32 
        %%%%%%%%%%umist需要这两个行一起注释
        %load umist 
        %fea=X';
        %%%%%%%%%%%%%
        data = fea;
        datal = gnd;
        kk = interval*time2;%类别数
        number = find(gnd==kk);%每一类的个数
        number=max(number);
        fea=data(1:number,:);%kk *number,:);  
        gnd=datal(1:number);%kk*number);
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
        for i = 1:10
            c0 =  kmeans(fea,kk);
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
        accuracy0_m(time1,time2)=mean(accuracy0);
        NMI0(time1,time2) = NormalizedMutualInformation(gnd,c0,length(gnd),kk); 
        accuracy1_m(time1,time2)=mean(accuracy1);
        NMI1(time1,time2) = NormalizedMutualInformation(gnd,c1,length(gnd),kk); 
        accuracy2_m(time1,time2)=mean(accuracy2);
        NMI2(time1,time2) = NormalizedMutualInformation(gnd,c2,length(gnd),kk); 
        accuracy3_m(time1,time2)=mean(accuracy3);
        NMI3(time1,time2) = NormalizedMutualInformation(gnd,c3,length(gnd),kk); 
        accuracy4_m(time1,time2)=mean(accuracy4);
        NMI4(time1,time2) = NormalizedMutualInformation(gnd,c4,length(gnd),kk); 
    end
end