clear;
% <<<<<<< HEAD
% =======
% <<<<<<< HEAD
% >>>>>>> a50b32285e197bde7a9f052ddccc9df1cb606843
ttime = 1;
for time1 = 1:1%ttime
    for time2 =1:ttime
        time1*time2/ttime
        %load COIL20;
        %load umist;
        %load YaleBext_3232;
        %load YaleB_3232;
        load ORL_32x32 
        %load Yale_32x32;
        %load AR_database_60_43  
        %load mnist_all
        %load PIE_32x32  
        %load umist
        %load coil20_64x64
        %  
        %  feaa = NewTrain_DAT;   
        %  fea =double(feaa'); 
        % gnd = trainlabels; 
        % gnd = gnd'; 
        %fea=X';
        data = fea;
        datal = gnd;
        kk = 4*time2 ;%类别数
        number = find(gnd==kk);%每一类的个数
        number=max(number);
        fea=data(1:number,:);%kk *number,:);  
        gnd=datal(1:number);%kk*number);
        kkk=5;
        fea=NormalizeFea(fea); 
        %[new1 b1 dis1] = LRSA(fea');
        %[new2 b2 dis2] = LRSA1(fea',0.1,0.1,kkk);
        %[new3 b3 dis3] = LRSA2(fea',0.1,0.1,3);
        [new4 b4 dis4] = LRSA3(fea',0.1,0.1,5,99);
        k=5;%knn的k
        Y = fea';
        Yg = fea';
        a=fkNN(Yg,k);
        b=constractmap(a);
        c = transmit(b,0);
        d = (c+c')/2;
        d(find(d>0))=1;
       %[new,OBJ] =  sparse_graph_LRR(Yg,d);
        for i = 1:10
%             c0 =  NJW(new,kk);
%             idx=bestMap(gnd,c0); % 匹配
%             accuracy0(i) = length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
%             c1 =  NJW(new1,kk);
%             idx = bestMap(gnd,c1); % 匹配
%             accuracy1(i) = length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
%             c2 =  NJW(new2,kk);
%             idx=bestMap(gnd,c2); % 匹配
%             accuracy2(i)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
%             c3 =  NJW(new3,kk);
%             idx=bestMap(gnd,c3); % 匹配
%             accuracy3(i)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
            c4 =  NJW(new4,kk);
            idx=bestMap(gnd,c4); % 匹配
            accuracy4(i)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
        end 
        %accuracy0_=max(accuracy0);
%         accuracy0_m(time1,time2)=mean(accuracy0);
%         NMI0(time1,time2) = NormalizedMutualInformation(gnd,c0,length(gnd),kk); 
%         %accuracy0_t=std(accuracy0);
%         %accuracy1_=max(accuracy1);
%         accuracy1_m(time1,time2)=mean(accuracy1);
%         NMI1(time1,time2) = NormalizedMutualInformation(gnd,c1,length(gnd),kk); 
%         %accuracy1_t=std(accuracy1);
%         % accuracy2_=max(accuracy2);
%         accuracy2_m(time1,time2)=mean(accuracy2);
%         NMI2(time1,time2) = NormalizedMutualInformation(gnd,c2,length(gnd),kk); 
%         % accuracy2_t=std(accuracy2);
%         %accuracy3_=max(accuracy3);
%         accuracy3_m(time1,time2)=mean(accuracy3);
%         NMI3(time1,time2) = NormalizedMutualInformation(gnd,c3,length(gnd),kk); 
        %accuracy3_t=std(accuracy3);
        %accuracy4_=max(accuracy4);
        accuracy4_m(time1,time2)=mean(accuracy4);
        NMI4(time1,time2) = NormalizedMutualInformation(gnd,c4,length(gnd),kk); 
        %accuracy4_t=std(accuracy4);
        % accuracy0t(l) = accuracy0_m;
        % accuracy2t(l) = accuracy2_m;
        % accuracy3t(l) = accuracy3_m;
        % accuracy4t(l) = accuracy4_m;
        % end
        %end
    end
end