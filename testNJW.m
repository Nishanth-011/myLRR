clear;
for time =1:10
    load COIL20;
    %load YaleBext_3232;
    %load ORL_32x32 
    %load Yale_64x64;
    %load AR_database_60_43 
    %load mnist_all
    %load PIE_32x32  
    %  feaa = NewTrain_DAT(:,1:kk*number); 
    %  fea =double(feaa'); 
    % gnd = trainlabels(1:kk*number); 
    % gnd = gnd';
    %load umist 

    time
    kk = 2*time ;%类别数
    number = find(gnd==kk);%每一类的个数
    number=max(number);
    %  feaa = NewTrain_DAT(:,1:kk*number);  
    %  fea =double(feaa'); 
    % gnd = trainlabels(1:kk*number); 
    % gnd = gnd'; 
    %fea=X';
    fea=fea(1:number,:);%kk *number,:); 
    gnd=gnd(1:number);%kk*number);
    % number=64;
    for i = 1:10
        c =  NJW(fea,kk); 
        %c =  kmeans(fea,kk); 

        idx=bestMap(gnd,c); % 匹配
        accuracy(i)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
        NMI(i) = NormalizedMutualInformation(gnd,c,length(gnd),kk);
    end 
    accuracy_m(time)=mean(accuracy); 
    MNI_m(time) = mean(NMI);
end