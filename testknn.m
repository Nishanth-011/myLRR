clear;
ttime = 38;
%K = 4;%the number of k nearest neighbor
%labelnumber = 1;
for time1 = 1:7%ttime
    for time2 =1:ttime
        K = time2;%the number of k nearest neighbor
        labelnumber = time1;
        time1*time2/ttime
        %load COIL20;
        %load umist;
        %load YaleBext_3232;
        load YaleB_3232;
        %load ORL_32x32 
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
        kk = 38 ;%nunmber of clustering
        number = find(gnd==kk);%每一类的个数
        number=max(number);
        fea=data(1:number,:);%kk *number,:);  
        gnd=datal(1:number);%kk*number);
        sum = 0;
        traindata = [];
        trainlabel = [];
        for i = 1:10
            sum = 0;
            for j =1:kk
                number = find(gnd==j);
                a =1+sum + round((length(number)-1)*rand(1,labelnumber));
                sum = round(sum + length(number));
                traindata = [traindata;fea(a,:)];
                trainlabel = [trainlabel;gnd(a,:)];
            end
            y = knn(fea',traindata',trainlabel',K);
            AC = y'-gnd;
            ac(i) = length(find(AC~=0))/length(gnd);
        end 
        acc(time1,time2) = mean(ac);
    end
end
bcc = 1.-acc;