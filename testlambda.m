%����ǲ������������Ĵ�����ݲ��ԵĲ�һ��ע��17��18�м���
clear;
ttime = 11;
for time1 = 1:1%ttime
    for time2 =1:ttime
        time1*time2/ttime
        load ORL_32x32 
        data = fea;
        datal = gnd;
        kk = 10%2*time2 ;%�����
        number = find(gnd==kk);%ÿһ��ĸ���
        number=max(number);
        fea=data(1:number,:);%kk *number,:);  
        gnd=datal(1:number);%kk*number);
        kkk=5;
        fea=NormalizeFea(fea); 
        [new4 b4 dis4] =  LRRHWAP(fea',0.1,0.05*2^(i-1),5,99);
        %[new4 b4 dis4] =  LRRHWAP(fea',0.05*2^(i-1),0.1,5,99);
        for i = 1:10
            c4 =  NJW(new4,kk);
            idx=bestMap(gnd,c4); % ƥ��
            accuracy4(i)=length(find(gnd == idx))/length(gnd);% �ҵ�gnd���Ѿ�����ŵ�idx����ƥ���ֵ����������
        end 
        accuracy4_m(time1,time2)=mean(accuracy4);
        NMI4(time1,time2) = NormalizedMutualInformation(gnd,c4,length(gnd),kk); 
    end
end