%����ǲ������������Ĵ�����ݲ��ԵĲ�һ��ע��17��18�м���
clear;
addpath('Datasets/');
addpath('Functions/');
ttime = 99;
load ORL_32x32 
data = fea;
datal = gnd;
for time =1:ttime
    kk = 10;%2*time2 ;%�����
    number = find(datal==kk);%ÿһ��ĸ���
    number=max(number);
    fea=data(1:number,:);%kk *number,:);  
    gnd=datal(1:number);%kk*number);
    kkk=5;
    fea=NormalizeFea(fea); 
    [new4 b4 dis4] =  LRRHWAP(fea',0.1,0.1,kkk,time);
    accuracy4 = zeros(1,10);
    for i = 1:10
        c4 =  NJW(new4,kk);
        idx=bestMap(gnd,c4); % ƥ��
        accuracy4(i)=length(find(gnd == idx))/length(gnd);% �ҵ�gnd���Ѿ�����ŵ�idx����ƥ���ֵ����������
    end 
    accuracy4_m(time)=mean(accuracy4);
    NMI4(time) = NormalizedMutualInformation(gnd,c4,length(gnd),kk); 
end
plot(1:ttime,accuracy4_m);
saveas(gcf, 'testk', 'fig')