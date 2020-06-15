%����ǲ������������Ĵ�����ݲ��ԵĲ�һ��ע��17��18�м���
clear;
addpath('Datasets/');
addpath('Functions/');
ttime = 11;
load ORL_32x32 
data = fea;
datal = gnd;
kkk=5;
parfor time =1:ttime
    kk = 20;%2*time2 ;%�����
    number = find(datal==kk);%ÿһ��ĸ���
    number=max(number);
    fea=data(1:number,:);%kk *number,:);  
    gnd=datal(1:number);%kk*number);
    fea=NormalizeFea(fea); 
    [new4 b4 dis4] =  LRRHWAP(fea',0.1,0.1*time,kkk,99);
    accuracy4 = zeros(1,10);
    for i = 1:10
        c4 =  NJW(new4,kk);
        idx=bestMap(gnd,c4); % ƥ��
        accuracy4(i)=length(find(gnd == idx))/length(gnd);% �ҵ�gnd���Ѿ�����ŵ�idx����ƥ���ֵ����������
    end 
    accuracy4_m(time)=mean(accuracy4);
    NMI4(time) = NormalizedMutualInformation(gnd,c4,length(gnd),kk); 
end
x = 0.1:0.1:1.1;
km = 0.4016*ones(1,11);
sc = 0.4459*ones(1,11);
fast = 0.3252*ones(1,11);
ns = 0.4894*ones(1,11);
ladp = 0.3917*ones(1,11);
plot(x,km,'-xr',x,sc,'-+k',x,fast,'-ok',x,ns,'-+b',x,ladp,'-ob',x,accuracy4_m,'-or','LineWidth',2); %���ԣ���ɫ�����
% set(gca,'XTick',[0.1:0.1:1.1]) %x�᷶Χ1-6�����1
% set(gca,'YTick',[0:0.1:1]) %y�᷶Χ0-700�����100
legend('Kmeans++','SC','FastESC','NSLLRR','LLRADP','LRRHWAP');   %���ϽǱ�ע
xlabel('The value of \lambda')  %x����������
ylabel('ACC') %y����������
axis([0.1 1 0 1]);
saveas(gcf, 'testlambda', 'fig')