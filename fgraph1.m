clear;
%load YaleBext_3232;
load ORL_32x32;
numC = 12 ;%类别数
knum = 5;
number = find(gnd==numC);%每一类的个数
number=max(number);
X = fea(1:number,:);
Xgnd = gnd(1:number);
[b,c]=fkNN(X',knum);
aa = constractmap(b);%求出连接矩阵
aa = aa+aa';
aa(find(aa>0)) = 1;
sumlist = sum(aa,2);
%isequal(bb,aaa)
%使用单个度最大的点
for i = 1:size(aa,1)
    list = aa(i,:);
    list = find(list>0);
    [maxvalue(i), maxlist(i)] = max(sumlist(list));
    maxlist(i) = list(maxlist(i));
end
%使用三个最大的度投票

helist = unique(maxlist);
for time = 1:10
    data = X(helist,:);
    label = NJW(data,2);
    for i = 1:number
        aaaa = find(helist == maxlist(i));
        label1(i) = label(aaaa);
    end
    label2 = NJW(X,numC);
    accuracy1(time)=length(find(Xgnd == label1'))/length(Xgnd);
    accuracy2(time)=length(find(Xgnd == label2))/length(Xgnd);
end
max(accuracy1)
max(accuracy2)