clear;
<<<<<<< HEAD
%load YaleBext_3232;
load ORL_32x32;
numC = 12 ;%类别数
knum = 5;
=======
load YaleBext_3232;
numC = 4 ;%类别数
knum = 2;
>>>>>>> a50b32285e197bde7a9f052ddccc9df1cb606843
number = find(gnd==numC);%每一类的个数
number=max(number);
X = fea(1:number,:);
Xgnd = gnd(1:number);
[b,c]=fkNN(X',knum);
aa = constractmap(b);%求出连接矩阵
<<<<<<< HEAD
%aa = aa+aa';
=======
aa = aa+aa';
>>>>>>> a50b32285e197bde7a9f052ddccc9df1cb606843
aa(find(aa>0)) = 1;
%bb = transmit(aa);
aaa = newtrans(aa,knum);
sumlist = sum(aaa,2);
%isequal(bb,aaa)
<<<<<<< HEAD
%使用单个度最大的点
=======
>>>>>>> a50b32285e197bde7a9f052ddccc9df1cb606843
for i = 1:size(aaa,1)
    list = aa(i,:);
    list = find(list>0);
    [maxvalue(i), maxlist(i)] = max(sumlist(list));
    maxlist(i) = list(maxlist(i));
end
<<<<<<< HEAD
%使用三个最大的度投票

=======
>>>>>>> a50b32285e197bde7a9f052ddccc9df1cb606843
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
<<<<<<< HEAD
end
max(accuracy1)
max(accuracy2)
=======
end
>>>>>>> a50b32285e197bde7a9f052ddccc9df1cb606843
