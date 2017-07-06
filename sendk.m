%这个函数是用来对近邻进行传播关系的
%time是传播的次数
%k是加权
function result = sendk(data, time, kn)
[m,~] = size(data);
if nargin<3
    kn = 1;
elseif nargin<2
    time = m;
end
datanew = data;
dataold = data;
for i = 1:time
    for j = 1:m
        listn = find(datanew(j,:)>0);%找到大于0的每一项
        newsum = sum(data(listn,:));%
        listsum = find(newsum>0);
        listk = setdiff(listsum,listn);%
        if(isempty(listk))
            continue;
        end
        datanew(j,listk) = kn^i;
    end
    if dataold == datanew
        break;
    end
    dataold = datanew;
end
result = dataold;
        
        