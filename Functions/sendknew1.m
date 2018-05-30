function result = sendknew1(data, time, kn)
m = size(data,1);
dataold = zeros(m,m);
i = 1;
data0 = data;
while(~isequal(data,dataold))
    dataold = data;
    data1 = data*data0;
    data2 = data;
    data1(find(data1>0)) = 1;
    data1(find(data1<=0)) = 0;
    data2(find(data2>0)) = 1;
    data2(find(data2<=0)) = 0;
    data3 = data1 - data2;
    data3(find(data3>0)) = 1;
    data3(find(data3<=0)) = 0;
    %data3 = data3 - diag(diag(data3));
    data = data3*(kn)^i + data;
    i = i + 1;
end
data = data- diag(diag(data));
result = data;