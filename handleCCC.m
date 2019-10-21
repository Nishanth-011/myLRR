clear;
addpath('Datasets/');
load CCC;
gnd = zeros(size(gtlabels));
num = 50
for i = 1:26
    gnd(find(gtlabels == char('A'-1+i))) = i;
    gnd(find(gtlabels == char('a'-1+i))) = i+26;
end
new = [];
newgnd = [];
for i =1:20
    new0 = fea(find(gnd ==i),:);
    newgnd0 = gnd(find(gnd ==i));
    new = [new;new0(1:num,:)];
    newgnd = [newgnd;newgnd0(1:num)];
end
fea = new;
gnd = newgnd;
save(strcat('G:/mycode/myLRR/Datasets/CCC',num2str(num)),'fea','gnd')