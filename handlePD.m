clear;
addpath('Datasets/');
% load CCC;
load data_PD
gnd = data(:,1);
fea = data(:,2:17);
gnd = gnd + 1;
num = 200;
newfea = [];
newgnd = [];
for i = 1:10
    place = find(gnd==i);
    newfea = [newfea;fea(place(1:num),:)];
    newgnd = [newgnd;gnd(place(1:num))];
end
fea = newfea;
gnd = newgnd;
save(strcat('Datasets/PD',num2str(num)),'fea','gnd');
