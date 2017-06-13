number=72*4;%图片的数量
filepath='.\pic\';%图片路径
Z=[];%变换矩阵
E=[];%噪声矩阵
data=[];%原始数据
Y=[];%变换前数据
Ynew=[];%还原后的数据
namelist=[];%图片名单，目前是按照顺序排列的
k=8;%kNN的k
file_path =  '.\pic\';% 图像文件夹路径  
img_path_list = dir(strcat(file_path,'*.bmp'));%获取该文件夹中所有jpg格式的图像  
img_num = length(img_path_list);%获取图像总数量  
if img_num > 0 %有满足条件的图像  
        for j = 1:img_num %逐一读取图像  
            image_name = img_path_list(j).name;% 图像名  
            image = imread(strcat(file_path,image_name));
            [m,n]=size(image);
            form=reshape(image,m*n,1);
            data=[data,form];
        end
end
% for i=1:number
%     if i<=72
%         filename=strcat('obj','1','__',num2str(i-1),'.png');%obj1__0.png
%     elseif i<=144
%         filename=strcat('obj','9','__',num2str(i-73),'.png');%
%     elseif i<=144+72
%          filename=strcat('obj','16','__',num2str(i-73-72),'.png');%
%     else
%          filename=strcat('obj','2','__',num2str(i-73-72-72),'.png');%
%     end
%     %namelist=[namelist,filename];
%     image=imread(strcat(filepath,filename));
%     [m,n]=size(image);
%     form=reshape(image,m*n,1);
%     data=[data,form];
% end
Y=double(data);
Yg=mapminmax(Y);%归一化
a=fkNN(Yg,k);
[m,n]=size(Y);
W=zeros(n,n);
%这一段的内容是查找是否在k近邻里面
for i=1:n
   aa=a(i,1:k);
   %aa(1)=0;%把近邻矩阵里面的第一个元素置0，也就是没有元素本身
   for j=1:n
       if any(aa==j)
           W(i,j)=1;
           W(j,i)=1;
       end
   end
   W(i,i)=0;
end
%****************************
%*********************************
%这一段是是距离扩张
% for i=1:n
%     aa=a(i,k+1);
%     for j=1:n
%         if norm(Yg(:,i)-Yg(:,j))<=aa
%             W(i,j)=1;
%             W(j,i)=1;
%         end
%     end
% end
[Z, E] = sparse_graph_LRR(Y,W);%, 0.1, 5, 8.0, 2.5, 0);
Ynew=Y*Z+E;