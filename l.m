%% 清理屏幕、工作区、窗口
clear; clc; close all;

%% 参数
gray_div = 0.8;    % 转二值图像时的阈值
fps = 1000;          % 每秒帧数
fpf = 1;           % 每帧间隔采样点数
vec_num = 150;      % 向量数量（总数为 vec_num*2+1）
start_delay = 0;    % 开始做图前等的秒数


%% 原始数据准备================================================================
%% GUI选择需要处理的图形文件
% uigetfile：图形界面打开文件
[filename, pathname, filterindex] = uigetfile(...
   {'*.*',      '所有文件(*.*)'; ...
    '*.bmp',    '图片(*.bmp)'; ...
    '*.jpg',    '图片(*.jpg)'}, ...
    '请选择要处理的选择图片');

%% imread 读取选择的文件，并保存到 input_image，利用 imshow 显示原始图像
input_image = imread([pathname filename]);      %存储类型是uint8(matlab为图像提供的特殊数据类型，是0~255的整数);
input_image = im2double(input_image);           %归一化，转成double [0, 1];input_image = double(input_image)/255
                                                %matlab默认double类型图片数据是位于0~1之间的
img_gray = rgb2gray(input_image);               %灰度化，rgb2gray将真彩色图像 RGB 转换为灰度图像
                                                %灰度图像的像素的RGB三值都是相同的，这就会出现一个现象，图像的每个像素点的颜色是处于白色到黑色之间
%figure; imshow(input_image); title('原始图像'); %显示原始图像

%% 图像处理================================================================
%% 二值化
main_fig = figure;                              %figure 创建图窗窗口
set(main_fig, 'position', get(0, 'ScreenSize'));    %position是一种属性，是main_fig窗口的位置
                                                    %get(0, 'ScreenSize')获取屏幕分辨率
                                                    %作用：全屏显示
img_bw = im2bw(img_gray, gray_div);         %im2bw():把灰度图像转换成二值图像
                                            %gray_div是预置的，范围为 0~1
                                            %二值图像：只有纯黑、纯白两种颜色的图像，每个像素取值只有0 和1
subplot(3, 2, 1); imshow(img_gray); title('灰度图像');
subplot(3, 2, 2); imshow(img_bw); title('二值化图像'); %subplot(m, n, p)是将多个图画到一个平面上的工具。其中，m 表示 p 个图排成 m 行，n 表示图排成 n 列
img_bw_inv = ~img_bw;                               %取反，在这里是 0 变成 1，1 变成0

%% 轮廓提取
[M, N] = size(img_bw);                      %获取图像大小
img_edge = edge(img_bw, 'sobel');           %edge():返回一个与img_bw相同大小的二值化图像img_edge，在函数检测到边缘的地方为1，其他地方为0。
                                            %'sobel'是索贝尔算子，是一种比较好的边缘检测方法
subplot(3, 2, 3); imshow(img_edge); title('边缘图像');

%% 填充
F1 = imfill(img_bw_inv,'holes');            %imfill():将一张二值图像显示在屏幕上， 允许用户使用鼠标在图像上点几个点， 这几个点围成的区域即要填充的区域。
                                            %'holes'用于自动填充二值图像孔洞区域
subplot(3, 2, 4); imshow(F1); title('填充图像');

%% 边界跟踪
img_bw_label = bwlabel(F1);         %把8连通的区域连接起来
                                    %8连通:一个像素，如果和其他像素在上、下、左、右、左上角、左下角、右上角或右下角连接着，则认为他们是联通的；
                                    %bwlabel(F1, n);n 可取值4或8
                                    %返回一个和F1大小相同的矩阵，包含了标记了F1中每个连通区域的类别标签，第一个连通区域被标记为1,第二个连通区域被标记为2,依此类推
[pr, pc] = find(img_bw_label);      %找出矩阵img_bw_label中的所有非零元素的行和列的索引值，行存储到pr，列存储到pc
                                    %pr、pc向量的作用就是随便找一个非零元素作为bwtraceboundary()的起始坐标
row = pr(4);                        %第4个元素，作为起始点行坐标，4可以是随便选的一个数
col = pc(4);                        %起始点列坐标
connectivity = 8;                   %8连通区域
contour = bwtraceboundary(img_bw_label, [row, col], 'S', connectivity);
                                    %找出二值图像边缘点，可输出基于4连通或8连通区域的轮廓。
                                    %[row, col]为开始跟踪的点的行和列坐标
                                    %'S'为初始搜索方向，即向下搜索
                                    %connectivity刻画了跟踪边界的连续性，其值可以为4或者8，此图已经连接出了8连通边界，因此取值为8
route = contour;
route(:,1)=route(:,1)*(-1)+N/2;     %contour 出来的坐标是图片左上角为原点，y轴（对应第一列的值）指向下
                                    %因此行坐标，此处乘以-1之后，使图片翻转
route(:,2)=route(:,2)-M/2;          %再让坐标加上N/2，列坐标减去M/2，使图片中心作为原点
subplot(3, 2, 5);
plot(route(:,2), route(:,1), 'g', 'LineWidth', 2), axis('equal');
                                    %表示以route(:,2)为x坐标，route(:,1)为y坐标，画出的线条，'g'表示绿色线条，'LineWidth',2表示线宽为2
title('边界跟踪');

%% 傅里叶分析================================================================
%% 生成复平面路径
mapsize = max(N,M);
route_c = (route(:,1)*1i/mapsize+route(:,2)/mapsize);
                                    %令y坐标route(:,1)除以mapsize再乘以1i，得到复平面的虚轴坐标
                                    %令x坐标route(:,2)除以mapsize得到复平面的实轴坐标
                                    %两坐标相加得到复平面的数值
subplot(3, 2, 6);
plot(route_c), axis('equal');       %matlab函数图象很多时候为了把x和y的信息都表达得充分明显，会使x轴绘制的单位长度和y轴绘制的单位长度不一样。
                                    %axis('equal')可以取消这种变化
title('复平面图像');

%% 傅里叶变换
% 复数的傅里叶变换，正频率与负频率不再相等，箭头应该一正一反。
% 一个周期是1秒，一共 tot_num 个轨迹采样点
tot_num = size(route_c, 1);         %返回rote_c的行数，记录在tot_num
dt = 1/tot_num;
%vec_num = 100;
c=zeros(vec_num, 2);    %zeros(m,n) 返回一个m行n列的零矩阵
                        %c记录每个箭头的系数，c[k][1] 是正（顺时针），c[k][2]是负（逆时针）
                        %vec_num为向量数量，是预置的参数
for k = 1:vec_num       %最外层循环，从c[1]开始求像函数的系数
    for sign = 1:2
        tmp = 0;        %积分
        for t_id = 1:tot_num
            tmp = tmp + route_c(t_id)*exp((-1)^sign*k * 2i*pi *(t_id*dt))*dt;
        end
        c(k, sign) = tmp;
    end
end
c0 = sum(route_c)*dt;


%% 绘制动画
%subplot(3, 2, 6);
figure;
draw_orbit = plot([0, 0], [0, 0], 'LineWidth', 1, 'Color', [.6 .6 .6]);  % 轨迹
hold on;
draw_arrows = plot([0, 0], [0, 0], 'LineWidth', 1, 'Color', [1 0 0]);    % 向量    
hold on;
draw_endpoint = plot(0, 0, 'k.', 'MarkerSize', 10);
map_min = min(min(real(route_c)), min(imag(route_c)))*1.2;
map_max = max(max(real(route_c)), max(imag(route_c)))*1.2;
xlim([map_min, map_max]);
ylim([map_min, map_max]);
%xlim([min(real(route_c))*1.2, max(real(route_c))*1.2]);
%ylim([min(imag(route_c))*1.2, max(imag(route_c))*1.2]);  %固定坐标范围
axis square;
title({'动画', ['arrnum=',num2str(vec_num)]});


data_orbit = [];
data_arrows = zeros(vec_num*2, 1);
data_endpoint = 0;
tot_arrow = vec_num*2+1;

pause(start_delay);
for t_id = 1:fpf:tot_num %每个时刻
    t = t_id * dt;
    %c0 重心
    data_arrows(1) = c0;
    %当前时刻每个频率的向量
    for k = 1:vec_num
        data_arrows(k*2) = c(k, 1)*exp(k*2i*pi*t);
        data_arrows(k*2+1)   = c(k, 2)*exp(-k*2i*pi*t); 
    end
    %叠加后每个频率的向量终点的位置
    for k = 2:tot_arrow
        data_arrows(k) = data_arrows(k-1) + data_arrows(k);
    end
    %轨迹与终点
    data_endpoint = data_arrows(tot_arrow);
    data_orbit = [data_orbit, data_endpoint];
    
    %更新图像
    set(draw_orbit, 'xdata', real(data_orbit), 'ydata', imag(data_orbit));
    set(draw_endpoint, 'xdata', real(data_endpoint), 'ydata', imag(data_endpoint));
    set(draw_arrows, 'xdata', real(data_arrows), 'ydata', imag(data_arrows));
    
    pause(1/fps);
end
