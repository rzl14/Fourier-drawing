%% ������Ļ��������������
clear; clc; close all;

%% ����
gray_div = 0.8;    % ת��ֵͼ��ʱ����ֵ
fps = 1000;          % ÿ��֡��
fpf = 1;           % ÿ֡�����������
vec_num = 150;      % ��������������Ϊ vec_num*2+1��
start_delay = 0;    % ��ʼ��ͼǰ�ȵ�����


%% ԭʼ����׼��================================================================
%% GUIѡ����Ҫ�����ͼ���ļ�
% uigetfile��ͼ�ν�����ļ�
[filename, pathname, filterindex] = uigetfile(...
   {'*.*',      '�����ļ�(*.*)'; ...
    '*.bmp',    'ͼƬ(*.bmp)'; ...
    '*.jpg',    'ͼƬ(*.jpg)'}, ...
    '��ѡ��Ҫ�����ѡ��ͼƬ');

%% imread ��ȡѡ����ļ��������浽 input_image������ imshow ��ʾԭʼͼ��
input_image = imread([pathname filename]);      %�洢������uint8(matlabΪͼ���ṩ�������������ͣ���0~255������);
input_image = im2double(input_image);           %��һ����ת��double [0, 1];input_image = double(input_image)/255
                                                %matlabĬ��double����ͼƬ������λ��0~1֮���
img_gray = rgb2gray(input_image);               %�ҶȻ���rgb2gray�����ɫͼ�� RGB ת��Ϊ�Ҷ�ͼ��
                                                %�Ҷ�ͼ������ص�RGB��ֵ������ͬ�ģ���ͻ����һ������ͼ���ÿ�����ص����ɫ�Ǵ��ڰ�ɫ����ɫ֮��
%figure; imshow(input_image); title('ԭʼͼ��'); %��ʾԭʼͼ��

%% ͼ����================================================================
%% ��ֵ��
main_fig = figure;                              %figure ����ͼ������
set(main_fig, 'position', get(0, 'ScreenSize'));    %position��һ�����ԣ���main_fig���ڵ�λ��
                                                    %get(0, 'ScreenSize')��ȡ��Ļ�ֱ���
                                                    %���ã�ȫ����ʾ
img_bw = im2bw(img_gray, gray_div);         %im2bw():�ѻҶ�ͼ��ת���ɶ�ֵͼ��
                                            %gray_div��Ԥ�õģ���ΧΪ 0~1
                                            %��ֵͼ��ֻ�д��ڡ�����������ɫ��ͼ��ÿ������ȡֵֻ��0 ��1
subplot(3, 2, 1); imshow(img_gray); title('�Ҷ�ͼ��');
subplot(3, 2, 2); imshow(img_bw); title('��ֵ��ͼ��'); %subplot(m, n, p)�ǽ����ͼ����һ��ƽ���ϵĹ��ߡ����У�m ��ʾ p ��ͼ�ų� m �У�n ��ʾͼ�ų� n ��
img_bw_inv = ~img_bw;                               %ȡ������������ 0 ��� 1��1 ���0

%% ������ȡ
[M, N] = size(img_bw);                      %��ȡͼ���С
img_edge = edge(img_bw, 'sobel');           %edge():����һ����img_bw��ͬ��С�Ķ�ֵ��ͼ��img_edge���ں�����⵽��Ե�ĵط�Ϊ1�������ط�Ϊ0��
                                            %'sobel'�����������ӣ���һ�ֱȽϺõı�Ե��ⷽ��
subplot(3, 2, 3); imshow(img_edge); title('��Եͼ��');

%% ���
F1 = imfill(img_bw_inv,'holes');            %imfill():��һ�Ŷ�ֵͼ����ʾ����Ļ�ϣ� �����û�ʹ�������ͼ���ϵ㼸���㣬 �⼸����Χ�ɵ�����Ҫ��������
                                            %'holes'�����Զ�����ֵͼ��׶�����
subplot(3, 2, 4); imshow(F1); title('���ͼ��');

%% �߽����
img_bw_label = bwlabel(F1);         %��8��ͨ��������������
                                    %8��ͨ:һ�����أ�����������������ϡ��¡����ҡ����Ͻǡ����½ǡ����Ͻǻ����½������ţ�����Ϊ��������ͨ�ģ�
                                    %bwlabel(F1, n);n ��ȡֵ4��8
                                    %����һ����F1��С��ͬ�ľ��󣬰����˱����F1��ÿ����ͨ���������ǩ����һ����ͨ���򱻱��Ϊ1,�ڶ�����ͨ���򱻱��Ϊ2,��������
[pr, pc] = find(img_bw_label);      %�ҳ�����img_bw_label�е����з���Ԫ�ص��к��е�����ֵ���д洢��pr���д洢��pc
                                    %pr��pc���������þ��������һ������Ԫ����Ϊbwtraceboundary()����ʼ����
row = pr(4);                        %��4��Ԫ�أ���Ϊ��ʼ�������꣬4���������ѡ��һ����
col = pc(4);                        %��ʼ��������
connectivity = 8;                   %8��ͨ����
contour = bwtraceboundary(img_bw_label, [row, col], 'S', connectivity);
                                    %�ҳ���ֵͼ���Ե�㣬���������4��ͨ��8��ͨ�����������
                                    %[row, col]Ϊ��ʼ���ٵĵ���к�������
                                    %'S'Ϊ��ʼ�������򣬼���������
                                    %connectivity�̻��˸��ٱ߽�������ԣ���ֵ����Ϊ4����8����ͼ�Ѿ����ӳ���8��ͨ�߽磬���ȡֵΪ8
route = contour;
route(:,1)=route(:,1)*(-1)+N/2;     %contour ������������ͼƬ���Ͻ�Ϊԭ�㣬y�ᣨ��Ӧ��һ�е�ֵ��ָ����
                                    %��������꣬�˴�����-1֮��ʹͼƬ��ת
route(:,2)=route(:,2)-M/2;          %�����������N/2���������ȥM/2��ʹͼƬ������Ϊԭ��
subplot(3, 2, 5);
plot(route(:,2), route(:,1), 'g', 'LineWidth', 2), axis('equal');
                                    %��ʾ��route(:,2)Ϊx���꣬route(:,1)Ϊy���꣬������������'g'��ʾ��ɫ������'LineWidth',2��ʾ�߿�Ϊ2
title('�߽����');

%% ����Ҷ����================================================================
%% ���ɸ�ƽ��·��
mapsize = max(N,M);
route_c = (route(:,1)*1i/mapsize+route(:,2)/mapsize);
                                    %��y����route(:,1)����mapsize�ٳ���1i���õ���ƽ�����������
                                    %��x����route(:,2)����mapsize�õ���ƽ���ʵ������
                                    %��������ӵõ���ƽ�����ֵ
subplot(3, 2, 6);
plot(route_c), axis('equal');       %matlab����ͼ��ܶ�ʱ��Ϊ�˰�x��y����Ϣ�����ó�����ԣ���ʹx����Ƶĵ�λ���Ⱥ�y����Ƶĵ�λ���Ȳ�һ����
                                    %axis('equal')����ȡ�����ֱ仯
title('��ƽ��ͼ��');

%% ����Ҷ�任
% �����ĸ���Ҷ�任����Ƶ���븺Ƶ�ʲ�����ȣ���ͷӦ��һ��һ����
% һ��������1�룬һ�� tot_num ���켣������
tot_num = size(route_c, 1);         %����rote_c����������¼��tot_num
dt = 1/tot_num;
%vec_num = 100;
c=zeros(vec_num, 2);    %zeros(m,n) ����һ��m��n�е������
                        %c��¼ÿ����ͷ��ϵ����c[k][1] ������˳ʱ�룩��c[k][2]�Ǹ�����ʱ�룩
                        %vec_numΪ������������Ԥ�õĲ���
for k = 1:vec_num       %�����ѭ������c[1]��ʼ��������ϵ��
    for sign = 1:2
        tmp = 0;        %����
        for t_id = 1:tot_num
            tmp = tmp + route_c(t_id)*exp((-1)^sign*k * 2i*pi *(t_id*dt))*dt;
        end
        c(k, sign) = tmp;
    end
end
c0 = sum(route_c)*dt;


%% ���ƶ���
%subplot(3, 2, 6);
figure;
draw_orbit = plot([0, 0], [0, 0], 'LineWidth', 1, 'Color', [.6 .6 .6]);  % �켣
hold on;
draw_arrows = plot([0, 0], [0, 0], 'LineWidth', 1, 'Color', [1 0 0]);    % ����    
hold on;
draw_endpoint = plot(0, 0, 'k.', 'MarkerSize', 10);
map_min = min(min(real(route_c)), min(imag(route_c)))*1.2;
map_max = max(max(real(route_c)), max(imag(route_c)))*1.2;
xlim([map_min, map_max]);
ylim([map_min, map_max]);
%xlim([min(real(route_c))*1.2, max(real(route_c))*1.2]);
%ylim([min(imag(route_c))*1.2, max(imag(route_c))*1.2]);  %�̶����귶Χ
axis square;
title({'����', ['arrnum=',num2str(vec_num)]});


data_orbit = [];
data_arrows = zeros(vec_num*2, 1);
data_endpoint = 0;
tot_arrow = vec_num*2+1;

pause(start_delay);
for t_id = 1:fpf:tot_num %ÿ��ʱ��
    t = t_id * dt;
    %c0 ����
    data_arrows(1) = c0;
    %��ǰʱ��ÿ��Ƶ�ʵ�����
    for k = 1:vec_num
        data_arrows(k*2) = c(k, 1)*exp(k*2i*pi*t);
        data_arrows(k*2+1)   = c(k, 2)*exp(-k*2i*pi*t); 
    end
    %���Ӻ�ÿ��Ƶ�ʵ������յ��λ��
    for k = 2:tot_arrow
        data_arrows(k) = data_arrows(k-1) + data_arrows(k);
    end
    %�켣���յ�
    data_endpoint = data_arrows(tot_arrow);
    data_orbit = [data_orbit, data_endpoint];
    
    %����ͼ��
    set(draw_orbit, 'xdata', real(data_orbit), 'ydata', imag(data_orbit));
    set(draw_endpoint, 'xdata', real(data_endpoint), 'ydata', imag(data_endpoint));
    set(draw_arrows, 'xdata', real(data_arrows), 'ydata', imag(data_arrows));
    
    pause(1/fps);
end
