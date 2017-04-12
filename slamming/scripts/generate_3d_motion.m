%Generate random x points
a = -100;
b = 100;
x = a + (b-a)*rand(100,1);

%Generate random y points
y = a + (b-a)*rand(100,1);

%Generate random z points
z = a + (b-a)*rand(100,1);

%stack them up
in_ = [x';y';z'];

%Transform them by a rotation and translation
t = [0.04;0.02;0.003];
theta = 0.26;
R = [cos(theta), -sin(theta) 0; sin(theta), cos(theta), 0; 0,0,1];

out_ = R'*in_ + repmat(-R'*t,1, size(in_,2));


A = [in_', out_'];
B = A + normrnd(0,0.06, 100,6);

%write to file
fileID = fopen('3d_motion_no_noise.txt','w');
fprintf(fileID,'%3.0f \n', 100);
D = reshape(A',[],1);
fprintf(fileID,'%6.6f %6.6f %6.6f %6.6f %6.6f %6.6f \n',D);
fclose(fileID)

fileID = fopen('3d_motion_noisy.txt','w');
fprintf(fileID,'%3.0f \n', 100);
D = reshape(B',[],1);
fprintf(fileID,'%6.6f %6.6f %6.6f %6.6f %6.6f %6.6f \n',D);
fclose(fileID)
