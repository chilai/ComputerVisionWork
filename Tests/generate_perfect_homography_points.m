%Generate random x points
a = -6;
b = 6;
x = a + (b-a)*rand(200,1);

%Generate random y points
y = a + (b-a)*rand(200,1);

%stack them up
in_ = [x';y']; 

%create Homography matrix
H = [3 1 -4; 2 9 5; 0 0 1];


%create correspondences 
in_new = [in_;ones(1,200)];
out_ = H*in_new;

out_ = out_./(out_(3,:));

out_ = out_(1:2,:);

A = [in_', out_'];
B = A + normrnd(0,0.03, 200,4);

%write to file
fileID = fopen('homography_no_noise.txt','w');
fprintf(fileID,'%3.0f \n', 200);
D = reshape(A',[],1);
fprintf(fileID,'%6.6f %6.6f %6.6f %6.6f \n',D);
fclose(fileID);

fileID = fopen('homography_noisy.txt','w');
fprintf(fileID,'%3.0f \n', 200);
D = reshape(B',[],1);
fprintf(fileID,'%6.6f %6.6f %6.6f %6.6f \n',D);
fclose(fileID);
