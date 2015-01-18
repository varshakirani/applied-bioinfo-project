pos = dlmread('posFileData');
neg = dlmread('negFileData');
B = pos > 0;
posLen = sum(B,2);
B = neg > 0;
negLen = sum(B,2);

for i = 1:100
   d   = randi(1320);
figure;
plot(pos(d,:)','.-b');
hold on;
plot(neg(d,:)','.-r');
pause;
close;
end
% binranges = -1:23;
% [bincounts] = histc(pos,binranges)
% figure;
%h = histogram(pos);
% bar(binranges,bincounts,'histc');