close all;
pos = dlmread('posFileData');
neg = dlmread('negFileData');
B = pos > 0;
posLen = sum(B,2);
B = neg > 0;
negLen = sum(B,2);

posChars   = pos(:);
posChars   = posChars(posChars~=0);
[pElmts, pCntrs]   = hist(posChars,[1:22]);
figure;
bar(pCntrs, pElmts/max(pElmts), 'FaceColor', [1, 0.5, 0.5]);
title('Histogram of Amino Acid Occurences in Positive DataSet');
ylabel('Frequency - Normalized');
xlabel('Amino Acids');
ylim([-0.1, 1.1]); xlim([0, 22]);

negChars   = neg(:);
negChars   = negChars(negChars~=0);
[nElmts, nCntrs]   = hist(negChars,[1:22]);
figure;
bar(nCntrs, nElmts/max(nElmts));
title('Histogram of Amino Acid Occurences in Negative DataSet');
ylabel('Frequency - Normalized');
xlabel('Amino Acids');
ylim([-0.1, 1.1]); xlim([0,22]);

for i = 1:0
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