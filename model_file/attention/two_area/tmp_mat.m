
%%
a = [1 1 1 1 1 1 1 2 2 2 2 2];

idx = findchangepts(a, 'statistic', 'mean')


%%
Fs = 1000;
t = 0:1/Fs:1-1/Fs;

x = cos(2*pi*100*t) + sin(2*pi*200*t) + 0.5*randn(size(t));
y = 0.5*cos(2*pi*100*t - pi/4) + 0.35*sin(2*pi*200*t - pi/2) + 0.5*randn(size(t));
%%
Fs = 1000;
t = 0:1/Fs:1-1/Fs;

x1 = cos(2*pi*100*t);
y1 = 0.5*cos(2*pi*100*t - pi/4);
y2 = 0.5*cos(2*pi*100*t + pi/4);
x2 = cos(2*pi*100*t);
%%
x = [x1,x2];
y = [y1,y2];
x = x + 0.2*randn(size(x));
y = y + 0.2*randn(size(y));

%%
x = [x1,x1];
x = x + 0.2*randn(size(x));
y = [y1,y1];
y = y + 0.2*randn(size(y));
%%
figure;
plot(x)
hold on
plot(y)

%%
[Cxy,F] = mscohere(x,y,hamming(100),80,100,Fs);

figure;
plot(F,Cxy)
title('Magnitude-Squared Coherence')
xlabel('Frequency (Hz)')
grid

%%
[Pxy,F] = cpsd(x,y,hamming(100),80,100,Fs);

Pxy(Cxy < 0.2) = 0;

figure;
plot(F,angle(Pxy)/pi)
title('Cross Spectrum Phase')
xlabel('Frequency (Hz)')
ylabel('Lag (\times\pi rad)')
grid

%%
x = (0:2000-1)/1000;
d = sin(2*pi*4*x);
%%
coef = cwt(d,0.001);




