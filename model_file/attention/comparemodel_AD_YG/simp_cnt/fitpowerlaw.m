%%
ie = ie(:);
rate_m = rate_m(:);
%%
ie1 = ie(4:19); rate1 = rate_m(4:19);
ie2 = ie(25:65); rate2 = rate_m(25:65);
ie3 = ie(70:end); rate3 = rate_m(70:end);
%%
[fitobject1,gof1] = fit(ie1, rate1,'power2');
%%
[fitobject2,gof2] = fit(ie2, rate2,'power2','StartPoint', [9.3345e+09 -17.1509 1.1622]);
%%
[fitobject3,gof3] = fit(ie3, rate3,'power2','StartPoint', [9.3345e+09 -17.1509 1.1622]);


%%
rate_fit1 = fitobject1.a*(ie1.^fitobject1.b) + fitobject1.c;

rate_fit2 = fitobject2.a*(ie2.^fitobject2.b) + fitobject2.c;

rate_fit3 = fitobject3.a*(ie3.^fitobject3.b)+ fitobject3.c;%+ fitobject2.c;

%%
figure;
% plot(ie(1:11), rate_fit1,'r*')
% hold on
plot(ie1, rate_fit1,'b-')
hold on
plot(ie2, rate_fit2,'g-')
hold on
plot(ie3, rate_fit3,'r-')

plot(ie(1:end), rate_m(1:end),'*')
set(gca, 'YScale', 'log')
%%
figure;
plot(ie(18:end), rate_fit,'r')
hold on
plot(ie(18:end), rate_m(18:end),'-.')
%%
figure;

plot(ie(1:end), rate_m(1:end),'r')
set(gca, 'YScale', 'log')
%%
figure;
plot(ie(1:11), rate_fit1,'r*')
hold on
plot(ie(13:end), rate_fit2,'r*')
hold on
plot(ie(1:end), rate_m(1:end),'-.')
set(gca, 'YScale', 'log')
%%
figure;
% plot(ie(1:11), rate_fit1,'r*')
% hold on
plot(ie(8:36), rate_fit2,'-')
hold on
plot(ie(45:80), rate_fit1,'b-')

plot(ie(8:80), rate_m(8:80),'*')
set(gca, 'YScale', 'log')
%%
figure;
cwt(mua,0.001);

