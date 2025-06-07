function [X,t]=demo_2dwell(~)

%>>>>> parameters >>>>>>>
a = 1.5; % Levy tail exponent
p.sigma2 = 0.1; % sigma^2 where sigma is the width of the well
p.momentum = 0; % momentum coefficient
p.gam = 1; % strength of the Levy noise
p.dt = 1e-3; % integration time step
T = 1e2; % total simulation time
p.location = pi/2; %modal location 
p.skip = 100; %number of time steps to skip for the video
%<<<<<< parameters <<<<<<<<

tic
disp('Generating raw data...')
[X,t] = fHMC_2d(T,a,p);
clc
toc
    
n = floor(T/p.dt); %number of samples
t = (0:n-1)*p.dt;

close all

show_video(X,t,p);

figure('color','w');
subplot(2,1,1)
plot(t,X(1,:))
xlabel('t')
ylabel('x')
subplot(2,1,2)
plot(t,X(2,:))
xlabel('t')
ylabel('y')

figure('color','w');
plot(X(1,:),X(2,:),'.','markersize',1)
hold on
plot([1 -1]*p.location, [0 0],'*r')
xlabel('x')
ylabel('y')
axis equal

end

function [X,t] = fHMC_2d(T,a,p)
%Levy Monte Carlo on a circle
%INPUT: T = time span (s), a = Levy characteristic exponent
% p = other parameters
%OUTPUT: X = samples, t = time
if isempty(p)    
    p.location = pi/2;
    p.sigma2 = 1;
    p.gam = 1;
    p.dt = 1e-3;
    p.momentum = 1;
end

m = 2; %dimension

dt = p.dt;%1e-3; %integration time step (s)
dta = dt.^(1/a); %fractional integration step

n = floor(T/dt); %number of samples
t = (0:n-1)*dt;
x = [0;0];%p.x0; %initial condition
v = [1;0];
X = zeros(m,n);

t0 = tic;
tic
for i = 1:n
    
    %drift term
    % non-fractional grad of 2d gaussian wells
    p1 = exp(-0.5*(x(1)-p.location).^2/p.sigma2 - 0.5*x(2).^2/p.sigma2 );
    p2 = exp(-0.5*(x(1)+p.location).^2/p.sigma2 - 0.5*x(2).^2/p.sigma2 );
    
    fx = p1*(-(x(1)-p.location)/p.sigma2 ) + p2*(-(x(1)+p.location)/p.sigma2 );
    fy = -x(2)/p.sigma2;
    f = [fx./(p1+p2); fy] ;
    
    dL = stblrnd(a,0,p.gam,0,[2 1]);
    r = sqrt(sum(dL.*dL)); %step length
    
    %g = [g1 ; g2];
    th = rand*2*pi;
    g = r*[cos(th);sin(th)];
    
    xnew = x + p.gam*f*dt + p.momentum*v*dt + g*dta;
    vnew = v + p.momentum*f*dt;
    
    x = xnew;
    v = vnew;
    
    x = wrapToPi(x); % apply periodic boundary to avoid run-away
    
    %x = wrapToPi(xnew); %periodic boundary condition
    if toc -t0 > 120
        disp('Time out!')
        %return
    end
    
    %endW
    X(:,i) = x;
    
end
end

function show_video(X,t,p)
figure('color','w');
for i = 1:p.skip:length(t)
    
    plot([1 -1]*p.location, [0 0],'*k')
    hold on
    plot(X(1,i),X(2,i),'o')
    hold off
    
    xlabel('x')
    ylabel('y')
    title(['t = ' num2str(t(i))])
    axis([-pi pi -pi pi])
    %axis equal   
    pause(1/60)   
end
end
