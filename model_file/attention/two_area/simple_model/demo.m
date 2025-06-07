%>>>> MODEL PARAMETERS
p.methods.solver = 'Hamiltonian2'; %fractional Hamiltonian MC
p.dt = 1e-3; %integration time step

%p.target = 'unimodal';
p.target = 'bimodal'; %uncomment this to sample from bimodal distribution
p.sigma = 0.7; %width of the gaussian distribution    % 1
p.depth = 1/p.sigma/p.sigma;
%p.depth = 0;
p.location = 3; %peak locations of the bimodal distribution    2.5
T = 30; %simulation time
% a = 1.2; %alpha
%<<<<

for a = 1.2:0.2:2.0
[X,t] = fHMC(T,a,p); %main function

fig = figure;
plot(t,X,'.','markersize',2)
xlabel('t')
ylim([-10,10])
title_str = sprintf('alpha: %.2f \nstim location: -%.1f, %.1f', a, p.location, p.location);
title(title_str)

save_str = sprintf('alpha_%.2f.png', a);
saveas(fig,save_str)
close(fig)
end

