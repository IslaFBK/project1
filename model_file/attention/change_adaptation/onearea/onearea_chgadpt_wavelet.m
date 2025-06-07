function onearea_chgadpt_wavelet(varargin)
PBS_ARRAYID = varargin{1};
%PBS_ARRAYID=1
filename = ['adpt_rate', num2str(PBS_ARRAYID), '.mat'];
load(filename)
%%
ratepy = data.rate;
%figure
%plot(ratepy)
cwtplot = figure;
cwt(ratepy,1000);
data.param.new_delta_gk
data.param.tau_s_di
data.param.ie_ratio
figname = ['wvt_',num2str(data.param.new_delta_gk),'_', num2str(data.param.tau_s_di),'_', num2str(data.param.ie_ratio),'_', num2str(PBS_ARRAYID),'.png'];
saveas(cwtplot,figname)

end
