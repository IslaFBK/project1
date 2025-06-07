function out = fit_curve_mat(x)

out = x*2;

end



%%

function r = response(param, x)

r = param(1) + param(2)*x; 

end





