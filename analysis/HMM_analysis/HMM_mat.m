% 
% %R = HMM(mua);
% 
% % mua_cell = {1,20};
% % for t = 1:20
% %     mua_cell{t} = mua(t,:);
% % end
% % 
% % R_cell = HMM(mua);
% %%
function R = HMM_mat(mua, cross_validation)
%% train HMM, infer states, and find explained varaiance for entire MUA data

% disp(size(mua{1}))
[E_train, TR_train, Pi0_train, logliks] = trainHMM(mua);

state_inferred = predictViterbi(mua,TR_train,E_train,Pi0_train);
[var_explained, mean_var, fano] = explianedVariance(mua, state_inferred, E_train);

R.E = E_train;
R.TR = TR_train;
R.Pi0 = Pi0_train;
R.logliks = logliks;
R.state_inferred = state_inferred;
if iscell(mua)
    ntrials = numel(mua);
    for t=1:ntrials
        R.state_inferred{t} = R.state_inferred{t} - 1;
    end
else
    ntrials = size(mua, 1);
    R.state_inferred = R.state_inferred - 1;
end
R.var_explained = var_explained;
R.mean_var = mean_var ;
R.fano = fano;

%ntrials = size(mua, 1);
% state_inferred = zeros(size(mua));
% for trial = 1:ntrials
%     [state_inferred(trial,:), logP] = hmmviterbiPoisson(mua(trial,:),TR_train,E_train,Pi0_train);
% end

%% cross validation
if cross_validation
    half_trials = round(ntrials/2);
    choose_trials = false(1,ntrials);
    choose_trials(randperm(ntrials, half_trials)) = true;
    all_trials = 1:ntrials;
    train_t = all_trials(choose_trials);
    test_t = all_trials(not(choose_trials));

    if iscell(mua)
        train = mua(train_t);
        test = mua(test_t);
    else
        train = mua(train_t,:);
        test = mua(test_t,:);
    end
    [E_train_cv, TR_train_cv, Pi0_train_cv, logliks_cv] = trainHMM(train);
    state_inferred_test = predictViterbi(test,TR_train_cv,E_train_cv,Pi0_train_cv);

    [var_explained_1, mean_var_1, fano_1] = explianedVariance(test, state_inferred_test, E_train_cv);

    [E_train_cv, TR_train_cv, Pi0_train_cv, logliks_cv] = trainHMM(test);
    state_inferred_test = predictViterbi(train,TR_train_cv,E_train_cv,Pi0_train_cv);

    [var_explained_2, mean_var_2, fano_2] = explianedVariance(train, state_inferred_test, E_train_cv);

    var_explained_cv = [var_explained_1;var_explained_2];
    mean_var_cv = [mean_var_1;mean_var_2];
    fano_cv = [fano_1;fano_2];

    R.var_explained_cv = var_explained_cv;
    R.mean_var_cv = mean_var_cv;
    R.fano_cv = fano_cv;
end
end
%%
% figure;
% fano_plt = 0.5:0.1:max(fano);
% max_explained = 1 - 1./fano_plt;
% plot(fano_plt, max_explained)
% hold on
% scatter(fano,var_explained)
% %%
% figure;
% t_num = 2;
% plot(mua(t_num,:))
% hold on
% plot(rate_inferred(t_num,:))
%%
function [var_explained, mean_var, fano] = explianedVariance(mua, state_inferred, E_train)
if iscell(mua)
    ntrials = numel(mua);
    rate_inferred = cell(1,ntrials);
    rate_inferred_tmp = zeros(1,numel(mua{1}));
    for t=1:ntrials
        %rate_inferred_tmp = zeros()
        rate_inferred_tmp(state_inferred{t} == 1) = E_train(1);
        rate_inferred_tmp(state_inferred{t} == 2) = E_train(2);
        rate_inferred{t} = rate_inferred_tmp;
    end
    resi = zeros(ntrials,1); %sum((mua - rate_inferred).^2, 2);
    variance = zeros(ntrials,1); %var(mua,0,2).*(size(mua,2)-1);
    var_explained = zeros(ntrials,1); %1 - resi./variance;
    mean_var = zeros(ntrials,2); %zeros(size(mua,1),2);
    %fano = zeros(ntrials,1);
    %mean_var(:,1) = mean(mua,2);
    %mean_var(:,2) = var(mua,0,2);
    %fano = mean_var(:,2)./mean_var(:,1);
    for t=1:ntrials
        resi(t) = sum((mua{t} - rate_inferred{t}).^2, 2);
        variance(t) = var(mua{t})*(numel(mua{t})-1);
        var_explained(t) = 1 - resi(t)/variance(t);
        mean_var(t,1) = mean(mua{t});
        mean_var(t,2) = var(mua{t});
        
    end
    fano = mean_var(:,2)./mean_var(:,1);
    
else
    rate_inferred = zeros(size(state_inferred));
    rate_inferred(state_inferred == 1) = E_train(1);
    rate_inferred(state_inferred == 2) = E_train(2);

    resi = sum((mua - rate_inferred).^2, 2);
    variance = var(mua,0,2).*(size(mua,2)-1);
    var_explained = 1 - resi./variance;
    mean_var = zeros(size(mua,1),2);
    mean_var(:,1) = mean(mua,2);
    mean_var(:,2) = var(mua,0,2);
    fano = mean_var(:,2)./mean_var(:,1);
end
end









%%
function state_inferred = predictViterbi(mua,TR_train,E_train,Pi0_train)
%ntrials = size(mua, 1);
if iscell(mua)
    ntrials = numel(mua);
    state_inferred = cell(ntrials,1);
    for trial = 1:ntrials
        [state_inferred{trial}, logP] = hmmviterbiPoisson(mua{trial},TR_train,E_train,Pi0_train);
    end   
else
    ntrials = size(mua, 1);
    state_inferred = zeros(size(mua));
    for trial = 1:ntrials
        [state_inferred(trial,:), logP] = hmmviterbiPoisson(mua(trial,:),TR_train,E_train,Pi0_train);
    end
end
end



%%
function [E_train, TR_train, Pi0_train, logliks] = trainHMM(mua)

if iscell(mua)
    n_trials = numel(mua);
    mean_spike_count = 0;
    for t = 1:n_trials
        mean_spike_count = mean_spike_count + mean(mua{t});
    end
    mean_spike_count = mean_spike_count/n_trials;
else
    mean_spike_count = mean(mean(mua(:,1:end)));
end

%[guessTR,guessPi0,guessE] = initialize_param(1,mean_spike_count);

max_iter = 600;
tole = 1e-4;
%discard_tstep = 50;
n_paramTry = 10;
TR_train_try = zeros(2,2,n_paramTry);
E_train_try = zeros(2,n_paramTry);
Pi0_train_try = zeros(2,n_paramTry);
logliks_try = zeros(1,n_paramTry);

for ii=1:n_paramTry
    [guessTR,guessPi0,guessE] = initialize_param(1,mean_spike_count);
    
    [TR_train_try(:,:,ii),E_train_try(:,ii),Pi0_train_try(:,ii),logliks] = ...
        hmmTrainPoisson(mua,guessTR,guessE,guessPi0,'maxiterations',max_iter,'tolerance',tole,...
        'VERBOSE',true);      
%     [TR_train_try(:,:,ii),E_train_try(:,ii),Pi0_train_try(:,ii),logliks] = ...
%         hmmTrainPoisson(mua(:, 1:end),guessTR,guessE,guessPi0,'maxiterations',max_iter,'tolerance',tole,...
%         'VERBOSE',true);
    
    logliks_try(ii) = logliks(end);
end

[~, maxloglik_i] = max(logliks_try);

TR_train = TR_train_try(:,:,maxloglik_i);
E_train = E_train_try(:,maxloglik_i);
Pi0_train = Pi0_train_try(:,maxloglik_i);
logliks = logliks_try(maxloglik_i);

if E_train(1) > E_train(2)
    E_train = E_train([2,1]);   
    TR_train = TR_train([2,1],:);
    TR_train = TR_train(:,[2,1]);
    Pi0_train = Pi0_train([2,1]);
end
end

%%
function [guessTR,guessPi0,guessE] = initialize_param(num_unit,mean_spike_count)
% sampled from a Dirichelet distribution
guessTR = drchrnd([1 1],2);
% sampled from a Dirichelet distribution
guessPi0 = drchrnd([1 1],1);
% initialized from a uniform distribution on [0,2*<n_t^j>], Emission is the
% rate of Poisson process, where j is a neuron
guessE = full(rand(2,num_unit)*2*mean_spike_count);
end

%%
function r = drchrnd(a,n)
% take a sample from a dirichlet distribution
% >> A = drchrnd([1 1 1 1], 3)
%
% A =
%3
% 0.3889 0.1738 0.0866 0.3507
% 0.0130 0.0874 0.6416 0.2579
% 0.0251 0.0105 0.2716 0.6928
%
% >> sum(A, 2)
%
% ans =
%
% 1
% 1
% 1
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);
end