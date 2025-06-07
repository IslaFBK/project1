% fit in HMM fro reproducing on-off state and modulation by attention
clear
%% re-formate spike data
d = dir('*RYG.mat');
num_trials = length(d);
[seqs,timeBins] = deal(cell(num_trials,1));
% sum_spike_count = zeros(num_trials,1);
sti_no = 1;% 1 is the center one; 0 is the corner one
win_size = 100; % unit: 0.1 ms, sync with subroutines!
for ii =1:num_trials
    R = load(d(ii).name,'stamp','spike_hist');
    [spike_temp_E,spike_temp_I] = get_specific_spike_hist(R,sti_no,4e4+1:1e5);
    spike_temp = spike_temp_E;%spike_temp_I];
    bin_temp = movsum(spike_temp,win_size,2,'endpoints','discard');
    seqs{ii} = bin_temp(:,1:win_size:end);
    sum_spike_count(ii,:) = sum(seqs{ii},2);
end
mean_spike_count = sum(sum_spike_count(:))/num_trials/size(seqs{1},1)/size(seqs{1},2);
disp('Load data: done!')
% save('HM.mat','-v7.3')

%% HMM train
% cross validation
clear frac_variance_explained_HMM TR_train E_train Pi0_train
kk = 1;
% frac_variance_explained_HMM = nan*zeros(4,10,2);
% TR_train = nan*zeros(30,10,2,4);
% E_train = nan*zeros(30,10,2,58);
% Pi0_train = nan*zeros(30,10,2,2);
% logliks = nan*zeros(30,10,2,5);
for max_iter = 600%100:100:1000    
    for tole = 1e-4%[1e-4,1e-5,1e-6]
        [Train, Test] = crossvalind('HoldOut', num_trials, 0.5);
        for ii = 1:10 % to avoid local optimal, repeat it for ten random parameter initializations
            [guessTR,guessPi0,guessE] = initialize_param(seqs,mean_spike_count);
            [frac_variance_explained_HMM(kk,ii,1),TR_train(kk,ii,1,:),E_train(kk,ii,1,:),Pi0_train(kk,ii,1,:),logliks(kk,ii,1,:)] = sub_cross_validation_HMM(Train,Test,seqs,sum_spike_count,guessTR,guessE,guessPi0,max_iter,tole);
            % swap test and train and use trained TR E Pi0 as initial value
            [frac_variance_explained_HMM(kk,ii,2),TR_train(kk,ii,2,:),E_train(kk,ii,2,:),Pi0_train(kk,ii,2,:),logliks(kk,ii,2,:)] = sub_cross_validation_HMM(Test,Train,seqs,sum_spike_count,squeeze(TR_train(kk,ii,1,:)),squeeze(E_train(kk,ii,1,:)),squeeze(Pi0_train(kk,ii,1,:)),max_iter,tole);
        end
        kk = kk + 1;
    end
    fprintf('Training max_iteration %d: done!',max_iter)
end
% % average to find the biggest score
% score = mean(mean(frac_variance_explained_HMM,2),3);
% [max_score,ind_score] = min(score)
% parse index
% [max_iter_optimal_ind,tole_optimal_ind] = ind2sub([4,46],ind_score); % [x,y] x is the length of tole; y is the length of max_iter
% get the optimal trained TR, E and Pi0
% temp_log_likes = mean(logliks(ind_score,:,:,:),4);
% max_loglikes = max(temp_log_likes(:));
% max_loglikes_ind = find(temp_log_likes == max_loglikes,1);
% [~,r,c] = ind2sub(size(temp_log_likes),max_loglikes_ind);
% TRANS = TR_train(ind_score,r,c,:);
% TRANS = reshape(TRANS,2,2);
% EMIS = E_train(ind_score,r,c,:);
% EMIS = reshape(EMIS,2,[]);
% Pi0 = Pi0_train(ind_score,r,c,:);
% Pi0 = reshape(Pi0,1,2);

% get the optimal trained TR, E and Pi0
temp_log_likes = mean(logliks,4);
max_loglikes = max(temp_log_likes(:));
max_loglikes_ind = find(temp_log_likes == max_loglikes,1);
[w,r,c] = ind2sub(size(temp_log_likes),max_loglikes_ind);
TRANS = TR_train(w,r,c,:);
TRANS = reshape(TRANS,2,2);
EMIS = E_train(w,r,c,:);
EMIS = reshape(EMIS,2,[]);
Pi0 = Pi0_train(w,r,c,:);
Pi0 = reshape(Pi0,1,2);
%% label state 1 2
label_state = cell(num_trials,1);
label_flag = false(num_trials,length(EMIS));
[on_off_corr,off_on_corr] = deal(nan*zeros(num_trials,length(EMIS)));
for ii = 1:num_trials
    for jj = 1:length(EMIS)
        [label_state{ii}(jj,:), ~] = hmmviterbiPoisson(seqs{ii}(jj,:),TRANS,EMIS(:,jj),Pi0);
        % determin on-off state by different mean firing rate
        state_1_rate = mean(EMIS(1,:));% state 1's
        state_2_rate = mean(EMIS(2,:));% state 2's
        if state_1_rate > state_2_rate
            label_flag(ii,jj) = 1;% 1 means state 1 is on state
        elseif state_1_rate < state_2_rate
            label_flag(ii,jj) = 0;% 0 means state 1 is off state
        end
        % check Markovian assumption by calculating the correlation between the duration of consecutive state x and state y episode (x y are on or off)
        % 1 duration
        S1 = bwconncomp(label_state{ii}(jj,:) < 2);
        temp_dur_S1 = cellfun(@length,S1.PixelIdxList);
        % 2 duration
        S2 = bwconncomp(label_state{ii}(jj,:) > 1);
        temp_dur_S2 = cellfun(@length,S2.PixelIdxList);
        % get the max index of state
        T_max = min(length(temp_dur_S1),length(temp_dur_S2));
        % determin the first state
        if label_state{ii}(jj,1) == 1 % the first state is S1
            if label_flag(ii,jj)% 1 means state 1 is on state
                % on-off correlation
                on_off_corr(ii,jj) = corr2(temp_dur_S1(1:T_max),temp_dur_S2(1:T_max));
                % off-on correlation
                off_on_corr(ii,jj) = corr2(temp_dur_S1(2:T_max),temp_dur_S2(1:T_max-1));
            else %means state 1 is off state
                % on-off correlation
                on_off_corr(ii,jj) = corr2(temp_dur_S1(2:T_max),temp_dur_S2(1:T_max-1));
                % off-on correlation
                off_on_corr(ii,jj) = corr2(temp_dur_S1(1:T_max),temp_dur_S2(1:T_max));
            end
        elseif label_state{ii}(jj,1) == 2 % the first state is S2
            if label_flag(ii,jj)% 1 means state 1 is on state
                % on-off correlation
                on_off_corr(ii,jj) = corr2(temp_dur_S1(2:T_max),temp_dur_S2(1:T_max-1));
                % off-on correlation
                off_on_corr(ii,jj) = corr2(temp_dur_S1(1:T_max),temp_dur_S2(1:T_max));
            else %means state 1 is off state
                % on-off correlation
                on_off_corr(ii,jj) = corr2(temp_dur_S1(1:T_max),temp_dur_S2(1:T_max));
                % off-on correlation
                off_on_corr(ii,jj) = corr2(temp_dur_S1(2:T_max),temp_dur_S2(1:T_max-1));
            end
        end
    end
end
mean_on_off_corr = nanmean(on_off_corr(:));
p_on_off = signrank(on_off_corr(:));
mean_off_on_corr = nanmean(off_on_corr(:));
p_off_on = signrank(off_on_corr(:));
disp('Label and check assumption of Markov process: done!')
%% use surrogate data calculate spike count correlation
for ii = 1:num_trials
    timeBins{ii} = zeros(size(seqs{1}));
end
[states, seq]= hmmGeneratePoisson(TRANS,EMIS,Pi0, timeBins);
corr_pool = nan*zeros(num_trials,1);
for ii = 1:num_trials
    % calculate correlation
    raw_corr = corr(seq{ii}');
    if ii == 1
        triu_raw_corr_indx = logical(triu(ones(size(raw_corr)),1));
    end
    corr_pool(ii) = mean(raw_corr(triu_raw_corr_indx),'omitnan');
end
figure
h=histogram(corr_pool,'normalization','probability');
saveas(gcf,'corr_pool.jpg')
r_sc_theoretical = nanmean(corr_pool);
saveas(gcf,'corr_on_off.jpg')
disp('Get therotical prediction of noise correlation: done!')

save('HMM_analysis.mat','r_sc_theoretical','r_sc_theoretical','mean_on_off_corr','p_on_off','mean_off_on_corr','p_off_on',...
    'TRANS','EMIS','Pi0','w','r','c','temp_log_likes','frac_variance_explained_HMM','label_state','label_flag')
%% get the attentional modulation. Based on the paper(Engel, Science, 2016), when the animals attended to a stimulus, the vigorous spiking states became longer and the faint spiking states became shorter
% load('HMM_analysis_1000.mat')
[on_duration_cen,off_duration_cen,on_duration_cor,off_duration_cor,on_rate_cen,...
    off_rate_cen,on_rate_cor,off_rate_cor,on_duration_cen_total,off_duration_cen_total,...
    on_duration_cor_total,off_duration_cor_total] = get_attentional_modulation(label_state,label_flag,seqs);
% on_duration_cen means when attention attends central area, the duration
% of on state in the stimulus area I am calculating, i.e., sti_no
save('on_off_mod.mat','on_duration_cen','off_duration_cen','on_duration_cor','off_duration_cor',...
    'on_rate_cen','off_rate_cen','on_rate_cor','off_rate_cor','on_duration_cen_total','off_duration_cen_total',...
    'on_duration_cor_total','off_duration_cor_total','-v7.3')

disp('Attentional modulation: done!')
%% subrountine
function [S_E,S_I] = get_specific_spike_hist(R,sti_no,time_duration)
% sti_no: 1 is the center one; 0 is the corner one
if sti_no
    spatialfilter = get_spatialfilter_from_coordinate(32,32,3);
else
    spatialfilter = get_spatialfilter_from_coordinate(0,0,3);
end
temp{1}=R;
[choosedNeuEInd,choosedNeuIInd] = get_neuron_index_from_position(temp,spatialfilter);
S_temp_E = R.spike_hist{1};
S_temp_I = R.spike_hist{2};
S_E = S_temp_E(choosedNeuEInd{1},time_duration);
S_I = S_temp_I(choosedNeuIInd{1},time_duration);
end

function [guessTR,guessPi0,guessE] = initialize_param(seqs,mean_spike_count)
% sampled from a Dirichelet distribution
guessTR = drchrnd([1 1],2);
% sampled from a Dirichelet distribution
guessPi0 = drchrnd([1 1],1);
% initialized from a uniform distribution on [0,2*<n_t^j>], Emission is the
% rate of Poisson process, where j is a neuron
guessE = full(rand(2,size(seqs{1},1))*2*mean_spike_count);
end

function [frac_variance_explained_HMM,TR_train,E_train,Pi0_train,logP] = sub_cross_validation_HMM(Train,Test,seqs,sum_spike_count,guessTR,guessE,guessPi0,max_iter,tole)
[numStates, checkTr] = size(guessTR);
if checkTr ~= numStates % prerequest is the guessTR has enough entries
    guessTR = reshape(guessTR,sqrt(length(guessTR)),[]);
end

[checkE, numEmissions] = size(guessE);
if checkE ~= numStates
    guessE = reshape(guessE,2,[]);
end

seqs_train = seqs(Train);
seqs_test = seqs(Test);
sum_spike_count_test = sum_spike_count(repmat(Test,1,size(sum_spike_count,2)));
sum_spike_count_test = reshape(sum_spike_count_test,[],size(sum_spike_count,2));
[TR_train,E_train,Pi0_train,logliks] = hmmTrainPoisson(seqs_train,guessTR,guessE,guessPi0,'maxiterations',max_iter,'tolerance',tole);
if sum(sum(TR_train<1e-5))>0 || sum(sum(E_train<1e-5))>0 %|| sum(Pi0_train<1e-5)>0
    error('!')
end
E_train_state_1 = E_train(1,:);
E_train_state_2 = E_train(2,:);
for ii = 1:length(seqs_test)
    fit_rate = nan*ones(size(seqs{1}));
    [currentState, logP(ii)] = hmmviterbiPoisson(seqs_test{ii},TR_train,E_train,Pi0_train);
    variance_data = sum(seqs_test{ii} - ((sum_spike_count_test(ii,:)/(size(seqs{1},2))).^2)',2);
    for jj = 1:size(seqs{1},1)
        temp = nan*zeros(1,size(seqs{1},2));
        temp(currentState < 2) = E_train_state_1(jj);% state 1's
        temp(currentState > 1) = E_train_state_2(jj);% state 2's
        fit_rate(jj,:) = temp;
    end
    variance_HMM_fit = sum((seqs_test{ii} - fit_rate).^2,2);
    frac_variance_explained_HMM_temp = 1 - variance_data./variance_HMM_fit;
    frac_variance_explained_HMM(ii) = mean(frac_variance_explained_HMM_temp(:));
end
frac_variance_explained_HMM = mean(frac_variance_explained_HMM);
TR_train = TR_train(:);
E_train = E_train(:);
Pi0_train = Pi0_train(:);
end

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

function [on_duration_cen,off_duration_cen,on_duration_cor,off_duration_cor,on_rate_cen,off_rate_cen,on_rate_cor,off_rate_cor,on_duration_cen_total,off_duration_cen_total,on_duration_cor_total,off_duration_cor_total] = get_attentional_modulation(label_state,label_flag,seqs)
% whether cen or cor is att depends on the input data
d = dir('*RYG.mat');
sti_no = 1;% 1 is the center one; 0 is the corner one
win_size = 100; % unit: 0.1 ms
num_trials = min(length(d),size(label_flag,1));
[on_duration_cen,off_duration_cen,on_duration_cor,off_duration_cor] = deal(cell(1,num_trials));
[on_rate_cen,off_rate_cen,on_rate_cor,off_rate_cor,on_duration_cen_total,off_duration_cen_total,on_duration_cor_total,off_duration_cor_total] = deal(zeros(1,num_trials));
for ii = 1:num_trials
    R = load(d(ii).name);
    if nargin < 3
        [spike_temp_E,spike_temp_I] = get_specific_spike_hist(R,sti_no,4e4+1:1e5);
        spike_temp = spike_temp_E;%spike_temp_I];
        bin_temp = movsum(spike_temp,win_size,2,'endpoints','discard');
        seq = bin_temp(:,1:win_size:end);
    else
        seq = seqs{ii};
    end
    [duration_centre_att,duration_corner_att] = get_duration(R,win_size);
    % HMM used 10 ms as a bin. So, discretize and logicalize the duration for comparison
    [dur_cen_att,dur_cor_att] = deal(false(size(4e3:1e4)));
    [on_duration_cen{ii},on_duration_cor{ii},off_duration_cen{ii},off_duration_cor{ii}] = deal(false(size(label_state{ii})));
    dur_cen_att(duration_centre_att) = 1;
    dur_cor_att(duration_corner_att) = 1;
    dur_cen_att = movmax(dur_cen_att,win_size/10,'Endpoints','discard');
    dur_cen_att = dur_cen_att(1:win_size/10:end);
    dur_cor_att = movmax(dur_cor_att,win_size/10,'Endpoints','discard');
    dur_cor_att = dur_cor_att(1:win_size/10:end);
    for kk = 1:size(label_state{ii},1)
        % 1 duration
        S1 = bwconncomp(label_state{ii}(kk,:) < 2);
        % 2 duration
        S2 = bwconncomp(label_state{ii}(kk,:) > 1);
        
        if label_flag(ii)% 1 means state 1 is on state
            % on duration
            for jj = 1:length(S1.PixelIdxList)
                temp = false(size(dur_cen_att));
                temp(S1.PixelIdxList{jj}) = 1;
                temp1 = dur_cen_att & temp;
                on_duration_cen{ii}(kk,:) = on_duration_cen{ii}(kk,:) | temp1;
                temp2 = dur_cor_att & temp;
                on_duration_cor{ii}(kk,:) = on_duration_cor{ii}(kk,:) | temp2;
            end
            % off duration
            for jj = 1:length(S2.PixelIdxList)
                temp = false(size(dur_cen_att));
                temp(S2.PixelIdxList{jj}) = 1;
                temp3 = dur_cen_att & temp;
                off_duration_cen{ii}(kk,:) = off_duration_cen{ii}(kk,:) | temp3;
                temp4 = dur_cor_att & temp;
                off_duration_cor{ii}(kk,:) = off_duration_cor{ii}(kk,:) | temp4;
            end
        else %means state 1 is off state
            % on duration
            for jj = 1:length(S2.PixelIdxList)
                temp = false(size(dur_cen_att));
                temp(S2.PixelIdxList{jj}) = 1;
                temp1 = dur_cen_att & temp;
                on_duration_cen{ii}(kk,:) = on_duration_cen{ii}(kk,:) | temp1;
                temp2 = dur_cor_att & temp;
                on_duration_cor{ii}(kk,:) = on_duration_cor{ii}(kk,:) | temp2;
            end
            % off duration
            for jj = 1:length(S1.PixelIdxList)
                temp = false(size(dur_cen_att));
                temp(S1.PixelIdxList{jj}) = 1;
                temp3 = dur_cen_att & temp;
                off_duration_cen{ii}(kk,:) = off_duration_cen{ii}(kk,:) | temp3;
                temp4 = dur_cor_att & temp;
                off_duration_cor{ii}(kk,:) = off_duration_cor{ii}(kk,:) | temp4;
            end
        end
    end
    % total duratoin
    on_duration_cen_total(ii) = sum(on_duration_cen{ii}(:))/size(on_duration_cen{ii},1);% average over #neurons
    on_duration_cor_total(ii) = sum(on_duration_cor{ii}(:))/size(on_duration_cen{ii},1);
    off_duration_cen_total(ii) = sum(off_duration_cen{ii}(:))/size(on_duration_cen{ii},1);
    off_duration_cor_total(ii) = sum(off_duration_cor{ii}(:))/size(on_duration_cen{ii},1);
    % rate duration
    on_rate_cen(ii) = sum(seq(on_duration_cen{ii}(:)))/on_duration_cen_total(ii)/size(on_duration_cen{ii},1);
    off_rate_cen(ii) = sum(seq(off_duration_cen{ii}(:)))/off_duration_cen_total(ii)/size(on_duration_cen{ii},1);
    on_rate_cor(ii) = sum(seq(on_duration_cor{ii}(:)))/on_duration_cor_total(ii)/size(on_duration_cen{ii},1);
    off_rate_cor(ii) = sum(seq(off_duration_cor{ii}(:)))/off_duration_cor_total(ii)/size(on_duration_cen{ii},1);
end
end


function [duration_centre_att,duration_corner_att] = get_duration(R,varargin)
% detect the attentional sampling rate and duration

hw = 31;
end_step = 10000;
sampling_interval = 1;%ms
onset = 4000;

% proporty can have hw,onset,end_step,sampling_interval
% etc
for i = 1:length(varargin)/2
    var_name = varargin{2*i-1};
    var_value = varargin{2*i};
    if isnumeric(var_value)
        eval([var_name, '=', num2str(var_value), ';']);
    else
        eval([var_name, '=''', var_value, ''';']);
    end
end

Fix = checkFixations_GC(R,'low_percent',0.97,'end_step',end_step,'FixDurCut',25,'onset',onset,'sampling_interval',sampling_interval);

dis_centre = (sum((wrapToPi(Fix.meanXY_pi - [0,0])).^2,2)).^0.5;
dis_corner = (sum((wrapToPi(Fix.meanXY_pi - [pi,pi])).^2,2)).^0.5;
attend_centre_flag = dis_centre < 0.5*pi;
attend_corner_flag = dis_corner < 0.5*pi;

duration_centre_att = [];
for ii = 1:length(attend_centre_flag)
    if attend_centre_flag(ii)
        duration_centre_att = [duration_centre_att, Fix.start(ii):Fix.end(ii)];
    end
end
duration_corner_att = [];
for ii = 1:length(attend_corner_flag)
    if attend_corner_flag(ii)
        duration_corner_att = [duration_corner_att, Fix.start(ii):Fix.end(ii)];
    end
end
end