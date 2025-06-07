#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:49:02 2021

@author: shni2598
"""

function out = VmChangePts(x)
%MYCHANGEPTS Find change points in membrane potential time series
%   INPUT: x = membrane potnetial time series

x = x(:);

idx = findchangepts(x,'MinThreshold', 5000,'MinDistance', 500); %min time 500 = 50 ms

% idx: time index of change points

% http://mres.uni-potsdam.de/index.php/2017/04/20/detecting-change-points-in-time-series-with-matlab/

mus = zeros(size(x)); %mean of each segment
t1 = 1;
for i=1:length(idx)
    t2 = idx(i);
    mus(t1:t2-1) = mean(x(t1:t2-1));
    %plot([t1 t2],mean(x(t1:t2))*[1 1],'k','LineWidth',1.5)
    %hold on
    t1 = t2;    
end
mus(t1:end) = mean(x(t1:end));


out.mus = mus; %mean of each segment
out.id = idx; %change points
out.mu = mean(x); %mean of time series

up = false(size(x)); %logic vector for up states
up(mus> -60) = true;

out.up = up;
%return

if all(out.up) || all(~out.up) %if all up or all down
    out.tup = [];
    out.tdown = [];
    return
end


%duration of up state
f = find(diff([false;up;false])~=0);
tup = f(2:2:end)-f(1:2:end);
if up(1); tup(1)=[];end %discard end points if they are up states
if up(end);tup(end)=[];end
out.tup = tup;


%duration of down state
f = find(diff([true;up;true])~=0);
tdown = f(2:2:end)-f(1:2:end);
if ~up(1); tdown(1)=[];end %discard end points if they are down states
if ~up(end);tdown(end)=[];end
out.tdown = tdown;


%to do:
%1. duration of up and down states and their distributions