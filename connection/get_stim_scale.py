#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:12:37 2020

@author: shni2598

本模块用于生成神经元网络仿真的刺激时序数组。
支持多组刺激强度、随机间隔、灵活拼接，便于批量参数扫描和复杂实验设计。
"""
import numpy as np

#%%
class get_stim_scale:
    """
    刺激时序生成类：
    - 支持多组刺激强度（stim_amp_scale）
    - 支持每组刺激的持续时间（stim_dura）和随机间隔（separate_dura）
    - 生成完整的刺激时序数组（scale_stim）和每组刺激的起止时间（stim_on）
    """
    def __init__(self):
        self.seed = 10  # 随机种子，保证间隔可复现
        self.stim_dura = 200 # 每组刺激持续时间（ms）
        self.separate_dura = np.array([300,500]) # 刺激间隔范围（ms），每组之间随机间隔
        self.dt_stim = 10 # 刺激时序的时间分辨率（ms）
        self.stim_amp_scale = None # 刺激强度数组（每组的幅值）
        # 生成后：
        # self.scale_stim：完整刺激时序数组
        # self.stim_on：每组刺激的起止时间（二维数组）

    def get_scale(self):
        """
        生成刺激时序：
        1. 根据 stim_amp_scale 生成 stim_num 组刺激
        2. 每组持续 stim_dura，组间随机间隔（separate_dura）
        3. scale_stim：完整时序数组，stim_on：每组起止时间
        """
        stim_num = self.stim_amp_scale.shape[0]  # 刺激组数
        stim_dura = self.stim_dura//self.dt_stim # 每组持续步数
        separate_dura = self.separate_dura//self.dt_stim # 间隔步数范围
        np.random.seed(self.seed)  # 固定随机种子
        # 生成每组之间的随机间隔（步数）
        sepa = np.random.rand(stim_num)*(separate_dura[1]-separate_dura[0]) + separate_dura[0]
        sepa = sepa.astype(int)
        # 初始化完整刺激时序数组（长度=所有刺激+所有间隔）
        self.scale_stim = np.zeros([int(round(stim_num*stim_dura+sepa.sum()))])
        self.stim_on = np.zeros([stim_num, 2], int) # 每组刺激的起止时间（单位：步数*dt_stim）
        for i in range(stim_num):
            # 给第i组刺激赋值（幅值为stim_amp_scale[i]，持续stim_dura步）
            self.scale_stim[i*stim_dura + sepa[:i].sum(): i*stim_dura + sepa[:i].sum()+stim_dura] = self.stim_amp_scale[i]
            # 记录第i组刺激的起止时间（单位：ms）
            self.stim_on[i] = np.array([i*stim_dura + sepa[:i].sum(), i*stim_dura + sepa[:i].sum()+stim_dura]) * self.dt_stim
        # 生成后，scale_stim 可直接用于 Brian2 的 TimedArray，stim_on 用于标记刺激区间
        pass