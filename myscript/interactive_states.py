import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import re
from pathlib import Path
import pickle
import itertools

def create_advanced_interactive_plot(data_states, dim_ext, para_range, 
                                   graph_dir="parallel/graph/",
                                   video_dir="parallel/video/",
                                   output_file="parallel/advanced_phase_diagram.html"):
    """
    创建高级交互式4D相图可视化系统
    
    参数:
        data_states: 包含参数和状态的数据字典
        dim_ext: 作为滑块控制的维度 (ee/ei/ie/ii)
        para_range: 各维度范围字典 {'num_ee': [min,max], ...}
        graph_dir: 图形存储目录
        video_dir: 视频存储目录
        output_file: 输出的HTML文件名
    """
    # 准备路径
    coactivity_dir = f"{graph_dir}/coactivity/"
    jump_dir = f"{graph_dir}/jump/"

    # 准备数据
    params_list = data_states['params']
    states_list = data_states['states']
    
    param_dims = ['ee', 'ei', 'ie', 'ii']
    if dim_ext not in param_dims:
        raise ValueError(f"dim_ext must be one of {param_dims}")
    
    fixed_dims = [dim for dim in param_dims if dim != dim_ext]
    
    # 创建DataFrame并添加所有路径信息
    data = []
    for i in range(min(len(params_list), len(states_list))):
        if states_list[i] is not None:
            row = {**params_list[i], **states_list[i]}
            # 生成common_path
            common_path = f"EE{row['num_ee']:03d}_EI{row['num_ei']:03d}_IE{row['num_ie']:03d}_II{row['num_ii']:03d}"
            row.update({
                'coactivity_path': f"{coactivity_dir}/coactivity_{common_path}.png",
                'jump_path': f"{jump_dir}/jump_{common_path}.png",
                'video_path': f"{video_dir}/{common_path}_pattern.mp4",
                'combined_path': f"{graph_dir}/combined_{common_path}.png",
                'common_path': common_path
            })
            # 检查文件是否存在
            row['coactivity_exists'] = os.path.exists(row['coactivity_path'])
            row['jump_exists'] = os.path.exists(row['jump_path'])
            row['video_exists'] = os.path.exists(row['video_path'])
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # 创建完整的参数值范围（包括没有数据点的值）
    dim_values = {
        dim: np.arange(para_range[f'num_{dim}'][0], 
                      para_range[f'num_{dim}'][1]+1, 
                      (para_range[f'num_{dim}'][1]-para_range[f'num_{dim}'][0])//10 or 1)
        for dim in param_dims
    }
    
    # 创建图形 (4个子图，每个对应一个状态变量)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=list(states_list[0].keys()),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 为每个状态变量添加初始空轨迹
    for i, state_key in enumerate(states_list[0].keys()):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(size=6, opacity=0.8),
                name=state_key,
                customdata=np.empty((0, 7))  # 存储额外数据
            ),
            row=row, col=col
        )
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text=f"4D参数空间探索 - 变化维度: {dim_ext}",
            x=0.5,
            xanchor="center"
        ),
        margin=dict(t=100, b=100),
        height=1200,
        sliders=[dict(
            active=0,
            currentvalue={"prefix": f"{dim_ext}="},
            steps=[
                dict(
                    method="update",
                    args=[
                        {"visible": [False]*len(fig.data)},  # 先隐藏所有
                        {"title": f"当前{dim_ext}={val}"},
                        # 更新每个子图的数据
                        *[{
                            f"xaxis{i+1}": [df[df[f'num_{dim_ext}']==val][f'num_{fixed_dims[0]}'].values],
                            f"yaxis{i+1}": [df[df[f'num_{dim_ext}']==val][f'num_{fixed_dims[1]}'].values],
                            f"zaxis{i+1}": [df[df[f'num_{dim_ext}']==val][f'num_{fixed_dims[2]}'].values],
                            "marker.color": [df[df[f'num_{dim_ext}']==val][state_key].values],
                            "customdata": df[df[f'num_{dim_ext}']==val][
                                ['coactivity_path', 'jump_path', 'video_path', 
                                 'coactivity_exists', 'jump_exists', 'video_exists',
                                 'common_path']].values
                        } for i, state_key in enumerate(states_list[0].keys())]
                    ],
                    label=str(val)
                ) for val in dim_values[dim_ext]
            ]
        )],
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="播放动画",
                method="animate",
                args=[None, {"frame": {"duration": 500}}]
            )]
        )]
    )
    
    # 设置子图轴范围和标签
    for i in range(4):
        fig.update_scenes(
            xaxis_title_text=fixed_dims[0],
            yaxis_title_text=fixed_dims[1],
            zaxis_title_text=fixed_dims[2],
            xaxis_range=para_range[f'num_{fixed_dims[0]}'],
            yaxis_range=para_range[f'num_{fixed_dims[1]}'],
            zaxis_range=para_range[f'num_{fixed_dims[2]}'],
            row=(i//2)+1, col=(i%2)+1
        )
    
    # 添加自定义JavaScript实现高级交互
    custom_js = """
    <script>
    // 创建媒体展示区域
    const mediaContainer = document.createElement('div');
    mediaContainer.style.position = 'fixed';
    mediaContainer.style.bottom = '0';
    mediaContainer.style.left = '0';
    mediaContainer.style.width = '100%';
    mediaContainer.style.height = '300px';
    mediaContainer.style.backgroundColor = 'white';
    mediaContainer.style.borderTop = '1px solid #ddd';
    mediaContainer.style.zIndex = '1000';
    mediaContainer.style.overflow = 'auto';
    mediaContainer.style.padding = '10px';
    mediaContainer.style.display = 'flex';
    mediaContainer.style.flexDirection = 'column';
    mediaContainer.style.alignItems = 'center';
    
    document.body.appendChild(mediaContainer);
    
    // 创建图片合并函数
    function combineAndShowImages(coactivityPath, jumpPath, commonPath) {
        // 创建容器
        const imgContainer = document.createElement('div');
        imgContainer.style.display = 'flex';
        imgContainer.style.justifyContent = 'center';
        imgContainer.style.gap = '20px';
        imgContainer.style.marginBottom = '10px';
        
        // 创建标题
        const title = document.createElement('h3');
        title.textContent = `模式详情: ${commonPath}`;
        title.style.textAlign = 'center';
        
        // 创建图片元素
        const img1 = document.createElement('img');
        img1.src = coactivityPath;
        img1.style.maxHeight = '250px';
        img1.style.border = '1px solid #ddd';
        
        const img2 = document.createElement('img');
        img2.src = jumpPath;
        img2.style.maxHeight = '250px';
        img2.style.border = '1px solid #ddd';
        
        // 组装元素
        imgContainer.appendChild(img1);
        imgContainer.appendChild(img2);
        
        // 清空容器并添加新内容
        mediaContainer.innerHTML = '';
        mediaContainer.appendChild(title);
        mediaContainer.appendChild(imgContainer);
        
        // 返回合并后的图片容器（用于后续添加视频）
        return imgContainer;
    }
    
    // 创建视频播放器
    function createVideoPlayer(videoPath) {
        const videoContainer = document.createElement('div');
        videoContainer.style.textAlign = 'center';
        videoContainer.style.marginTop = '10px';
        
        const video = document.createElement('video');
        video.src = videoPath;
        video.controls = true;
        video.style.maxHeight = '200px';
        video.style.maxWidth = '80%';
        
        videoContainer.appendChild(video);
        return videoContainer;
    }
    
    // 点击事件处理
    function handlePointClick(point) {
        const [coactivityPath, jumpPath, videoPath, 
               coactivityExists, jumpExists, videoExists,
               commonPath] = point.customdata;
        
        // 清空媒体容器
        mediaContainer.innerHTML = '';
        
        // 检查文件是否存在
        if (!coactivityExists || !jumpExists) {
            mediaContainer.innerHTML = '<p style="color:red">部分图形文件缺失</p>';
            return;
        }
        
        // 显示合并后的图片
        const imgContainer = combineAndShowImages(coactivityPath, jumpPath, commonPath);
        
        // 如果视频存在则添加
        if (videoExists) {
            const videoContainer = createVideoPlayer(videoPath);
            mediaContainer.appendChild(videoContainer);
        } else {
            const warning = document.createElement('p');
            warning.textContent = '视频文件未找到';
            warning.style.color = 'red';
            mediaContainer.appendChild(warning);
        }
    }
    
    // 绑定点击事件
    document.querySelector('.plotly-graph-div').on('plotly_click', function(data) {
        if (data.points && data.points[0]) {
            handlePointClick(data.points[0]);
        }
    });
    </script>
    """
    
    # 保存为HTML文件
    fig.write_html(
        output_file,
        include_plotlyjs=True,
        full_html=True,
        post_script=custom_js
    )
    
    print(f"高级交互式图表已保存到 {output_file}")
    print(f"请用浏览器打开，点击数据点可查看对应的图形和视频")


# 使用示例
# 使用示例
# load phase data and rebuild Analyzer object
root_dir = 'parallel'
data_dir = 'parallel/raw_data/'
graph_dir = 'parallel/graph/'
vedio_dir = 'parallel/vedio/'
state_dir = 'parallel/state/'

# automatically identify looping parameters
path_list = os.listdir(data_dir)
# filter out non-data files
params_list = []
pattern = r'EE(\d+)_EI(\d+)_IE(\d+)_II(\d+)'
# extract parameters from file names
for path in path_list:
    match = re.search(pattern, path)
    if match:
        params = {
            'num_ee': int(match.group(1)),
            'num_ei': int(match.group(2)),
            'num_ie': int(match.group(3)),
            'num_ii': int(match.group(4))
        }
        params_list.append(params)

# generate looping parameter combinations
loop_combinations = [
    (np.int64(p['num_ee']), np.int64(p['num_ei']), np.int64(p['num_ie']), np.int64(p['num_ii']))
    for p in params_list
]
# get total looping number
loop_total = len(loop_combinations)

''' load phase data '''
with open(f"{state_dir}auto_states.file", 'rb') as file:
    data_states = pickle.load(file)

# create_interactive_plot(data_states, 'ii', {'num_ee': [100, 400], ...}, "my_plot.html")