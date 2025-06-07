import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
import anywidget
from IPython.display import display, HTML
import os
import brian2.numpy_ as np
import itertools

# paths
data_dir = './phasesearch2/raw_data/'
video_dir = './phasesearch2/vedio/'

# appoint looping parameters
params_loop = {
    'num_ee': np.arange(200, 400+1, 200),
    'num_ei': np.arange(200, 400+1, 200),
    'num_ie': np.arange(200, 400+1, 200),
    'num_ii': np.arange(200, 400+1, 200)
}
# generate looping parameter combinations
loop_combinations = itertools.product(*params_loop.values())
# get total looping number
loop_total = 1
for arr in params_loop.values():
    loop_total *= len(arr)

states_path = ''.join(
    f"{k.replace('num_','')}({v[0]}_{v[-1]})"
    for k, v in sorted(params_loop.items())
)
# load states data
with open(
    f"{data_dir}data_{loop_total}_states_{states_path}.file", 'rb'
) as file:
    data = pickle.load(file)

# transition to DataFrame in order to handle
import pandas as pd
df = pd.DataFrame({
    'num_ee': [p['num_ee'] for p in data['params']],
    'num_ei': [p['num_ei'] for p in data['params']],
    'num_ie': [p['num_ie'] for p in data['params']],
    'num_ii': [p['num_ii'] for p in data['params']],
    'alpha_jump': [s['alpha_jump'] for s in data['states']],
    'r2_jump': [s['r2_jump'] for s in data['states']],
    'alpha_spike': [s['alpha_spike'] for s in data['states']],
    'r2_spike': [s['r2_spike'] for s in data['states']]
})

# establish interactive controls
dimension_choices = ['num_ee', 'num_ei', 'num_ie', 'num_ii']
slider_dim = widgets.Dropdown(options=dimension_choices, value='num_ii', description='slider dimension:')
plot_3d = go.FigureWidget()

def update_plot(slider_dim):
    # get slide axis dimension
    fixed_dim_value = slider_dim.value

    # update 3D scatter
    plot_3d.data = []
    plot_3d.add_trace(go.Scatter3d(
        x=df[dimension_choices[0]], # X轴用第一个维度
        y=df[dimension_choices[1]], # Y轴用第二个维度
        z=df[dimension_choices[2]], # Z轴用第三个维度
        mode='markers',             # 显示为散点
        marker=dict(
            size=8,                 # 点的大小
            color=df['alpha_jump'], # 颜色表示α值
            colorscale='Viridis',   # 使用彩虹色系
            opacity=0.8             # 半透明效果
        ),
        # 隐藏的数据：每个点对应的视频文件名
        customdata=df.apply(
            lambda row:
            f"EE{row['num_ee']:03d}_EI{row['num_ei']:03d}_IE{row['num_ie']:03d}_II{row['num_ii']:03d}",
            axis=1),
            # hoverinfo='text',
            # 鼠标悬停时显示的信息
            hovertext=df.apply(
                lambda row:
                f"EE: {row['num_ee']}<br>EI: {row['num_ei']}<br>IE: {row['num_ie']}<br>II: {row['num_ii']}<br>"
                f"alpha: {row['alpha_jump']:.2f}<br>R^2: {row['r2_jump']:.2f}", axis=1
            )
    ))

    # set 3D axis lable
    plot_3d.update_layout(
        scene=dict(
            xaxis_title=dimension_choices[0],
            yaxis_title=dimension_choices[1],
            zaxis_title=dimension_choices[2]
        ),
        title=f"critical phase(fixed axis:{fixed_dim_value})"
    )

def on_click(trace, points, state):
    """ 点击散点时触发"""
    if points.point_inds:  # 如果点击了有效点
        idx = points.point_inds[0]  # 获取点的索引
        video_name = trace.customdata[idx] + ".mp4"  # 拼接视频文件名
        video_path = os.path.join(video_dir, video_name)  # 完整路径

        if os.path.exists(video_path):  # 如果视频存在
            # 在Jupyter中嵌入视频播放器（像弹出DVD播放窗口）
            display(HTML(f"""
                vedio width="640" height="480" controls>
                <source src="{video_path}" type="video/mp4">
                您的浏览器不支持视频标签
                </video>
            """))
        else:
            print(f"video not found:{video_path}")

# 给散点图绑定点击事件（像给按钮接线）
plot_3d.data[0].on_click(on_click)

# 初始绘制（开机第一屏）
update_plot(slider_dim)

# 连接下拉菜单和绘图函数（像连接油门和发动机）
widgets.interactive(update_plot, slider_dim=slider_dim)

# 显示整个控制面板（像启动汽车仪表盘）
display(widgets.VBox([slider_dim, plot_3d]))