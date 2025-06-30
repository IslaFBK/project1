算法*：遗传算法
搜索脚本栅格化初始参数；
第0代：输入初始参数，输出0代解（slope，r2，线性范围，alpha）；
首先判断是否足够线性（r2和线性范围够大）
    不是：认为波包未形成，舍弃该组解
    是：  认为波包形成，判断是否重尾（alpha<2）
        不是：认为是高斯波包，舍弃该组解
        是：  认为是临界态，保留解，优化参数传给下一代
                优化（优化算法索引*）alpha越小越好，保持MSD开头线性
第1代：传递0代参数、解、优化参数、计算1代参数。计算1代解（需重复计算保证alpha可靠）；
首先判断是否足够线性（r2和线性范围够大）
    不是：认为波包未形成，舍弃该组解
    是：  认为波包形成，判断是否重尾（alpha<2）
        不是：认为是高斯波包，舍弃该组解
        是：  认为是临界态，保留解，优化参数传给下一代
                优化（优化算法索引*）alpha越小越好，保持MSD开头线性

聚类算法：
    读取firing_rate，波包存在的条件(坐标都要周期化)，对一段时间而言，存在一刻满足即可：
        圆形范围内firing_rate大于全局firing_rate的 $1-\epsilon$；
        圆形直径小于r倍边长；
        圆形圆心由mass_centre给出
        或者(
            圆形内firing_rate==极大值的点数大于全局极大点数的 $1-\epsilon$；
            圆形直径小于r倍边长；
            圆形圆心由mass_centre给出
        )

# 算上聚类的优化策略：
## 判断模块 (state_evaluator):
加载firing_rate, centre_mass:
```python
load.a1.ge.get_spike_rate()
load.a1.ge.get_centre_mass()
spk_rate = load.a1.ge.spike_rate.spike_rate
centre = load.a1.ge.centre_mass.center
```
用`utils.wave_packet_exist()`判断波包是否存在；
用`utils.is_critical_state()`判断是否足够线性足够重尾；
## 迭代方向计算模块
随机邻域内生成子代，遗传淘汰，子代更优孙代+1，子代更劣孙代-1，（邻域大小随代数指数缩小），is_critical==False则直接淘汰。  
第零代邻域半径为格点间距（1）；

栅格化初始条件0代计算：
```python
params_loop = ...
loop_combinations = ...
results = Parallel(n_jobs=-1)(
    delayed(compute_MSD_pdx)(comb=comb, index=i+1)
    for i, comb in enumerate(loop_combinations)
)
```

第零代判断：
for i in parents:
    if not is_critical:
        
```python
from joblib import Parallel, delayed
import os

def evolve_search(parent_param, eval_func, r0=1.0, k=0.2, max_gen=10, n_child=5, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)  # 留1核给系统
    history = []
    p = np.array(parent_param)
    direction = 1
    for g in range(max_gen):
        r = r0 * np.exp(-k * g)
        children = [p + r * np.random.randn(*p.shape) for _ in range(n_child)]
        # 并行评估所有子代
        scores = Parallel(n_jobs=n_jobs)(
            delayed(eval_func)(child) for child in children
        )
        best_idx = np.argmin(scores)
        if scores[best_idx] < eval_func(p):
            p = children[best_idx]
            direction += 1
        else:
            direction -= 1
        history.append((p.copy(), scores[best_idx], r, direction))
        # 这里可以根据direction调整r或其他策略
    return p, history
```

```python
parents = 初始解列表
for gen in range(max_gen):
    next_parents = []
    for p in parents:
        is_critical, score = eval_func(p)
        if not is_critical:
            continue  # 该解不产生子代，直接淘汰
        # 生成子代
        children = [p + r * np.random.randn(*p.shape) for _ in range(n_child)]
        # 并行评估子代
        child_results = Parallel(n_jobs=n_jobs)(
            delayed(eval_func)(child) for child in children
        )
        # 选出优秀子代，加入下一代
        for child, (is_critical_child, score_child) in zip(children, child_results):
            if is_critical_child:
                next_parents.append(child)
        # 也可保留父代本身
        next_parents.append(p)
    parents = next_parents
```

```python
已汇总会话历史记录你可以这样从 evolution 搜索得到的 `history` 结构中，筛选出所有 `critical==True` 的点，并找到其中 `alpha` 最小的点：

假设 `history` 是一个列表，每个元素是一个 dict，包含至少如下字段：`param`、`critical`、`alpha` 等。你可以用如下代码实现：


# 假设 history 已经存在
critical_points = [h for h in history if h.get('critical', False)]
if critical_points:
    min_alpha_point = min(critical_points, key=lambda h: h['alpha'])
    print("alpha最小的critical点：", min_alpha_point)
else:
    print("没有找到critical==True的点")
```

这样 `min_alpha_point` 就是所有 critical==True 的点中 alpha 最小的那个点（包含其参数、alpha值等信息）。

如需进一步封装为函数或用于后续分析，可以将其写成函数：

```python
def get_min_alpha_critical(history):
    critical_points = [h for h in history if h.get('critical', False)]
    if not critical_points:
        return None
    return min(critical_points, key=lambda h: h['alpha'])
```

你可以直接在 notebook 或 main.py 的最后加上这段代码进行调用和输出。
