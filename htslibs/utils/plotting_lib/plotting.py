import matplotlib
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# from IPython.core.pylabtools import figsize

from htslibs.utils.file_operation_lib.file_operation import create_folder
# from htslibs.paper.paper3 import plt_settings

class boxplot_draw:
    def __init__(
            self, 
            boxplot_data=None, 
            title='',
            x_label='x', y_label='y', 
            group_colors = 'coolwarm'
    ):
        """
        初始化箱型图绘制类
        :param boxplot_data: 字典，格式为 {标签1: {组1: 数据列表, 组2: 数据列表, ...}, 标签2: {...}, ...}
        :param title: 图形标题
        :param x_label: x轴标签
        :param y_label: y轴标签
        :param group_colors: 组颜色列表或调色板名称
        例如：
            boxplot_data = {
                'Label1': {'A1': [数据], 'A2': [数据], 'A3': [数据]},
                'Label2': {'A1': [数据], 'A2': [数据], 'A3': [数据]},
                ...
            }
        """
        self.boxplot_data = boxplot_data
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.group_colors = group_colors
        self.df = None
        self.ax = None
        self.color_line_median = 'red'  # 中位数线颜色
        self.color_line_whisker = 'gray'  # 须线颜色

        # 设置字体和负号显示（可显示中文）
        # matplotlib.rcParams.update({
        #     'font.sans-serif': ['SimSun'],  # 设置中文字体为宋体
        #     'font.family': 'serif',  # 英文字体使用 serif 族
        #     'font.serif': ['Times New Roman'],  # 指定 serif 用 Times New Roman
        #     'axes.unicode_minus': False,  # 正确显示负号
        #     'mathtext.default': 'it',  # 数学公式斜体
        # })
        # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
        # matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号 '-' 显示为方块的问题
        # plt.rcParams.update({
        #     "text.usetex": False,  # 使用 LaTeX
        #     "mathtext.fontset": "cm",  # 数学字体（可选 'stix', 'cm' 等）
        #     "font.family": "sans-serif",  # 衬线字体（Times New Roman）
        #     "font.serif": ['DejaVu Sans', 'SimHei', 'Microsoft YaHei'],  # 指定 Times 系列
        #     "font.size": 8,  # 全局默认
        # })

    def get_data(self):
        data = {'Label': [], 'Group': [], 'Value': []}
        # 遍历外层字典中的每个标签
        for label, group_dict in self.boxplot_data.items():
            # 遍历内层字典中的每个组
            for group_name, group_data in group_dict.items():
                # 将标签、组和数据添加到 data 字典中
                data['Label'].extend([label] * len(group_data))
                data['Group'].extend([group_name] * len(group_data))
                data['Value'].extend(group_data)
        # 将数据转换为DataFrame
        self.df = pd.DataFrame(data)

    def plot(self, run_mode=1, file_path='', is_showfliers=True, dpi=300):
        """
        绘制箱型图
        :param run_mode: 1 显示图形，2 保存图形
        :param file_path: 保存路径
        :return: None
        """
        # 获取数据
        self.get_data()
        # 创建箱型图
        with plt.rc_context(rc):
            plt.figure(figsize=(3.5, 2.625))
            self.ax = sns.boxplot(
                x='Label', y='Value', hue='Group', data=self.df,
                palette=self.group_colors,    # 使用自定义颜色映射
                boxprops={},
                linewidth=1,  # 设置箱线宽度
                width=0.6,  # 设置箱子宽度
                fliersize=5,
                gap=0.3,  # 设置箱子间距
                showfliers=is_showfliers,  # 不显示异常值
            )
            self.ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')  # 科学计数法
            # self.ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
            # self.ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True) # 科学计数法
            # self.ax.yaxis.get_offset_text().set_position((0.00, 0.9))
            # plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            # 设置箱型图线条的颜色
            self.set_boxplot_colors()
            # 添加x轴分割线
            self.set_x_axis_lines()
            # 设置图形标题和标签
            self.set_title_and_labels(self.title, self.x_label, self.y_label)
            # 调整图例
            self.set_legend()
            # 显示图形
            # plt.ylim(0, 1.2)
            if run_mode == 1:
                plt.show()
            elif run_mode == 2:
                create_folder(file_path)
                os.remove(file_path) if os.path.exists(file_path) else None
                plt.savefig(file_path, dpi=dpi, bbox_inches='tight', transparent=False, pad_inches=0.0)  # dpi 设置分辨率，bbox_inches='tight' 去除多余空白
                plt.close()

    def set_boxplot_colors(self):
        """
        设置箱型图线条的颜色
        """
        # 获取箱体颜色
        box_colors = [patch.get_facecolor() for patch in self.ax.patches]
        # 设置边框颜色与箱体颜色一致
        for i, patch in enumerate(self.ax.patches):
            patch.set_edgecolor(box_colors[i])  # 设置箱体边框颜色
            patch.set_facecolor('none')  # 设置箱体填充颜色为透明
        # 设置须线和中位数线的颜色
        for i, line in enumerate(self.ax.lines):
            box_index = i // 6  # 每组标签有6条线: 2 条须线（上下须线） + 2条须线端点线（上下须线的端点） + 1条中位数线 + 若干条离散点线（数量取决于异常值的数量）
            if i % 6 in [0, 1, 2, 3]:  # 须线和须线端点线
                line.set_color(box_colors[box_index])  # 设置须线颜色
            if i % 6 in [4]:  # 中位数线
                line.set_color(self.color_line_median)  # 设置中位数线颜色
            elif line.get_marker() not in ['None', '']:  # 离散点
                line.set_markeredgecolor(box_colors[box_index])

    def set_x_axis_lines(self):
        """
        添加x轴分割线
        """
        xticks = self.ax.get_xticks()  # 获取 x 轴刻度位置
        # 计算网格线的位置（分隔标签的中间位置）
        for i in range(len(xticks) - 1):
            mid_point = (xticks[i] + xticks[i + 1]) / 2
            plt.axvline(x=mid_point, color='gray', linestyle='-', linewidth=0.6, alpha=0.3)
        plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)  # 添加网格线

    def set_title_and_labels(self, title, x_label, y_label):
        """
        设置图形标题和标签
        """
        plt.title(title)
        plt.xlabel(x_label, labelpad=0)#控制标签与坐标轴的距离（单位为磅）
        plt.ylabel(y_label, labelpad=0)  # r'$I_{perf}$'

    def set_legend(self):
        """
        调整图例
        """
        legend = self.ax.get_legend()
        # 获取图例句柄和标签
        handles, labels = self.ax.get_legend_handles_labels()
        if len(labels) == 1:
            legend.set_visible(False)
        else:
            # # 移除原有图例（因为 `ncol` 参数需要重新创建图例）
            # legend.remove()
            #
            # # 重新创建图例，并设置横排（ncol=标签数量）
            # new_legend = self.ax.legend(
            #     handles=handles,
            #     labels=labels,
            #     loc='best',  # 自动选择最佳位置
            #     ncol=len(labels),  # 关键：横排（列数=标签数）
            #     framealpha=0.5,  # 直接设置图例背景透明度
            # )

            legend.get_frame().set_alpha(0.5)  # 设置图例背景透明度（可选）
            for handle in legend.legend_handles:
                handle.set_edgecolor('none')  # 设置图例颜色块的边框为透明


def boxplot_draw1(boxplot_data=None, title='Boxplot for Different Labels and Groups', x_label='x', y_label='y', group_colors = 'coolwarm'):
    # 设置字体和负号显示（可显示中文）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

    data = {'Label': [],'Group': [],'Value': []}
    # 遍历外层字典中的每个标签
    for label, group_dict in boxplot_data.items():
        # 遍历内层字典中的每个组
        for group_name, group_data in group_dict.items():
            # 将标签、组和数据添加到 data 字典中
            data['Label'].extend([label] * len(group_data))
            data['Group'].extend([group_name] * len(group_data))
            data['Value'].extend(group_data)

    # # 假设数据如下
    # np.random.seed(10)
    #
    # # 模拟数据：五个标签，每个标签对应三组数据
    # num_sample = 100
    # data = {
    #     'Label': ['Label1']*3*num_sample + ['Label2']*3*num_sample + ['Label3']*3*num_sample + ['Label4']*3*num_sample + ['Label5']*3*num_sample,
    #     'Group': ['Group1', 'Group2', 'Group3']*5*num_sample,
    #     'Value': np.random.randn(3*5*num_sample)  # 生成15个随机数
    # }

    # 将数据转换为DataFrame
    df = pd.DataFrame(data)

    # 定义每组的颜色
    # group_colors = ['#87CEEB', '#4682B4', '#191970']  # 每个组的颜色 '#ADD8E6', '#87CEEB', '#4682B4', '#191970'

    # 创建箱型图
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Label', y='Value', hue='Group', data=df,
                     palette=group_colors,    # 使用自定义颜色映射
                     boxprops={},
                     linewidth=1,  # 设置箱线宽度
                     width=0.6,  # 设置箱子宽度
                     fliersize=5,
                     gap=0.3,  # 设置箱子间距
                     # showfliers=False,  # 不显示异常值
                    )

    # 设置箱型图线条的颜色 -----------------------------------------------------------------------------------------------------
    # 获取箱体颜色
    box_colors = [patch.get_facecolor() for patch in ax.patches]
    # 设置边框颜色与箱体颜色一致
    for i, patch in enumerate(ax.patches):
        patch.set_edgecolor(box_colors[i])  # 设置箱体边框颜色
    # 设置须线和中位数线的颜色
    for i, line in enumerate(ax.lines):
        box_index = i // 6  # 每组标签有 6 条线: 2 条须线（上下须线）+2 条须线端点线（上下须线的端点）+1 条中位数线+若干条离散点线（数量取决于异常值的数量）
        if i % 6 in [0, 1, 2, 3]:  # 须线和须线端点线
            line.set_color(box_colors[box_index])  # 设置须线颜色
        if i % 6 in [4]:  # 中位数线
            line.set_color('black')  # 设置中位数线颜色
        # elif i-90 > 0:  # 中位数线
        #     line.set_color('red')  # 设置中位数线颜色为蓝色
        elif line.get_marker() not in ['None', '']:  # 离散点
            # line.set_markerfacecolor(box_colors[box_index])
            line.set_markeredgecolor(box_colors[box_index])

    # 添加x轴分割线 ----------------------------------------------------------------------------------------------------------
    xticks = ax.get_xticks()  # 获取 x 轴刻度位置
    # 计算网格线的位置（分隔标签的中间位置）
    for i in range(len(xticks) - 1):
        mid_point = (xticks[i] + xticks[i + 1]) / 2
        plt.axvline(x=mid_point, color='gray', linestyle='-', linewidth=0.6, alpha=0.7)
    plt.grid(axis='y', linestyle='-', linewidth=0.3, alpha=0.2, color='gray')  # 添加网格线

    # 设置图形标题和标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)  # r'$I_{perf}$'

    # 调整图例
    legend = ax.get_legend()
    # 获取图例句柄和标签
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) == 1:
        legend.set_visible(False)
    else:
        legend.get_frame().set_alpha(0.5)  # 设置图例背景透明度（可选）
        for handle in legend.legend_handles:
            handle.set_edgecolor('none')  # 设置图例颜色块的边框为透明

    # 显示图形
    plt.show()


# —— 字体用局部 rc 上下文，避免全局污染 ——
rc = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei'],  # 设置中文字体（比如 SimHei 黑体）
    'axes.unicode_minus': False,  # 解决负号显示为方块
    'text.usetex': True
    # 'font.family': ['Times New Roman']  # 设置西文字体（比如 Times New Roman）
}

def bo_plot(
        x, y,  # 测试点及函数值
        mean, std,  # 高斯过程预测均值及标准差
        eval_x=None, eval_y=None,  # 评估点
        x_best=None, y_best=None,  # 最佳点
        z=None,  # 诱导点
        label={
            'true': 'true f(x)',
            'pred': 'GP mean',
            'ci': r'95\% CI',
            'train': 'evaluations',
            'inducing': 'inducing points',
            'best': 'best',
            'xlabel': 'x',
            'ylabel': 'f(x)',
        },  # 图例
        color=['r', 'b', 'k'],  # true, predictive, evaluation
        title='title',  # 图形标题
        run_mode=1,  # 0:不运行图像，1:显示图形, 2:不显示图像，保存图像
        file_path="figure.png",  # 保存路径
        ax=None,  # 可传入外部 Axes
):
    """
    绘制贝叶斯优化的图形
    :param x: 输入数据点
    :param y: 真实函数值
    :param mean: 高斯过程的预测均值
    :param std: 高斯过程的预测标准差
    :param eval_x: 评估点的输入数据
    :param eval_y: 评估点的真实函数值
    :param x_best: 最佳输入点（可选）
    :param y_best: 最佳函数值（可选）
    :param title: 图形标题
    :param run_mode: 0:不运行图像，1:显示图形, 2:不显示图像，保存图像, 3:运行图像，不显示，不保存，返回 fig和ax继续操作
    :return: None
    """
    if run_mode == 0:
        return

    # 数据检测
    x = process_array(x, "x")
    y = process_array(y, "y")
    mean = process_array(mean, "mean")
    std = process_array(std, "std")
    eval_x = process_array(eval_x, "eval_x")
    eval_y = process_array(eval_y, "eval_y")
    z = process_array(z, "z") if z is not None else None
    x_best = float(x_best) if x_best is not None else None
    y_best = float(y_best) if y_best is not None else None
    # 数据预处理
    x_sorted_id = np.argsort(x,)  # 对 x 进行排序
    x, y = x[x_sorted_id], y[x_sorted_id]
    mean, std = mean[x_sorted_id], std[x_sorted_id]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    with plt.rc_context(rc):
        # ax.figure(figsize=(7, 4.5))
        ax.plot(x, y, color=color[0], linestyle='-', label=label.get('true', 'true f(x)'))
        line, = ax.plot(x, mean, color=color[1], linestyle=(0, (4, 3)), label=label.get('pred', 'GP mean'))#(0, (10, 6)) # 虚线样式 (10,6)表示画 10 个点，空 6 个点, (0, ...) 是 offset（从 0 开始）
        ax.fill_between(x.ravel(), mean - 1.96 * std, mean + 1.96 * std, color=line.get_color(), alpha=0.4, label=label.get('ci', r'95\% CI'))  # 95% Confidence Interval (95%置信区间)
        if eval_x is not None and eval_y is not None:
            ax.scatter(eval_x, eval_y, c=color[2], s=50, marker='.', label=label.get('train', 'evaluations'), zorder=5)
        if z is not None:
            ax.scatter(z, np.full_like(z, np.min(y)-1), c='g', s=30, marker='.', label=label.get('inducing', 'inducing points'))
        if x_best is not None and y_best is not None:
            ax.scatter([x_best], [y_best], c='g', s=70, marker='*', label=label.get('best', 'best'), zorder=10)
            ax.text(x_best, y_best,
                     f'({x_best:.4f}, {y_best:.4f})',
                     color='k', fontsize=10, ha='right', va='bottom')
        ax.set_title(title)
        ax.set_xlabel(label.get('xlabel', 'x'))
        ax.set_ylabel(label.get('ylabel', 'f(x)'))
        ax.legend()
        # plt_settings.grid_config(ax, grid_on=False)
        # plt_settings.plt_frames(ax)
        ax.grid(linestyle='-', linewidth=0.3, alpha=0.2, color='gray')
        fig.tight_layout()  # 使图形整体看起来更加整齐

    if run_mode == 1:
        plt.show()
    elif run_mode == 2:
        create_folder(file_path)
        os.remove(file_path) if os.path.exists(file_path) else None
        plt.savefig(file_path, dpi=300, bbox_inches='tight')  # dpi 设置分辨率，bbox_inches='tight' 去除多余空白
        plt.close()
    elif run_mode == 3:
        pass
    return fig, ax


def plt_save(plt_handle, file_path):
    """
    保存图形
    :param file_path: 文件路径
    :return: None
    """
    create_folder(file_path)  # 如果不存在，则创建文件夹
    os.remove(file_path) if os.path.exists(file_path) else None  # 如果文件存在，则删除
    plt_handle.savefig(file_path, dpi=300, bbox_inches='tight')  # dpi 设置分辨率，bbox_inches='tight' 去除多余空白
    plt_handle.close()

def process_array(arr, name="var"):
    """
    检查并处理输入数组：
    1. 如果不是 np.ndarray，则转换为 ndarray
    2. 如果是一维数组，保持不变
    3. 如果是二维数组且只有一列，则压平为一维
    4. 其他情况报错
    """
    if arr is None:
        return None

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if arr.ndim != 1:
        if arr.shape[1] == 1:
            arr = arr.ravel()
        else:
            raise ValueError(f"输入数组 {name} 的维度不符合要求，无法进行处理。")
    return arr

if __name__ == "__main__":
    boxplot_data = {
        'Label1': {'A1': np.random.randn(50), 'A2': np.random.randn(50), 'A3': np.random.randn(50)},
        'Label2': {'A1': np.random.randn(50), 'A2': np.random.randn(50), 'A3': np.random.randn(50)},
        'Label3': {'A1': np.random.randn(50), 'A2': np.random.randn(50), 'A3': np.random.randn(50)},
        'Label4': {'A1': np.random.randn(50), 'A2': np.random.randn(50), 'A3': np.random.randn(50)},
        'Label5': {'A1': np.random.randn(50), 'A2': np.random.randn(50), 'A3': np.random.randn(50)},
        'Label6': {'A1': np.random.randn(50), 'A2': np.random.randn(50), 'A3': np.random.randn(50)},
    }

    bp = boxplot_draw(
        boxplot_data=boxplot_data,
        title='不同试验设计方法在G04问题上的对比',
        x_label='采样方法',
        y_label='最大值',
        group_colors=['#87CEEB', '#4682B4', '#191970']
    )  # '#87CEEB', '#4682B4', '#191970'
    bp.plot()
