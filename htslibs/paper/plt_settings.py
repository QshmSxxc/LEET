import os

from htslibs.utils.file_operation_lib.file_operation import create_folder
from matplotlib import pyplot as plt


def plt_subplot(nrows, ncols, figsize=(5.4, 4.5), wspace=0.05, hspace=0.30, top=0.88, bottom=0.22, left=0.05, right=0.995):
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize,
        squeeze=False, sharex='none', sharey='none'
    )
    fig.subplots_adjust(
        left=left, right=right, bottom=bottom, top=top,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, axs

def plt_tick(ax):
    ax.tick_params(direction='out', top=True, right=True, length=2, width=0.4)

def plt_legend(ax, localtion="best"):
    leg = ax.legend(
        # fontsize=4,
        frameon=False,  # 是否显示图例边框
        framealpha=0.8,  # 背景透明度
        loc=localtion,
        handlelength=1.4,  # 线条长度
        labelspacing=0.2,  # 图例项之间的垂直间距
        borderpad=0.3  # 图例内容与边框之间的内边距
    )
    leg.get_frame().set_linewidth(1)  # 图例边框粗细

def plt_fronts():
    rc = {
        "text.usetex": True,  # 让 Matplotlib 使用 LaTeX 来渲染图中的文本。
        "font.family": "serif",  # 设置字体家族为衬线字体（serif），也就是字母边缘有小装饰线条的字体
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",  # 它表示加载两个 LaTeX 宏包：字体风格调成接近 Times / Times New Roman 的效果（和论文配置相同）
        "font.size": 7,       # 全局字体大小
        # "axes.titlesize": 7,  # 图标题
        # "axes.labelsize": 7,  # 坐标轴标签
        # "legend.fontsize": 6, # 图例
        # "xtick.labelsize": 6, # x刻度
        # "ytick.labelsize": 6, # y刻度

        # 'axes.labelpad': 4,
        # 'title.fontsize': 4,
        # 'text.fontsize': 5,
    }
    return rc

# def plt_frames(ax):
#     # 关掉原生 spines （边框）
#     for s in ["top", "right", "bottom", "left"]:
#         ax.spines[s].set_visible(False)
#     # # 设置边框粗细
#     # ax.spines["left"].set_linewidth(0.8)  # 推荐 0.8 ~ 1.2
#     # ax.spines["bottom"].set_linewidth(0.8)
#     # 画箭头轴（使用 axes fraction，和数据范围无关，稳）
#     arrow_kw = dict(arrowstyle="-|>", lw=0.5, color="black", mutation_scale=5)
#
#     ax.annotate("", xy=(1.05, 0), xytext=(-0.02, 0),
#                 xycoords="axes fraction", textcoords="axes fraction",
#                 arrowprops=arrow_kw)  # x轴箭头
#     ax.annotate("", xy=(0, 1.02), xytext=(0, -0.02),
#                 xycoords="axes fraction", textcoords="axes fraction",
#                 arrowprops=arrow_kw)  # y轴箭头

def plt_frames(ax, tick_top=True, tick_right=True, tick_labelleft=True, tick_labelbottom=True):
    # 使刻度朝内
    ax.tick_params(
        axis='both', which='both',  # which: major 主刻度, minor 次刻度, both 都设置
        direction='in', top=tick_top, right=tick_right,
        length=2.0, width=0.5,
        colors='dimgray',  # 刻度颜色（深灰色）
        labelcolor='black',  # 刻度标签颜色
        labelbottom=tick_labelbottom, labelleft=tick_labelleft,  # 显示 x 和 y 轴的刻度标签
    )
    # 设置边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # 常用 0.8 ~ 1.2


def fig_save1(fig, format=['pdf'], file_path='image'):
    """
    保存图片
    :param fig: matplotlib.figure.Figure 对象
    :param format: 保存格式，默认为 'pdf'，也可以是 'png' 'all' 等
    :param file_path: 文件路径，不需要后缀，函数会自动添加 '.pdf' 和 '.png'
    """
    create_folder(file_path+'.pdf')  # 如果不存在，则创建文件夹

    # 保存为 PDF 和 PNG 格式

    file_path_temp = file_path+'.pdf'
    os.remove(file_path_temp) if os.path.exists(file_path_temp) else None  # 如果文件存在，则删除
    fig.savefig(
        file_path_temp,
        format='pdf',
        bbox_inches='tight',  # 自动裁掉多余白边
        transparent=True,  # 背景透明
        pad_inches=0.005,  # 裁剪时不留额外空白
    )
    file_path_temp = file_path + '.png'
    os.remove(file_path_temp) if os.path.exists(file_path_temp) else None  # 如果文件存在，则删除
    fig.savefig(
        file_path_temp,
        dpi=300,
        bbox_inches='tight',  # 自动裁掉多余白边
        transparent=False,  # 背景透明
        pad_inches=0.005,  # 裁剪时不留额外空白
    )

# import os
from pathlib import Path

def fig_save(fig, file_path='image', format='all', pad_inches=0.005):
    """
    保存图片

    :param fig: matplotlib.figure.Figure 对象
    :param format: 'pdf' / 'png' / 'all' / ['pdf', 'png']
    :param file_path: 文件路径，不带后缀
    """

    # 统一 format 输入
    if format == 'all':
        formats = ['pdf', 'png']
    elif isinstance(format, str):
        formats = [format]
    else:
        formats = list(format)

    # 创建目录
    file_path = Path(file_path)
    create_folder(file_path.with_suffix('.pdf'))  # 如果不存在，则创建文件夹

    # 保存
    for fmt in formats:
        file_path_temp = f'{file_path}.{fmt}'

        if os.path.exists(file_path_temp):
            os.remove(file_path_temp)

        if fmt == 'pdf':
            fig.savefig(
                file_path_temp,
                format='pdf',
                bbox_inches='tight',
                transparent=True,
                pad_inches=pad_inches,#0.005
            )

        elif fmt == 'png':
            fig.savefig(
                file_path_temp,
                format='png',
                dpi=300,
                bbox_inches='tight',
                transparent=False,
                pad_inches=pad_inches,
            )

        else:
            raise ValueError(f"Unsupported format: {fmt}")

def grid_config(ax, grid_on=True):
    if grid_on:
        ax.grid(
            True,
            linestyle='-',
            linewidth=0.1,
            color='gray',
            alpha=0.5,
        )
    else:
        ax.grid(False)


def plt_base_settings(ax, grid_f=True, legend_location="best"):
    # 使刻度朝内
    plt_frames(ax)
    # 设置图例
    plt_legend(ax, localtion=legend_location)
    # 设置网格
    grid_config(ax, grid_on=grid_f)

