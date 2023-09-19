import numpy as np 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
from matplotlib import ticker
from matplotlib.patches import Ellipse
from matplotlib import transforms

import seaborn as sns

from .stdio import *

def preproc_conf_mat(conf_mat: np.ndarray) -> np.ndarray :
    conf_mat = conf_mat.astype(np.float32)
    conf_mat = conf_mat/(conf_mat.sum(0) + 1e-6)
    conf_mat = np.round((conf_mat*100)).astype(np.int32)
    return conf_mat


def write_conf_mat(
    conf_mat: np.ndarray, 
    label_to_name: dict, 
    file_path: str,
) :

    conf_mat = preproc_conf_mat(conf_mat)
    fhand = get_file_handle(file_path, 'w+')

    # write 1st row
    fhand.write(' ')
    for i in range(conf_mat.shape[0]) :
        fhand.write(',' + label_to_name[str(i)].ljust(12))
    fhand.write('\n')

    for i in range(conf_mat.shape[0]) :
        fhand.write(label_to_name[str(i)].ljust(12))
        for x in conf_mat[i] :
            fhand.write(',' + str(x) )
        fhand.write('\n')

    fhand.close()


def save_conf_mat_image(
    conf_mat: np.ndarray,
    label_to_name: dict, 
    file_path: str,
    cmap: str = 'hot', # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colorbar: bool = False,
    include_values: bool = True,
    fig_size: float = 6.0,
) :

    """Source: https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/metrics/_plot/confusion_matrix.py
    """

    conf_mat = preproc_conf_mat(conf_mat)

    fig, ax = plt.subplots()
    fig.tight_layout()
    fig.set_size_inches(fig_size, fig_size)

    if conf_mat.size == 1:
        print("Error: Just one class for the first task")
    else:
        n_classes = conf_mat.shape[0]

        label_l = [label_to_name[str(i)].capitalize() for i in range(n_classes)]

        im_kw = dict(interpolation="nearest", cmap=cmap)
        im_ = ax.imshow(conf_mat, **im_kw)

        cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

        if include_values:
            text_ = np.empty_like(conf_mat, dtype=object)

            # print text with appropriate color depending on background
            thresh = (conf_mat.max() + conf_mat.min()) / 2.0

            for i in range(n_classes) :
                for j in range(n_classes) :
                    color = cmap_max if conf_mat[i, j] < thresh else cmap_min
                    text_[i, j] = ax.text(
                        j, i, conf_mat[i, j], ha="center", va="center", color=color
                    )

        if colorbar:
            fig.colorbar(im_, ax=ax)

        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=label_l,
            yticklabels=label_l,
            ylabel="Predictions",
            xlabel="Ground truth",
        )

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation='vertical')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)



def plot_2d(points, colors, title, class_ids=None, class_names=None, figsize=(3, 3), fontsize=None):
    """Source: https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
    """    
    fig, ax = plt.subplots(figsize=figsize, facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, colors, class_ids, class_names, fontsize)
    return fig

def add_2d_scatter(ax, points, colors, class_ids=None, class_names=None, fontsize=None, title=None) :
    """Source: https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
    """    

    x, y = points.T

    if class_ids is None :
        ax.scatter(x, y, c=colors, s=10, alpha=0.8)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())   
        return

    class_ids = np.array(class_ids, dtype=np.int32)
    class_names = np.array(class_names)
    for cid in np.unique(class_ids) :
        mask = class_ids==cid
        label = class_names[mask][0]
        colors_i = colors[mask]
        ax.scatter(x[mask], y[mask], c=colors_i, label=label, s=10, alpha=0.8)
        confidence_ellipse(x[mask], y[mask], ax, n_std=2.0, show_mean=True, color=colors_i[0])

    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())   
    ax.legend(loc='best') if fontsize is None else ax.legend(fontsize=14, loc='best')

def save_figure(fig, fpath) :
    fig.tight_layout()    
    fig.savefig(fpath)


def confidence_ellipse(x, y, ax, n_std=3.0, show_mean=False, color=None, **kwargs) :
    """
    Source: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html 
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      edgecolor=color, facecolor='none', **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    if show_mean :
        ax.plot(mean_x, mean_y, marker="x", c=color)

    return ax.add_patch(ellipse)


def plot_known_unknown_2d(
    known, unknown, w_known=None, w_unknown=None, th=None,
    xlabel=None, ylabel=None, title=None,
) :
    color_l = ['tab:blue', 'tab:orange', 'tab:green']

    fig, ax = plt.subplots()
  
    x = [known, unknown]
    if w_known is None :
        w_known = np.ones_like(known)
    if w_unknown is None :
        w_unknown = np.ones_like(unknown)
    ax = sns.distplot(known, hist_kws={'weights': w_known}, kde=True)
    ax = sns.distplot(unknown, hist_kws={'weights': w_unknown}, kde=True)

    mean_known = np.mean(known)
    mean_unknown = np.mean(unknown)

    if th is not None :
        ax.axvline(x=th, color=color_l[-1], label="threshold")    
        legends = [f'known ({mean_known:.4f})', f'unknown ({mean_unknown:.4f})', f'threshold ({th:.4f})']
    else :
        legends = [f'known ({mean_known:.4f})', f'unknown ({mean_unknown:.4f})']

    
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_title(title, fontsize=22)       
    ax.legend(legends, fontsize=20)
    leg = ax.get_legend()
    for lh, color in zip(leg.legendHandles, color_l) :
        lh.set_color(color)

    # show mean lines
    ax.axvline(x=mean_known, color=color_l[0], ls='--')    
    ax.axvline(x=mean_unknown, color=color_l[1], ls='--')    
        
    fig.tight_layout()
    return fig


def display_sample(pts, target, label2name):

    import matplotlib
    matplotlib.use('TkAgg')   
    import matplotlib.pyplot as plt 

    # First two indeces == origin-destiny. Last index == finger_id
    kepoints_connection = [
                [0, 1,0],
                [0, 2,0],
                [2, 3,0],
                [3, 4,0],
                [4, 5,0],
                [1, 6,1],
                [6, 7,1],
                [7, 8,1],
                [8, 9,1],
                [1, 10,2],
                [10, 11,2],
                [11, 12,2],
                [12, 13,2],
                [1, 14,3],
                [14, 15,3],
                [15, 16,3],
                [16, 17,3],
                [1, 18,4],
                [18, 19,4],
                [19, 20,4],
                [20, 21,4]
                ]
	# for i in human36m_connectivity_dict:
    color_dict ={'0':'b','1':'g','2':'r','3':'c','4':'m'}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-90, azim=90, roll=0)
    #ax.scatter(pts[0][:,0].cpu().numpy(), pts[0][:,1].cpu().numpy(), pts[0][:,2].cpu().numpy())
    RADIUS = 0.15  # space around the subject
    xroot, yroot, zroot = pts[0, 1, 0].cpu().numpy(), pts[0, 1, 1].cpu().numpy(), pts[0, 1, 2].cpu().numpy()

    for sample in range(pts.shape[0]):
        for i in kepoints_connection:
            x = [pts[sample, i[0], 0].cpu().numpy(), pts[sample, i[1], 0].cpu().numpy()]
            y = [pts[sample, i[0], 1].cpu().numpy(), pts[sample, i[1], 1].cpu().numpy()]
            z = [pts[sample, i[0], 2].cpu().numpy(), pts[sample, i[1], 2].cpu().numpy()]
            
            ax.plot(x, y, z, lw=3, c =color_dict[str(i[2])])

        ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
        ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
        ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        #plt.show()
        fig.savefig(osp.join('/ogr_cmu/output/others', f"test_display_sequence_{str(sample)}.png"))
        ax.clear()
    plt.close(fig)