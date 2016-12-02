import matplotlib.pyplot as plt


def grid_axes_labels(f, xlabel=None, ylabel=None, **kws):

    axes = f.axes

    plt.setp(axes.flat, xlabel="", ylabel="")

    if xlabel is not None:
        for ax in axes[-1]:
            ax.set_xlabel(xlabel, **kws)

    if ylabel is not None:
        for ax in axes[0]:
            ax.set_ylabel(ylabel, **kws)
