import matplotlib.pyplot as plt
import numpy as np
def joint_plot(ratio=1, height=3):
    """
    Makes a plot for two random variables (two fully marginalized pdfs, one joint pdf).
    Taken from Seaborn JointGrid.
    """
    fig = plt.figure(figsize=(height, height))
    gsp = plt.GridSpec(ratio + 1, ratio + 1)

    # uses GridSpec to add subplots
    ax_joint = fig.add_subplot(gsp[1:, :-1])
    ax_marg_x = fig.add_subplot(gsp[0, :-1], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gsp[1:, -1], sharey=ax_joint)

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)

    # Make the grid look nice
    from seaborn import utils
    # utils.despine(fig)
    utils.despine(ax=ax_marg_x, left=True)
    utils.despine(ax=ax_marg_y, bottom=True)
    fig.tight_layout(h_pad=0, w_pad=0)

    # sets axis ticks
    ax_marg_y.tick_params(axis='y', which='major', direction='out')
    ax_marg_x.tick_params(axis='x', which='major', direction='out')
    ax_marg_y.tick_params(axis='y', which='minor', direction='out')
    ax_marg_x.tick_params(axis='x', which='minor', direction='out')
    ax_marg_y.margins(x=0.1, y=0.)

    fig.subplots_adjust(hspace=0, wspace=0)

    return fig, ax_joint, ax_marg_x, ax_marg_y

def offset_xlabel(ax):
    """
    Sets x-axis ticklabels according to a certain style.
    :param ax:
    :return:
    """
    ax.set_xticks([0])
    ax.set_xticklabels(labels=[0], fontdict=dict(color='w'))
    ax.tick_params(axis='x', length=0)
    return ax

def corner_plot(n_plots=3, height=9):
    """
    Creates a square corner plot (of side length height) for n_plots random variables.
    :param n_plots:
    :param height:
    :return:
    """
    # creates the figure
    fig = plt.figure(figsize=(height, height))
    gsp = plt.GridSpec(n_plots, n_plots, wspace=0.05, hspace=0.05)

    #     ax_joint_array_unshaped = [[fig.add_subplot(gsp[n_plots * i + j - 1])
    #                       for j in range(1, i + 1)]
    #                       for i in range(1, n_plots)]
    #     print(ax_joint_array_unshaped)
    #     print(fig.axes)
    #     ax_joint_array_iter = list(itertools.chain.from_iterable(ax_joint_array_unshaped))
    #     print(ax_joint_array_iter)
    #     ax_joint_array = np.reshape(ax_joint_array_iter,
    #                 int(n_plots * (n_plots - 1) / 2))
    #     print(ax_joint_array)
    #     print("ax_joint_array has shape " + str(np.shape(ax_joint_array)))

    # adds and formats subplots
    for i in range(1, n_plots):
        for j in range(1, i + 1):
            if ((n_plots * i + j - 1) % n_plots != 0) and ((n_plots * i + j - 1) < (n_plots * (n_plots - 1))):
                # print(n_plots * i + j - 1)
                fig.add_subplot(gsp[n_plots * i + j - 1],
                                xticklabels=[],
                                yticklabels=[])
            elif (n_plots * i + j - 1) % n_plots != 0:
                # print(n_plots * i + j - 1)
                fig.add_subplot(gsp[n_plots * i + j - 1],
                                yticklabels=[])
            elif (n_plots * i + j - 1) < (n_plots * (n_plots - 1)):
                # print(n_plots * i + j - 1)
                fig.add_subplot(gsp[n_plots * i + j - 1],
                                xticklabels=[])
            else:
                # print(n_plots * i + j - 1)
                # print("I have both sets of labels.")
                fig.add_subplot(gsp[n_plots * i + j - 1])
    # print(fig.axes)

    # reshapes the arrays of joint and fully marginalized pdfs
    ax_joint_array = np.reshape(fig.axes,
                                int(n_plots * (n_plots - 1) / 2))
    # print(ax_joint_array)
    # print("ax_joint_array has shape " + str(np.shape(ax_joint_array)))
    # print(ax_joint_array[-1].get_yticklabels())
    # print(ax_joint_array[-2].get_yticklabels())
    # print(ax_joint_array[-3].get_yticklabels())
    # print("\n")
    # print(ax_joint_array[-1].get_xticklabels())
    # print(ax_joint_array[-2].get_xticklabels())
    # print(ax_joint_array[-3].get_xticklabels())
    ax_marg_array = np.reshape(
        [fig.add_subplot(gsp[i * (n_plots + 1)],
                         yticklabels=[], yticks=[],
                         xticklabels=[])
         for i in range(0, n_plots)],
        (n_plots))

    # print((i * (n_plots + 1) == n_plots**2 - 1))
    # print(fig.axes)
    # print("ax_marg_array has shape " + str(np.shape(ax_marg_array)))

    # creates a blank space with a title
    ax_title = fig.add_subplot(gsp[n_plots - 1])
    ax_title.axis('off')

    return fig, ax_joint_array, ax_marg_array, ax_title

def draw_summary_statistics(bounds68, bounds95, median, height=0, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(bounds68, [height, height], c='gray', lw=6, solid_capstyle='round')
    ax.plot(bounds95, [height, height], c='gray', lw=2, solid_capstyle='round')
    ax.plot([median], [height], c='white', marker='o', zorder=10, markersize=3)