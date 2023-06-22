import matplotlib.pyplot as plt
import numpy as np
import gsum as gm
from matplotlib.patches import Patch
import matplotlib as mpl
from .utils import mean_and_stddev, sig_figs, correlation_coefficient

# See: https://ianstormtaylor.com/design-tip-never-use-black/
# softblack = '#262626'
softblack = "k" # Looks better when printed on tex file
gray = "0.7"
edgewidth = 0.6
text_bbox = dict(boxstyle="round", fc=(1, 1, 1, 0.6), ec=softblack, lw=0.8)

def setup_rc_params():
    mpl.rcParams["figure.dpi"] = 180
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"

    mpl.rcParams["axes.labelsize"] = 9  # 14
    mpl.rcParams["axes.edgecolor"] = softblack
    mpl.rcParams["axes.xmargin"] = 0
    mpl.rcParams["axes.labelcolor"] = softblack
    mpl.rcParams["axes.linewidth"]

    mpl.rcParams['lines.markersize'] = 6

    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.labelsize"] = 9  # 12
    mpl.rcParams["ytick.labelsize"] = 9  # 12
    mpl.rcParams["xtick.color"] = softblack
    mpl.rcParams["ytick.color"] = softblack
    mpl.rcParams["xtick.minor.size"] = 2.4
    mpl.rcParams["ytick.minor.size"] = 2.4

    mpl.rcParams["legend.title_fontsize"] = 9
    mpl.rcParams["legend.fontsize"] = 9  # 11
    mpl.rcParams["legend.edgecolor"] = "inherit"  # inherits from axes.edgecolor, to match
    mpl.rcParams["legend.facecolor"] = (
        1,
        1,
        1,
        0.6,
    )  # Set facecolor with its own alpha, so edgecolor is unaffected
    mpl.rcParams["legend.fancybox"] = True
    mpl.rcParams["legend.borderaxespad"] = 0.8
    mpl.rcParams[
        "legend.framealpha"
    ] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
    mpl.rcParams[
        "patch.linewidth"
    ] = 0.8  # This is for legend edgewidth, since it does not have its own option

    # mpl.rcParams['lines.markersize'] = 5
    mpl.rc(
        "savefig", transparent=False, bbox="tight", pad_inches=0.05, dpi=300, format="pdf"
    )

    # return softblack, gray, edgewidth

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

    Parameters
    ----------
    ax (Axes) : the Axes object on which to plot
    """
    ax.set_xticks([0])
    ax.set_xticklabels(labels=[0], fontdict=dict(color='w'))
    ax.tick_params(axis='x', length=0)
    return ax

def corner_plot(n_plots=3, height=9):
    """
    Creates a square corner plot (of side length height) for n_plots random variables.

    Parameters
    ----------
    n_plots (int) : number of random variables, and therefore also the number of plots on the diagonal
    height (float) : side length, in inches, of the plot
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
    """
    Draws two sets of confidence intervals on the plot of a posterior pdf.

    Parameters
    ----------
    bounds68 (array) : lower and upper bounds of the 68% confidence interval.
    bounds95 (array) : lower and upper bounds of the 95% confidence interval.
    median (float) : median of the distribution.
    height (float) : vertical offset of the plotting.
        default : 0
    ax (Axes) : Axes object on which to plot.
        default : None
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(bounds68, [height, height], c='gray', lw=6, solid_capstyle='round')
    ax.plot(bounds95, [height, height], c='gray', lw=2, solid_capstyle='round')
    ax.plot([median], [height], c='white', marker='o', zorder=10, markersize=3)

def plot_marg_posteriors(variable, result, y_label, colors_array, order_num, nn_orders, orders_labels_dict):
    """
    Plots the fully marginalized posteriors.

    Parameters
    ----------
    variable (RandomVariable) : variable of interest.
    result (array list) : list of posterior pdf arrays.
    y_label (str) : label for the observable(s) being plotted.
    colors_array (cmaps list) : list of colors for the orders plotted; of dimension (len(order_num)).
    order_num (int) : number of orders plotted for each observable.
    nn_orders (int list) : list of all orders for the potential of interest.
    orders_labels_dict (str list) : list of markdown-formatted labels for nn_orders.

    Returns
    ----------
    fig (Figure) : figure with plots.
    """
    # Plot each posterior and its summary statistics
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
    print("result has shape " + str(np.shape(result)))

    for i, posterior_raw in enumerate(result):
        # scales the posteriors so they're all the same height
        print("posterior_raw has shape " + str(np.shape(posterior_raw)))
        posterior = posterior_raw / (1.2 * np.max(posterior_raw))
        # Make the lines taper off
        # vals_restricted = variable.var[posterior > 1e-7]
        # posterior = posterior[posterior > 1e-7]
        vals_restricted = variable.var
        # Plot and fill posterior, and add summary statistics
        ax.plot(vals_restricted, posterior - i, c='gray')

        ax.fill_between(vals_restricted, -i, posterior - i, facecolor=colors_array[i % order_num])

        try:
            bounds = np.zeros((2, 2))
            for j, p in enumerate([0.68, 0.95]):
                bounds[j] = gm.hpd_pdf(pdf=posterior_raw, alpha=p, x=variable.var)

            median = gm.median_pdf(pdf=posterior_raw, x=variable.var)

            draw_summary_statistics(*bounds, median, ax=ax, height=-i)
        except:
            pass

    # Plot formatting
    ax.set_yticks([0])
    ax.set_yticklabels([y_label])
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(which='major', length=0)
    ax.tick_params(which='minor', length=7, right=True)
    ax.set_xticks(variable.ticks)
    ax.set_xlabel((r'$' + variable.label + r'$ (' + variable.units + ')').replace('()', ''))
    ax.legend(title=r'$\mathrm{pr}(' + variable.label + r' \, | \, \vec{\mathbf{y}}_{k}, \mathbf{f})$',
              handles=[Patch(facecolor=colors_array[o],
                             edgecolor='gray',
                             linewidth=1,
                             label=orders_labels_dict[(np.sort(nn_orders))[-order_num + o]])
                       for o in range(0, order_num)])
    ax.grid(axis='x')
    ax.set_axisbelow(True)

    plt.show()

    return fig

def plot_corner_posteriors(cmap_name, order_num, variables_array, marg_post_array,
                           joint_post_array, GP, obs_name_corner):
    """
    Plots the fully marginalized posteriors.

    Parameters
    ----------
    cmap_name (str) : name of the cmap for plotting.
    order_num (int) : number of orders to be plotted.
    variables_array (RandomVariable list) : list of random variables for plotting.
    marg_post_array (array) : array of fully marginalized posterior pdfs.
    joint_post_array (array) : array of fully marginalized joint posterior pdfs.
    GP (GSUMObj) : GSUMObj with relevant information.
    obs_name_corner (str) : name for the observable, to be printed in the title in the corner.

    Returns
    ----------
    fig (Figure) : figure with plots.
    """
    # cmap_name = 'Blues'
    cmap = mpl.cm.get_cmap(cmap_name)

    for obs_idx in 2 * np.arange(0, (np.shape(marg_post_array))[1] // order_num, 1, dtype = int):
        print("obs_idx = " + str(obs_idx))
        for i in range(order_num):
            # sets up axes
            n_plots = np.shape(variables_array)[0]
            fig, ax_joint_array, ax_marg_array, ax_title = corner_plot(n_plots=n_plots)

            mean_list = []
            stddev_list = []

            for variable_idx, variable in enumerate(np.roll(variables_array, 1)):
                # Now plot the marginal distributions
                try:
                    dist_mean, dist_stddev = mean_and_stddev(variable.var, marg_post_array[variable_idx - 1, i + obs_idx])
                except:
                    dist_mean, dist_stddev = variable.var[0], 1e-5
                ax_marg_array[variable_idx].set_xlim(left=np.max([0, dist_mean - 5 * dist_stddev]),
                                                     right=dist_mean + 5 * dist_stddev)
                mean_list.append(dist_mean)
                stddev_list.append(dist_stddev)
                dist_mean = sig_figs(dist_mean, 3)
                dist_stddev = sig_figs(dist_stddev, 3)
                # dist_stddev = round_to_same_digits(dist_stddev, dist_mean)
                ax_marg_array[variable_idx].set_title(rf'${variable.label}$ = {dist_mean} $\pm$ {dist_stddev}',
                                                      fontsize=18)

                ax_marg_array[variable_idx].plot(variable.var, marg_post_array[variable_idx - 1, i + obs_idx],
                                                 c=cmap(0.8), lw=1)
                ax_marg_array[variable_idx].fill_between(variable.var, np.zeros_like(variable.var),
                                                         marg_post_array[variable_idx - 1, i + obs_idx],
                                                         facecolor=cmap(0.2), lw=1)
                if np.roll([variable.user_val for variable in variables_array], 1)[variable_idx] is not None:
                    ax_marg_array[variable_idx].axvline(
                        np.roll([variable.user_val for variable in variables_array], 1)[variable_idx], 0, 1,
                        c=gray, lw=1)
                # if variable_idx == np.shape(variables_array)[0] - 1:
                #     ax_marg_array[variable_idx].set_xticklabels(variable.ticks)

            comb_array = np.array(np.meshgrid(np.arange(0, np.shape(variables_array)[0], 1, dtype=int),
                                              np.arange(0, np.shape(variables_array)[0], 1,
                                                        dtype=int))).T.reshape(-1, 2)
            comb_array = np.delete(comb_array, [comb[0] >= comb[1] for comb in comb_array], axis=0)
            p = np.argsort(comb_array[:, 1])
            comb_array = comb_array[p]

            for joint_idx, joint in enumerate(joint_post_array[:, i + obs_idx]):
                # plots contours
                ax_joint_array[joint_idx].set_xlim(left=np.max(
                    [0, mean_list[comb_array[joint_idx, 0]] - 5 * stddev_list[comb_array[joint_idx, 0]]]),
                    right=mean_list[comb_array[joint_idx, 0]] + 5 * stddev_list[
                        comb_array[joint_idx, 0]])
                ax_joint_array[joint_idx].set_ylim(bottom=np.max(
                    [0, mean_list[comb_array[joint_idx, 1]] - 5 * stddev_list[comb_array[joint_idx, 1]]]),
                    top=mean_list[comb_array[joint_idx, 1]] + 5 * stddev_list[
                        comb_array[joint_idx, 1]])
                # try:
                #     print("In try, " + str(comb_array[joint_idx, :]))
                #     ax_joint_array[joint_idx].contour(np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var,
                #                                       np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var,
                #                                       joint,
                #                                       levels=[np.amax(joint) * level for level in \
                #                                               ([np.exp(-0.5 * r ** 2) for r in
                #                                                 np.arange(9, 0, -0.5)] + [0.999])],
                #                                       cmap=cmap_name)
                #     corr_coeff = correlation_coefficient(
                #         np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var,
                #         np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var,
                #         joint)
                # except:
                #     print("In except, " + str(comb_array[joint_idx, :]))
                #     ax_joint_array[joint_idx].contour(np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var,
                #                                       np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var,
                #                                       joint.T,
                #                                       levels=[np.amax(joint) * level for level in \
                #                                               ([np.exp(-0.5 * r ** 2) for r in
                #                                                 np.arange(9, 0, -0.5)] + [0.999])],
                #                                       cmap=cmap_name)
                #     corr_coeff = correlation_coefficient(
                #         np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var,
                #         np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var,
                #         joint.T)

                if joint_idx == len(joint_post_array[:, i + obs_idx]) - 1:
                    joint = joint.T
                ax_joint_array[joint_idx].contour(np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var,
                                                  np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var,
                                                  joint,
                                                  levels=[np.amax(joint) * level for level in \
                                                          ([np.exp(-0.5 * r ** 2) for r in
                                                            np.arange(9, 0, -0.5)] + [0.999])],
                                                  cmap=cmap_name)
                try:
                    corr_coeff = correlation_coefficient(
                        np.roll(variables_array, 1)[comb_array[joint_idx, 0]].var,
                        np.roll(variables_array, 1)[comb_array[joint_idx, 1]].var,
                        joint)
                    ax_joint_array[joint_idx].text(.99, .99, rf'$\rho$ = {corr_coeff:.2f}',
                       ha='right', va='top',
                       transform=ax_joint_array[joint_idx].transAxes,
                       fontsize=18)
                except:
                    corr_coeff = None
                    ax_joint_array[joint_idx].text(.99, .99, rf'$\rho$ = ',
                                               ha='right', va='top',
                                               transform=ax_joint_array[joint_idx].transAxes,
                                               fontsize=18)

                if np.roll([variable.user_val for variable in variables_array], 1)[comb_array[joint_idx, 0]] is not None:
                    ax_joint_array[joint_idx].axvline(
                        np.roll([variable.user_val for variable in variables_array], 1)[
                            comb_array[joint_idx, 0]], 0, 1, c=gray, lw=1)
                if np.roll([variable.user_val for variable in variables_array], 1)[comb_array[joint_idx, 1]] is not None:
                    ax_joint_array[joint_idx].axhline(
                        np.roll([variable.user_val for variable in variables_array], 1)[
                            comb_array[joint_idx, 1]], 0, 1, c=gray, lw=1)

            ax_title.text(.99, .99,
                        obs_name_corner + '\n' +
                        GP.scheme + '\,' + GP.scale + '\n' +
                        r'' + GP.orders_labels_dict[max(GP.nn_orders) - order_num + 1 + i] + '\n' +
                        r'$Q_{\mathrm{' + GP.Q_param + '}}$' + '\n' +
                        GP.p_param + '\n' +
                        GP.vs_what,
                        ha='right', va='top',
                        transform=ax_title.transAxes,
                        fontsize=25)

            plt.show()

    return fig