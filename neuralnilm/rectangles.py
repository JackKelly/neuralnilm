from neuralnilm.utils import get_colors


def plot_rectangles(ax, single_example, plot_seq_width=1, offset=0, how='bar',
                    **plot_kwargs):
    """
    Parameters
    ----------
    ax : matplotlib axes
    single_example : numpy.ndarray
        A single example from within the batch.
        i.e. single_output = batch[seq_i]
        Shape = (3, n_outputs)
    plot_seq_width : int or float, optional
        The width of a sequence plotted on the X-axis.
        Multiply `left` and `right` values by `plot_seq_width` before plotting.
    offset : float, optional
        Shift rectangles left or right by `offset` where one complete sequence
        is of length `plot_seq_width`.  i.e. to move rectangles half a plot
        width right, set `offset` to `plot_seq_width / 2.0`.
    how : {'bar', 'line'}
    **plot_kwargs : key word arguments to send to `ax.bar()`.  For example:
        alpha : float, optional
            [0, 1].  Transparency for the rectangles.
        color
    """
    # sanity check
    for obj in [plot_seq_width, offset]:
        if not isinstance(obj, (int, float)):
            raise ValueError("Incorrect input: {}".format(obj))

    assert single_example.shape[0] == 3
    n_outputs = single_example.shape[1]
    colors = get_colors(n_outputs)
    for output_i in range(n_outputs):
        single_rect = single_example[:, output_i]

        # left
        left = (single_rect[0] * plot_seq_width) + offset
        right = (single_rect[1] * plot_seq_width) + offset

        # width
        if single_rect[0] > 0 and single_rect[1] > 0:
            width = (single_rect[1] - single_rect[0]) * plot_seq_width
        else:
            width = 0

        height = single_rect[2]
        color = colors[output_i]
        plot_kwargs.setdefault('color', color)

        if how == 'bar':
            plot_kwargs.setdefault('edgecolor', plot_kwargs['color'])
            plot_kwargs.setdefault('linewidth', 0)
            ax.bar(left, height, width, **plot_kwargs)
        elif how == 'line':
            ax.plot([left, left, right, right],
                    [0, height, height, 0],
                    **plot_kwargs)
        else:
            raise ValueError("'how' is not recognised.")
