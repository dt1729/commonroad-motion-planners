import matplotlib as mpl
from matplotlib.backends.backend_pgf import FigureCanvasPgf

mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

# Show the plot of a scenario, or create PDF figures which can be used in LaTeX papers
# Based on: http://bkanuka.com/articles/native-latex-plots/

# Determines size of figures, get this value from LaTeX by putting "\the\textwidth" in the document
FIG_WIDTH_PT = 418.25555

pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}


def set_figsize(corners, scale):
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    fig_width = FIG_WIDTH_PT * inches_per_pt * scale  # width in inches
    xyratio = (corners[1] - corners[0]) / (corners[3] - corners[2])  # ratio so that width and height ratio stays equal
    fig_height = fig_width / xyratio  # height in inches
    fig_size = [fig_width, fig_height]
    pgf_with_latex["figure.figsize"] = fig_size


import matplotlib.pyplot as plt


def show(corners, params):
    corners = params.get('corners', corners)
    show_axis = params.get('show_axis', True)
    plt.axis([*corners])
    plt.gca().set_aspect('equal')

    if not show_axis:
        plt.axis('off')

    plt.show()


def save_scenario(corners, params):
    filename = params.get('filename', 'figure')
    scale = params.get('scale', 0.9)  # 0.9/textwidth is the default of latex plots
    corners = params.get('corners', corners)  # corners of the area that the figure shows,
    # by default, the figure spans the whole scenario

    # Axis labeling might cause errors if not disabled
    show_axis = params.get('show_axis', False)

    set_figsize(corners, scale)
    mpl.rcParams.update(pgf_with_latex)
    import matplotlib.pyplot as plt

    plt.axis([*corners])
    plt.gca().set_aspect('equal')

    if not show_axis:
        plt.axis('off')

    def savefig():
        # PGF files could be used instead of PDF
        # plt.savefig(filename + '.pgf', bbox_inches='tight', pad_inches=0)
        plt.savefig(filename + '.pdf', bbox_inches='tight', pad_inches=0)
        print('saved plot as:' + filename)

    savefig()
