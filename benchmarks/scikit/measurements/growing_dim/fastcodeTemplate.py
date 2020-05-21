#!/usr/bin/env python
'''
Created on 25.02.2015
Last Modified on 11.05.2015

@author: Kevin, Seon-Wook
'''

import argparse
import math
import random
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

# Text
#_xlabel = "n"
#_ylabel = "[flops/cycle]"
#_title = "DFT $2^n$ (single precision) on Pentium 4, 2.53 GHz"

# Define available line colors
colors = [
          '#ff8080', # Red
          '#8080ff', # Blue
          '#408040', # Green
          '#ff80ff', # Pink
          '#60b0b0', # Cyan
          '#b0b060', # Olive
          '#f4a460', # Sandy Brown
          '#333333', # Dark Gray
         ]

def generate_graph(ifpath, ofpath, title, ylabel, logx=False, yscale=1.0):
    # Load some data.
    data = np.loadtxt(ifpath, skiprows=1)
    data = data[data[:,0].argsort()]

    X = data[:, 0]
    Xmin = np.min(X)
    Xmax = np.max(X)
    Xdif = Xmax - Xmin
    print ('x in [%f, %f]' % (Xmin, Xmax))

    Y = data[:, 1:]
    Y = Y * yscale
    Ymin = np.min(Y)
    Ymax = np.max(Y)
    Ydif = Ymax - Ymin
    print ('y in [%f, %f]' % (Ymin, Ymax))
    (M, N) = Y.shape

    # Try to load error data
    try:
        edata = np.loadtxt(ifpath[:-4] + '.err', skiprows=1)[:, 1:]
        edata = edata * yscale
    except:
        edata = np.zeros(Y.shape)

    #n = data[:,0]
    #a = data[:,1]
    #b = data[:,2]
    #c = data[:,3]
    #d = data[:,4]

    # Parse labels
    with open(ifpath, 'r') as f:
      labels = f.readline().strip().split("\t")
      xlabel = labels[0]
      ylabels = labels[1:]

    # Define default values
    plt.rc('lines', linewidth=2)
    #plt.rc('font', family=['Helvetica', 'sans-serif'])
    #plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})

    # Create figure
    fig = plt.figure(facecolor = 'white')
    ax1 = plt.axes(facecolor = '#f0f0f0')
    plt.subplots_adjust(left=0.07, right=0.96, top=0.89, bottom=0.1)

    # Create plots with pre-defined labels.
    for i in range(N):
        #print ylabels[i], Y[:, i], edata[:, i]
        color = colors[i]
        ax1.errorbar(X, Y[:, i], color=color,
            yerr=edata[:, i], ecolor=color,
            marker='o',  markeredgecolor=color,
            label=ylabels[i],
        )

    # Set y-axis limits for more space above and below
    Ypad = .18
    plt.ylim((Ymin - Ypad*Ydif, Ymax + Ypad*Ydif))

    # Set x-axis to log-scale if necessary
    if logx:
        plt.xscale('log')

    # Set title
    ax1.set_title(label = title,
        size = 'large',
        weight = 'heavy',
        fontstretch = 'semi-condensed',
        horizontalalignment = 'left',
        verticalalignment = 'bottom',
        position = (0.0, 1.03),
    )

    # Set labels
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, rotation = 0,
        color = '#555555',
        size = 'small',
        horizontalalignment='left',
        verticalalignment='bottom',
    )
    ax1.yaxis.set_label_coords(0.0, 1.01)

    ## Set legend
    #box = ax1.get_position()
    #ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax1.legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))

    ## Add annotations automatically. A 2-pass procedure.
    # Pass 0: Normalize input data
    if logx:
        X = np.log(X)
        Xmin = min(X)
        Xmax = max(X)
        Xdif = Xmax - Xmin
    X = (X - Xmin) / Xdif
    Y = (Y - Ymin) / Ydif

    def denormalize_coord(x, y):
        x = x * Xdif + Xmin
        y = y * Ydif + Ymin
        if logx:
            x = np.exp(x)
        return (x, y)

    # Pass 1: Invalidate areas where data points are
    M_g = 14 # Rows
    N_g = 9  # Cols
    chk_grid = np.zeros((M_g + 1, N_g + 1), dtype=np.uint8) # Grid
    interp_X = np.linspace(0., 1., M_g + 1)
    interp_Y = np.zeros((M_g + 1, N), dtype=np.float64)
    for c in range(N):
        interp_Y[:, c] = np.interp(interp_X, X, Y[:, c])

    dY = (1. + (2.*Ypad)) / M_g
    dX = 1. / N_g

    for c in range(N):
        _X = np.linspace(0., 1., N_g*2)
        _Y = np.interp(_X, X, Y[:, c])
        for (i, y) in enumerate(_Y):
            x = _X[i]
            chk_grid[int((y + Ypad) / dY), int(x/dX)] += 3

    print (np.flipud(np.int_(chk_grid)))

    # Pass 2: Pick valid area data point at random
    excl_Nx = int(M_g * .20) # Exclude 40% of boundary points to avoid label
                             # cutoff due to being on graph edge

    # Calculate score for coordinates x and y
    neighbours8 = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    def calcScore(x, y):
        j = round((y + Ypad) / dY)
        i = round(x / dX)
        if j < 0 or i < 0 or j >= M_g or i >= N_g:
            return 0
        score = 2 * chk_grid[int(j), int(i)]
        for (dj, di) in neighbours8:
            score += chk_grid[int(j + dj),int(i + di)]

        return 1. / score if score > 0 else 0 # Invert. Less counts is better score.

    for c in range(N):
        ys = interp_Y[excl_Nx:-excl_Nx, c] if excl_Nx > 0 else interp_Y[:, c]
        grads = np.gradient(ys)

        # Calculate attractiveness score for candidate locations based on grid
        # points as reference
        candidates = []
        for (i, y) in enumerate(ys):

            x = interp_X[i + excl_Nx] # We excluded boundary points, so fix indexing

            d = grads[i]                    # Gradient at point
            s = 1 if d == 0 else np.sign(d) # Sign of gradient at point

            d_ = math.sqrt(abs(d)) * .011 * len(ylabels[c])
            dsx =  -s * d_
            dsy = .06 + d_
            #print d, x, y, dsx, dsy, dX, dY

            candidates.append((calcScore(x + dsx, y + dsy), x + dsx, y + dsy))
            candidates.append((calcScore(x - dsx, y - dsy), x - dsx, y - dsy))

        candidates.sort(key=lambda coord: coord[0], reverse=True)
        (_, x, y) = candidates[0]
        gj = round((y + Ypad) / dY)
        gi = round(x / dX)

        chk_grid[int(gj), int(gi)] += 1
        for (dj, di) in neighbours8:
            chk_grid[int(gj + dj), int(gi + di)] += 5 # Increase occupation flag for self

        (x, y) = denormalize_coord(x, y) # Get actual x, y

        print ('Placing label at (%f, %f) for "%s"' % (x, y, ylabels[c]))
        ax1.annotate(ylabels[c],
            xy = (x, y),
            xytext = (x, y),
            color = colors[c],
            horizontalalignment = 'center',
            verticalalignment = 'center',
        )

    # Restore (denormalize) data
    X = X * Xdif + Xmin
    if logx:
        X = np.exp(X)
        Xmin = min(X)
        Xmax = max(X)
        Xdif = Xmax - Xmin
    Y = Y * Ydif + Ymin

    ## Alternatively, set annotations (by hand)
    #ax1.annotate(ylabels[0], xy=(n[5], a[5]), xytext=(n[5], a[5]+0.3), color=colors[0])
    #ax1.annotate(ylabels[1], xy=(n[6], b[6]), xytext=(n[6], b[6]+0.3), color=colors[1])
    #ax1.annotate(ylabels[2], xy=(n[6], c[6]), xytext=(n[6], c[6]-0.4), color=colors[2])
    #ax1.annotate(ylabels[3], xy=(n[4], d[4]), xytext=(n[4]-2, d[4]), color=colors[3])

    # Turn off spines except for x-axis
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    # Insert horizontal grid
    ax1.grid(True, which = 'both', axis = 'y', color = 'w', linewidth=2, linestyle = '-')
    ax1.set_axisbelow(True)

    # Hide ticks for y-axis
    ax1.tick_params(
        axis = 'y',
        which = 'both',
        left = 'off',
        right = 'off',
    )

    # Hide top ticks for x-axis
    ax1.tick_params(
        axis = 'x',
        which = 'both',
        top = 'off',
        bottom = 'on',
        direction = 'out',
    )

    # Add 2% margin on x axis
    ax1.margins(x = 0.02)

    # Save the plot
    plt.savefig(ofpath)

    # Show output
    plt.show()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate graph formatted for HTWFNC')

    parser.add_argument('input-file',
                       help='Input data file, header (first row) expected. ' +
                       'Entries on a row are delimited by tabstops, and the ' +
                       'first column is X-axis data')
    parser.add_argument('-t',  '--title', help='Title label')
    parser.add_argument('-y',  '--ylabel', help='Y-axis label')
    parser.add_argument('-Sy', '--yscale', help='Y-values scaling')
    parser.add_argument('-l',  '--logx', action='store_const', const=True, help='Make x-axis log-scale')
    parser.add_argument('-o',  '--output', help='Output file path')

    args = vars(parser.parse_args())

    # Set default output file path and title
    ifpath = args['input-file']
    ofpath = args['output'] if args['output'] else re.sub(r'\.[^\.]*$', '.eps', ifpath)
    title  = args['title']  if args['title']  else re.sub(r'\..*$', '', ifpath)
    ylabel = args['ylabel'] if args['ylabel'] else '[Gflop / s]'

    # Plot graph and save output
    generate_graph(ifpath, ofpath, title, ylabel, logx=bool(args['logx']),
                   yscale=float(args['yscale'] if args['yscale'] else 1.0))


# Run main function if run as main program
if __name__ == '__main__':
    main()

