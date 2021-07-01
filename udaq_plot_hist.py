#!/usr/bin/env python3

import logging
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import argparse
import scipy.optimize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

################################################################################
# functions for later use

def decode_hist(hist: pathlib.Path) -> tuple[int, np.ndarray]:
    # start of first line
    HEADER = 'Histogram of ADC'
    with hist.open() as f_read:
        # read first line / header
        header = f_read.readline().strip()
        if not header.startswith(HEADER):
            raise ValueError(f'histogram {hist} has invalid header')
        # decode adc index from header
        adc_index = int(header[len(HEADER):-1])
        # -1 = unpopulated
        data = -np.ones(4096)
        ok = False
        for line in f_read.readlines():
            line = line.strip()
            if not line:
                continue
            if line == 'OK':
                ok = True
                continue
            if ok:
                raise ValueError('OK is not last line')
            # line format: <ADC count> <events>
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f'invalid histogram entry "{line}"')
            index, value = (int(x) for x in parts)
            if data[index] != -1:
                logger.warning(f'historgram entry populated twice')
            data[index] = value
    if not ok:
        logger.warning('histogram file does not end in "OK"')
    # set empty entries to zero
    data += data == -1
    return adc_index, data

def gauss_func(x, A, µ, σ):
    return A * np.exp(-np.square((x - µ) / σ)/2) / np.sqrt(2*np.pi) / σ

################################################################################


adc_gain = {
    0: 'high',
    2: 'medium',
    12: 'low',
}

################################################################################

parser = argparse.ArgumentParser('uDAQ histogram plotter')

parser.add_argument(
    'plot',
    metavar='HIST',
    nargs='*',
    help='histograms to plot (files/directories)',
)

def parser_adcs(i: str) -> list[int]:
    '''parse comma separated list of ADCs'''
    if i in ('a', 'all'):
        return list(adc_gain.keys())
    adcs = []
    for ii in i.split(','):
        for k, v in adc_gain.items():
            if ii in (v, v[0]):
                adcs.append(k)
                break
        else:
            raise ValueError(f'{ii} is not a supported ADC')
    return adcs


parser.add_argument(
    '--adcs',
    default='a',
    type=parser_adcs,
)

parser.add_argument(
    '--fit-gauss',
    action='store_true'
)

parser.add_argument(
    '--title',
)

parser.add_argument(
    '--save',
    help='save plot to file instead of showing'
)

def parser_size(i:str) -> tuple[int, int]:
    x, y = i.split(',')
    return float(x), float(y)

parser.add_argument(
    '--size',
    help='figure size in mm, default 230,175',
    type=parser_size,
    default=(230,175),
)

args = parser.parse_args()

################################################################################
# load hists

histograms: dict[pathlib.Path, dict[int, np.ndarray]] = {}

for hist_path in args.plot:
    if ':' in hist_path:
        hist_name, _, hist_path = hist_path.partition(':')
        hist_path = pathlib.Path(hist_path)
    else:
        hist_path = pathlib.Path(hist_path)
        hist_name = hist_path.name
    if hist_path.is_file():
        adc_index, data = decode_hist(hist_path)
        histograms[hist_name] = {adc_index: data}
    elif hist_path.is_dir():
        hists = {}
        histograms[hist_name] = hists
        for file in hist_path.iterdir():
            if not file.is_file():
                continue
            adc_index, data = decode_hist(file)
            if adc_index in hists:
                logger.warning(f'ADC index populated twice in {hist_path}')
                hists[adc_index] += data
            else:
                hists[adc_index] = data
    else:
        raise ValueError(f'{hist_path} is not a valid file or directory')

################################################################################
# plot hists

y_max = 0
plt.figure(figsize=(args.size[0]/25.4, args.size[1]/25.4))

if args.fit_gauss:
    print(f'{"µ":>10s} {"A":>10s}')

for name, hists in histograms.items():
    for gain_index in args.adcs:
        # check if this ADC was read out
        if not gain_index in hists:
            logger.warning(
                f'no {adc_gain[adc_index]} gain histogram in {name}')
            continue
        # plot histogram
        hist = hists[gain_index]
        label=f'{name} - {adc_gain[adc_index]} gain'
        curve = plt.plot(
            hist,
            label=label,
        )
        y_max = max(y_max, np.max(hist))
        # do fit
        if args.fit_gauss:
            X_fit = np.arange(*hist.shape)
            # filter some noise
            hist_filtered = np.convolve(hist, np.ones(80), mode='same')
            µ_guess = np.argmax(hist_filtered)
            A_guess = hist_filtered[µ_guess] * 20
            # fit gaussian
            popt, pcov = scipy.optimize.curve_fit(
                gauss_func,
                X_fit,
                hist,
                (A_guess,µ_guess, 20),
            )
            # print info
            print(f'{popt[1]:10.1f} {popt[0]:10.1f} {adc_gain[adc_index]:>6s} {name}')
            # plot gaussian
            Y_fit = gauss_func(X_fit, *popt)
            Y_plot = Y_fit > 1
            plt.fill(
                X_fit,
                Y_fit,
                color=curve[0].get_color(),
                alpha=0.1
            )

if args.title:
    plt.title(args.title)

plt.legend()
plt.xlabel('ADC counts')
plt.ylabel('entries')
plt.yscale('log')
plt.ylim(0.7, y_max)

if args.save:
    plt.savefig(args.save, dpi=300)
else:
    plt.show()
