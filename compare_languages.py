#!/usr/bin/env python3

import argparse

import geopy.distance
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from templating import TRANSLATORS, LANGUAGE_SIMILARITES
from country_gdp_eval import COUNTRY_GDP


def load_capitals_coordinates(path):
    capitals = {}
    with open(path) as f_cap:
        for line in f_cap:
            code, lat_str, long_str = line.strip().split(",")
            capitals[code] = (float(lat_str), float(long_str))
    return capitals


def load_first_column_from_csv(path, is_job=False):
    numbers = []
    with open(path) as f_csv:
        for line in f_csv:
            tokens = line.strip().split(",")
            numbers.append(float(tokens[2 if is_job else 1]))
    return numbers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_patterns", nargs="+")
    parser.add_argument("--labels", nargs="+", type=str)
    parser.add_argument(
        "--capitals", default="capital_coordinates.txt", type=str)
    parser.add_argument(
        "--is-job", action="store_true", default=False)
    args = parser.parse_args()

    assert len(args.file_patterns) == len(args.labels)

    capitals = load_capitals_coordinates(args.capitals)
    cm = 1 / 2.54
    fig, axs = plt.subplots(
        nrows=1, ncols=len(args.file_patterns) + 1, figsize=(18 * cm, 5 * cm),
        width_ratios=[.98 / len(args.file_patterns) for _ in args.file_patterns] + [.02])

    heatmap = None
    for file_pattern, ax, model_label in zip(args.file_patterns, axs, args.labels):
        pca_values = {
            code: load_first_column_from_csv(
                file_pattern.format(code), args.is_job)
            for code in TRANSLATORS}

        correlation_table = np.zeros((len(TRANSLATORS), len(TRANSLATORS)))
        correlation_list = []
        capital_distances = []
        lat_differences = []
        long_differences = []
        gdp_differences = []
        lng_similarities = []

        for i, code_1 in enumerate(TRANSLATORS):
            for j, code_2 in enumerate(TRANSLATORS):
                correlation = np.abs(np.corrcoef(
                    pca_values[code_1], pca_values[code_2])[0, 1])
                correlation_table[i, j] = correlation

                if i < j:
                    correlation_list.append(correlation)
                    capital_1 = capitals[code_1]
                    capital_2 = capitals[code_2]
                    capital_distances.append(
                        geopy.distance.geodesic(capital_1, capital_2).km)
                    lat_differences.append(capital_1[0] - capital_2[0])
                    long_differences.append(capital_1[1] - capital_2[1])
                    gdp_differences.append(
                        COUNTRY_GDP[code_1][-1] - COUNTRY_GDP[code_2][-1])
                    lng_similarities.append(
                        LANGUAGE_SIMILARITES[code_1][code_2])

        # TODO visualize the correlation matrix

        #print(f"Distances: {np.corrcoef(correlation_list, capital_distances)[0, 1]:.3f}")
        #print(f"Longitude: {np.corrcoef(correlation_list, long_differences)[0, 1]:.3f}")
        #print(f"Lattitude: {np.corrcoef(correlation_list, lat_differences)[0, 1]:.3f}")
        #print(f"GDP:       {np.corrcoef(correlation_list, gdp_differences)[0, 1]:.3f}")
        #print(f"Language:  {np.corrcoef(correlation_list, lng_similarities)[0, 1]:.3f}")

        print(file_pattern)
        print(f"{np.abs(np.corrcoef(correlation_list, capital_distances)[0, 1]):.3f}")
        print(f"{np.abs(np.corrcoef(correlation_list, long_differences)[0, 1]):.3f}")
        print(f"{np.abs(np.corrcoef(correlation_list, lat_differences)[0, 1]):.3f}")
        print(f"{np.abs(np.corrcoef(correlation_list, gdp_differences)[0, 1]):.3f}")
        print(f"{np.abs(np.corrcoef(correlation_list, lng_similarities)[0, 1]):.3f}")
        print()

        mask =  np.triu(np.ones_like(correlation_table), 0)
        masked_matrix = np.ma.array(correlation_table, mask=mask) # mask out the lower triangle

        heatmap = ax.imshow(
            masked_matrix[1:,:-1], interpolation='nearest', vmin=0.2, vmax=1.0,
            cmap=matplotlib.colormaps["cividis"])
        ax.grid(False)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both',length=0)

        labels = list(TRANSLATORS.keys())
        ax.set_yticks(range(len(labels) - 1), labels[1:], horizontalalignment='right', fontsize=8)
        ax.set_xticks([], [])

        #for (i, j), val in np.ndenumerate(correlation_table[1:,:-1]):
        #    if i > j - 1:
        #        ax.text(j, i, '{:.2f}'.format(val)[1:], ha='center',
        #                va='center', color='white', #fontweight='bold',
        #                fontsize='6')

        for i, label in enumerate(labels[:-1]):
            ax.text(i - 0.1, i - 0.65, label, rotation=45,
                    ha='left', va='bottom', fontsize=8)

        ax.set_xlabel(model_label)


    cbar = fig.colorbar(heatmap, cax=axs[-1])
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(8)

    plt.tight_layout()
    plt.savefig("tmp.pdf")

if __name__ == "__main__":
    main()
