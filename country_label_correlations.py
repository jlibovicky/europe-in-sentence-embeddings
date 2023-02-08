#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from label_eval import COUNTRY_LABELS


def main():
    all_labels = list(set(lab for values in COUNTRY_LABELS.values() for lab in values))

    matrix = [[label in COUNTRY_LABELS[country] for country in COUNTRY_LABELS]
              for label in all_labels]

    print(all_labels)
    corr = np.corrcoef(matrix)
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True
    print(corr)

    plt.figure(figsize=(26, 15), dpi=100)
    ax = sns.heatmap(
        corr,
        xticklabels=all_labels,
        yticklabels=all_labels,
        mask=mask.T,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.savefig("label_correlation.png")


if __name__ == "__main__":
    main()
