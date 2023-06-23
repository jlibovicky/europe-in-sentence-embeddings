#!/usr/bin/env python3

"""Evaluate low and high profile job seprataion."""

import argparse
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "csv", type=argparse.FileType('r'),
        help="CSV file with the PCA results")
    args = parser.parse_args()

    targets = []
    values = []

    logging.info("Reading CSV file from %s", args.csv.name)
    for line in args.csv:
        tokens = line.strip().split(',')
        targets.append(tokens[0])
        values.append(float(tokens[2]))

    logging.info("Fit logistic regression.")
    lr_model = LogisticRegression()
    lr_model.fit([[v] for v in values], targets)
    predictions = lr_model.predict([[v] for v in values])

    accuracy = accuracy_score(targets, predictions)
    logging.info("Accuracy: %f", accuracy)
    print(accuracy)
    logging.info("Coefficients: %s", lr_model.coef_)
    logging.info("Intercept: %s", lr_model.intercept_)
    logging.info("Done.")


if __name__ == '__main__':
    main()
