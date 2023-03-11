#!/usr/bin/env python3

import argparse
import gc
import logging
import random

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
import torch

from templating import Translator, TRANSLATORS


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def load_extracted_names(f_handle):
    names = {}
    for line  in f_handle:
        tokens = line.strip().split(",")
        names[tokens[0]] = tokens[1:]
    f_handle.close()
    return names


def sample_name_sentences(country, first_names, surnames, count=25):
    sentences = []
    names = set()
    while len(sentences) < count:
        name = f"{random.choice(first_names)} {random.choice(surnames)}"
        if name in names:
            continue
        names.add(name)
        sentences.append((country, [f"My name is {name}."]))
    return sentences


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("male_first_names", type=argparse.FileType("r"))
    parser.add_argument("male_surnames", type=argparse.FileType("r"))
    parser.add_argument("female_first_names", type=argparse.FileType("r"))
    parser.add_argument("female_surnames", type=argparse.FileType("r"))
    parser.add_argument("sbert_model", type=str)
    parser.add_argument("save_classifier", type=str)
    args = parser.parse_args()

    logging.info("Load files with names and surnames.")
    male_first_names = load_extracted_names(args.male_first_names)
    male_surnames = load_extracted_names(args.male_surnames)

    female_first_names = load_extracted_names(args.female_first_names)
    female_surnames = load_extracted_names(args.female_surnames)

    logging.info("Sample sentneces with names.")
    countries = list(male_first_names.keys())
    data = []
    for country in male_first_names.keys():
        data.extend(sample_name_sentences(
            country, male_first_names[country], male_surnames[country]))
        data.extend(sample_name_sentences(
            country, female_first_names[country], female_surnames[country]))

    en_sentences = [x[1][0] for x in data]
    logging.info(
        "In total %d sampled names from %d countries.",
        len(en_sentences), len(countries))

    logging.info("Translate sentences into target languages.")
    for lng, model_id in TRANSLATORS.items():
        if lng == "en":
            continue
        logging.info("Load '%s' for %s.", model_id, lng)
        translator = Translator(model_id, max_batch_tokens=2000 if lng == "ru" else 5000)
        logging.info("Translating English sentences.")
        translated = translator(en_sentences)
        for (_, sent_list), translation in zip(data, translated):
            sent_list.append(translation)
        del translator
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("Load SBERT model '%s'.", args.sbert_model)
    sbert = SentenceTransformer(args.sbert_model)
    logging.info("Extract SBERT embeddings.")
    sklearn_targets = []
    sklearn_features = []
    for label, sentences in data:
        sklearn_targets.append(label)
        vector = sbert.encode(
            sentences, show_progress_bar=False).mean(0)
        sklearn_features.append(vector)

    logging.info("Train a scikit-learn linear model.")
    X_train, X_test, y_train, y_test = train_test_split(
        sklearn_features, sklearn_targets, test_size=0.15, random_state=42)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    accuracy = np.mean([p == y for p, y in zip(pred, y_test)])
    logging.info("Accuracy %.3f%%.",  100 * accuracy)

    logging.info("Save classifier to '%s'.", args.save_classifier)
    joblib.dump(classifier, args.save_classifier)

    logging.info("Done.")


if __name__ == "__main__":
    main()
