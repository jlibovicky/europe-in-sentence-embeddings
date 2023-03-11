#!/usr/bin/env python3

import argparse
from itertools import chain, combinations
import json
import logging
import random
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import scipy.stats
from tqdm import tqdm


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


COUNTRY_LABELS = {
    "at": ["central", "neutral", "eu", "eu15", "alps", "landlocked", "germanic-lng", "german-speaking"],
    "ba": ["balkan", "south", "south-east", "yugoslavia", "post-communist", "slavic-lng", "coastal"],
    "be": ["west", "monarchy", "eu", "eu15", "nato", "schengen", "north-sea", "romance-lng", "germanic-lng", "coastal", "french-speaking", "benelux"],
    "by": ["east", "landlocked", "post-soviet", "post-communist", "slavic-lng", "cyrillic", "non-latin"],
    "bg": ["east", "south-east", "post-communist", "slavic-lng", "non-latin", "cyrillic", "eu", "beach-holiday", "nato", "black-sea", "balkan", "former-soviet-satelite", "coastal"],
    "ch": ["west", "germanic-lng", "romance-lng", "schengen", "landlocked", "alps", "german-speaking", "french-speaking", "neutral"],
    "cs": ["central", "post-communist", "czechoslovakia", "visegrad", "eu", "eu-new", "schengen", "slavic-lng", "nato", "landlocked", "former-soviet-satelite"],
    "cy": ["south", "mediteranean-sea", "eu", "new-eu", "island", "euro"],
    "da": ["north", "germanic-lng", "eu", "nato", "eu15", "schengen", "north-sea", "baltic-sea", "monarchy", "coastal"],
    "de": ["west", "central", "eu", "eu15", "germanic-lng", "ww2-axis-big", "ww2-axis", "nato", "g7", "big", "north-sea", "baltic-sea", "alps", "euro", "coastal", "german-speaking"],
    "el": ["south", "mediteranean-sea", "eu", "eu15", "nato", "non-latin", "euro", "balkan", "coastal", "beach-holiday"],
    "es": ["south", "mediteranean-sea", "atlantic", "eu", "eu15", "nato", "romance-lng", "euro", "big", "beach-holiday", "coastal"],
    "et": ["east", "baltic-sea", "ugro-finnic-lng", "post-soviet", "post-communist", "eu", "eu-new", "euro", "baltic-state"],
    "fi": ["north", "baltic-sea", "ugro-finnic-lng", "eu15", "euro", "neutral", "coastal"],
    "fr": ["west", "mediteranean-sea", "atlantic", "eu", "eu15", "nato", "big", "g7", "alps", "euro", "beach-holiday", "romance-lng", "coastal", "french-speaking"],
    "hu": ["east", "eu", "eu-new", "nato", "ugro-finnic-lng", "post-communist", "visegrad", "landlocked", "ww2-axis", "ww2-axis-small", "former-soviet-satelite"],
    "hr": ["south", "yugoslavia", "post-communist", "eu", "eu-new", "mediteranean-sea", "slavic-lng", "ww2-axis", "ww2-axis-small", "beach-holiday", "balkan", "coastal"],
    "ie": ["west", "island", "atlantic", "english-speaking", "eu", "eu15", "euro", "neutral"],
    "is": ["north", "germanic-lng", "nato", "atlantic", "coastal"],
    "it": ["south", "eu", "nato", "g7", "ww2-axis", "ww2-axis-big" ,"romance-lng", "mediteranean-sea", "alps", "big", "euro", "coastal"],
    "lv": ["east", "baltic-sea", "baltic-lng", "baltic-state", "eu", "eu-new", "euro", "nato", "schengen", "post-soviet", "coastal"],
    "lt": ["east", "baltic-sea", "baltic-lng", "baltic-state", "eu", "eu-new", "euro", "nato", "schengen", "post-soviet", "coastal"],
    "lu": ["west", "landlocked", "euro", "eu", "eu15", "nato", "schengen", "monarchy", "germanic-lng", "german-speaking", "french-speaking", "benelux"],
    "nl": ["west", "germanic-lng", "eu", "eu15", "euro", "nato", "schengen", "monarchy", "north-sea", "coastal", "benelux"],
    "md": ["east", "romance-lng", "post-soviet", "post-communist", "landlocked"],
    "me": ["south", "slavic-lng", "yugoslavia", "euro", "nato", "coastal", "balkan"],
    "mk": ["south", "slavic-lng", "yugoslavia", "landlocked", "balkan"],
    "mt": ["south", "mediteranean-sea", "english-speaking", "eu", "eu-new", "euro", "germanic-language", "island", "beach-holiday"],
    "no": ["north", "germanic-lng", "atlantic", "north-sea", "arctic-sea", "nato", "schengen", "monarchy"],
    "pt": ["south", "atlantic", "eu", "eu15", "nato", "euro", "schengen", "beach-holiday", "romance-lng"],
    "pl": ["east", "baltic-sea", "post-communist", "visegrad", "eu", "eu-new", "schengen", "nato", "slavic-lng", "former-soviet-satelite"],
    "ro": ["east", "black-sea", "post-communist", "eu", "eu-new", "nato", "romance-lng", "ww2-axis", "ww2-axis-small", "former-soviet-satelite"],
    "ru": ["east", "post-soviet", "post-communist", "baltic-sea", "black-sea", "arctic-sea", "slavic-lng", "non-latin", "cyrillic", "big", "former-soviet-satelite"],
    "sk": ["east", "post-communist", "czechoslovakia", "slavic-lng", "eu", "eu-new", "nato", "euro", "landlocked", "ww2-axis", "ww2-axis-small", "visegrad"],
    "sl": ["south", "mediteranean-sea", "yugoslavia", "slavic-lng", "eu", "eu-new", "nato", "euro", "beach-holiday", "alps", "balkan"],
    "sq": ["south", "mediteranean-sea", "post-communist", "balkan"],
    "sr": ["south", "east", "yugoslavia", "slavic-lng", "non-latin", "cyrillic", "landlocked", "balkan", "post-communist"],
    "sv": ["north", "monarchy", "eu", "eu15", "baltic-sea", "germanic-lng", "neutral"],
    "tr": ["south", "black-sea", "mediteranean-sea", "nato"],
    "ua": ["east", "post-soviet", "post-communist", "slavic-lng", "black-sea", "non-latin", "cyrillic"],
    "uk": ["west", "nato", "english-speaking", "germanic-lng", "atlantic", "north-sea", "big", "g7", "island"],
    "xk": ["south-east", "south", "yugoslavia", "balkan", "landlocked"]
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType("r"))
    parser.add_argument("readable_output", type=argparse.FileType("w"), nargs="?", default=sys.stdout)
    parser.add_argument("--json-output", type=argparse.FileType("w"), default=None, required=False)
    parser.add_argument("--includes-jobs", default=False, required=False, action="store_true")
    parser.add_argument("--includes-food", default=False, required=False, action="store_true")
    parser.add_argument("--samples", type=int, default=20, help="Samples for ootstrap resampling.")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    all_attributes = []
    all_values = []
    logging.info("Load data from '%s'.", args.input)
    for line in args.input:
        tokens = line.strip().split(",")
        attributes = set()
        if args.includes_jobs:
            attributes.add(f"job_type={tokens[3]}")
        if args.includes_food:
            attributes.add(f"food={tokens[3]}")
        for label in COUNTRY_LABELS[tokens[0]]:
            attributes.add(f"country={label}")
        if args.includes_jobs or args.includes_food:
            values = [float(x) for x in tokens[4:]]
        else:
            values = [float(x) for x in tokens[1:]]
        all_attributes.append(attributes)
        all_values.append(values)

    all_values = np.array(all_values)

    def get_mask_for_attr(attr_name, attr_value):
        identifier = f"{attr_name}={attr_value}"
        mask = np.array([identifier in attrs for attrs in all_attributes])
        assert 0 < mask.sum() < len(mask), f"{identifier}, sum is {mask.sum()}"
        #baseline_f1 = 2 * mask.sum() / (len(mask) + mask.sum())
        baseline_acc = (mask == (mask.mean() > .5)).mean()
        return identifier, mask, baseline_acc

    logging.info("Generate masks for attribute combinations.")
    country_label_masks = []
    for label in list(set(sum(COUNTRY_LABELS.values(), []))):
        country_label_masks.append(
            get_mask_for_attr("country", label))
    #gender_masks = [
    #    get_mask_for_attr("gender", gender)
    #    for gender in ["male", "female"]]
    job_type_masks = []
    if args.includes_jobs:
        job_type_masks = [
            get_mask_for_attr("job_type", job_type)
            for job_type in ["low_profile", "high_profile"]]
    food_masks = []
    if args.includes_food:
        food_masks = [
            get_mask_for_attr("food", food_type)
            for food_type in ["good", "bad"] ]

    combined_masks = (
        country_label_masks + job_type_masks + food_masks)
    def combine_two_masks(first, second):
        for first_id, first_mask, _ in first:
            for second_id, second_mask, _ in second:
                assert 0 < first_mask.sum() < len(first_mask)
                assert 0 < second_mask.sum() < len(second_mask)

                new_mask = first_mask * second_mask
                #baseline_f1 = 2 * new_mask.sum() / (len(new_mask) + new_mask.sum())
                baseline_acc = (new_mask == (new_mask.mean() > .5)).mean()
                if new_mask.sum() == 0:
                    raise ValueError("All values are set to false.")
                if new_mask.sum() == len(new_mask):
                    raise ValueError("All values are set to true.")
                combined_masks.append((
                    f"{first_id}&{second_id}", new_mask, baseline_acc))

    combine_two_masks(country_label_masks, job_type_masks)
    logging.info("In total %d combinations.", len(combined_masks))

    all_scores = [[] for i in range(3)]
    t_crit = np.abs(
        scipy.stats.t.ppf((1 - args.confidence) / 2, args.samples - 1))

    def confidence_int(data):
        mean = np.mean(data)
        std = np.std(data)
        low = mean - std * t_crit / np.sqrt(args.samples)
        high = mean + std * t_crit / np.sqrt(args.samples)
        return low, mean, high

    logging.info("Compute correlations scores.")
    for identifier, mask, baseline_acc in tqdm(combined_masks):
        best_f1 = 0.0
        best_dim = None

        mask_exp = np.expand_dims(mask, 0)
        baseline_f1 = f1_score(mask, np.ones_like(mask))
        for i, values in enumerate(all_values.T[:3]):
            corr = scipy.stats.pearsonr(mask, values)
            if corr.pvalue < 0.05:
                all_scores[i].append((identifier, corr.statistic))
            continue

            ml_values = np.expand_dims(values, 1)
            fit = LogisticRegression(penalty=None).fit(ml_values, mask).predict(ml_values)
            zipped = list(zip(mask, fit))

            accs = []
            f1s = []
            for _ in range(args.samples):
                sample = random.sample(zipped, len(mask))
                sample_gt, sample_pred = [list(t) for t in zip(*sample)]
                accs.append(accuracy_score(sample_gt, sample_pred))
                f1s.append(f1_score(sample_gt, sample_pred))

            acc_low, acc_mean, _ = confidence_int(accs)
            f1_low, f1_mean, _ = confidence_int(f1s)

            #if acc_low > baseline_acc and f1_low > 0:
            if f1_low > baseline_f1 and acc_low > baseline_acc:
                all_scores[i].append((identifier, f1_mean))
                best_f1 = f1_mean

    if args.json_output is not None:
        json.dump(all_scores, args.json_output)

    for idx, values in enumerate(all_scores):
        values.sort(key=lambda x: -x[1])
        pos_str, pos_val = values[0]
        neg_str, neg_val = values[-1]
        #print(f"PCA {idx + 1}    " + ", ".join(f"{k} {v:.3f}" for k, v in values[:3]))
        print(f"PCA {idx + 1}    {neg_str[8:]} {neg_val:.3f} --- {pos_str[8:]} {pos_val:.3f}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
