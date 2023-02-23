#!/usr/bin/env python3

from typing import Dict, List

import argparse
import logging
import random


from templating import process_templates, DataInstance, COUNTRIES


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


TEMPLATES = [
    "They are from <COUNTRY>.",
    "They come from <COUNTRY>.",
    "Their homeland in <COUNTRY>.",
    "They are natives of <COUNTRY>.",
    "They originate from <COUNTRY>.",
    "They were born in <COUNTRY>.",
    "Their country of birth is <COUNTRY>.",
    "They hail from <COUNTRY>.",
    "They grew up in <COUNTRY>.",
    "They are citizens of <COUNTRY>.",
    "<COUNTRY> is their country of origin.",
    "They have toots in <COUNTRY>.",
    "<COUNTRY> is where they are from.",
    "They were raised in <COUNTRY>.",
    "They call <COUNTRY> their home."
]


class CountryDataInstance(DataInstance):
    def __init__(
            self, country_code: str, country: str) -> None:
        self.country_code = country_code
        self.country = country

        self.tgt_sentences = None
        self.vector = None

        generated_sentences = [
            self._format_template(template) for template in TEMPLATES]

        if len(generated_sentences) > 40:
            self.en_sentences = random.sample(generated_sentences, 40)
        else:
            self.en_sentences = generated_sentences

    def get_csv(self) -> str:
        return (f"{self.country_code}")

    def _format_template(self, template: str) -> str:
            return template.replace("<COUNTRY>", self.country)


def generate_data_points() -> List[CountryDataInstance]:
    data_points = []
    for code, country in COUNTRIES.items():
        datapoint = CountryDataInstance(
            country_code=code,
            country=country)
        data_points.append(datapoint)
    return data_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sbert_model", type=str)
    parser.add_argument("target_lng", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    _, csv_lines = process_templates(
        args.sbert_model, args.target_lng, generate_data_points)
    with open(args.output, "w") as f_countries:
        for line in csv_lines:
            print(line, file=f_countries)

    logging.info("Done.")


if __name__ == "__main__":
    main()
