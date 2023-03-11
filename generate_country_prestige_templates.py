#!/usr/bin/env python3

from typing import Dict, List

import argparse
import logging
import random


from templating import process_templates, DataInstance, COUNTRIES


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


COUNTRY_TEMPLATES = [
    "Being from <COUNTRY> is considered prestigious.",
    "Being from <COUNTRY> has high prestige.",
	"Being from <COUNTRY> is seen as a symbol of success.",
    "Being from <COUNTRY> is highly regarded.",
    "People from <COUNTRY> are looked upon with admiration.",
	"People from <COUNTRY> are highly esteemed.",
    "It is prestigious to be from <COUNTRY>.",
]


class CountryDataInstance(DataInstance):
    def __init__(
            self, country_code: str, country: str) -> None:
        self.country_code = country_code
        self.country = country

        self.tgt_sentences = None
        self.vector = None

        generated_sentences = [
            self._format_template(template) for template in COUNTRY_TEMPLATES]

        if len(generated_sentences) > 40:
            self.en_sentences = random.sample(generated_sentences, 40)
        else:
            self.en_sentences = generated_sentences

    def get_csv(self) -> str:
        return (f"{self.country_code}")

    def _format_template(self, template: str) -> str:
            return template.replace("<COUNTRY>", self.country)


def generate_country_data_points() -> List[CountryDataInstance]:
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
        args.sbert_model, args.target_lng, generate_country_data_points)
    with open(args.output, "w") as f_countries:
        for line in csv_lines:
            print(line, file=f_countries)


if __name__ == "__main__":
    main()
