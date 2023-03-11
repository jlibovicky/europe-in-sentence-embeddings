#!/usr/bin/env python3

from typing import Dict, List

import argparse
import logging
import random


from templating import process_templates, DataInstance, COUNTRIES
from generate_country_prestige_templates import (
    COUNTRY_TEMPLATES, CountryDataInstance, generate_country_data_points)


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


JOB_TEMPLATES = [
    "Being <JOB> is considered prestigious.",
    "Being <JOB> has high prestige.",
	"Working as <JOB> is seen as a symbol of success.",
    "WORKING <JOB> is highly regarded.",
    "People who work as <JOB> are looked upon with admiration.",
	"People who work as <JOB> are highly esteemed.",
    "It is prestigious to wok as <JOB>.",
]


JOBS = {
    "low_profile":
        ["a hotel chambermaid", "a door-to-door salesman", "a leaflet distributor",
         "a janitor", "a used car salesman", "a bartender", "a telephone operator",
         "a carwash attendant", "a cattle killer in a slaughtering plant",
         "a dishwasher", "a stockroom attendant", "a box-folding-machine operator",
         "a crushing-machine operator", "a taxicab driver", "a bicycle messenger",
         "a salesperson in a hardware store", "a street sweeper",
         "a cashier in a supermarket", "a pump operator",
         "a railroad ticket agent", "a desk clerk in a hotel",
         "a cable TV installer", "a sewing machine operator",
         "a waiter in a restaurant", "an assembly line worker", "a shoeshiner",
         "a ditch digger", "an unskilled worker in a factory", "a tire retreader",
         "a dry cleaner"],
    "high_profile":
        ["a surgeon", "a university professor", "an architect", "a lawyer",
         "a priest", "a banker", "a school principal", "an airline pilot",
         "an economist", "a network administrator", "an air traffic controller",
         "an author", "a nuclear plant operator", "a computer scientist",
         "a psychologist", "a pharmacist", "a colonel in the army",
         "a mayor of a city", "a university president", "a dentist",
         "a fire department lieutenant", "a high school teacher", "a policeman",
         "a software developer", "a actor", "a fashion model", "a journalist",
         "a musician in a symphony orchestra", "a psychiatrist",
         "a chemical engineer"]
}


class JobDataInstance(DataInstance):
    def __init__(
            self, job_type: str, job: str) -> None:
        self.job = job
        self.job_type = job_type

        self.tgt_sentences = None
        self.vector = None

        generated_sentences = [
            self._format_template(template) for template in JOB_TEMPLATES]

        if len(generated_sentences) > 40:
            self.en_sentences = random.sample(generated_sentences, 40)
        else:
            self.en_sentences = generated_sentences

    def get_csv(self) -> str:
        return (f"{self.job_type},{self.job}")

    def _format_template(self, template: str) -> str:
        return template.replace("<JOB>", self.job)


def generate_job_data_points() -> List[CountryDataInstance]:
    data_points = []
    for job_type, jobs in JOBS.items():
        for job in jobs:
            datapoint = JobDataInstance(
                job_type=job_type,
                job=job)
            data_points.append(datapoint)
    return data_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sbert_model", type=str)
    parser.add_argument("target_lng", type=str)
    parser.add_argument("job_output", type=str)
    parser.add_argument("country_output", type=str)
    args = parser.parse_args()

    pca, csv_lines = process_templates(
        args.sbert_model, args.target_lng, generate_job_data_points)
    with open(args.job_output, "w") as f_jobs:
        for line in csv_lines:
            print(line, file=f_jobs)

    _, csv_lines = process_templates(
        args.sbert_model, args.target_lng, generate_country_data_points, pca=pca)
    with open(args.country_output, "w") as f_countries:
        for line in csv_lines:
            print(line, file=f_countries)

    logging.info("Done.")


if __name__ == "__main__":
    main()
