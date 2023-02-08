#!/usr/bin/env python3

from typing import Dict, List

import argparse
import logging
import random


from templating import process_templates, DataInstance, COUNTRIES


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


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


TEMPLATES = [
    "<NAME> from <COUNTRY> works as <OCCUPATION>.",
    "<NAME> from <COUNTRY> is <OCCUPATION>.",
    "<NAME>'s job is <OCCUPATION-NO-ART>, and <PRONOUN> is from <COUNTRY>.",
    "<NAME>'s job is <OCCUPATION-NO-ART>, and <PRONOUN> comes from <COUNTRY>.",
    "<NAME> comes from <COUNTRY> and works as <OCCUPATION>.",
    "<NAME> is from <COUNTRY> and works as <OCCUPATION>.",
    "<NAME> is <OCCUPATION> from <COUNTRY>.",
]



class JobDataInstance(DataInstance):
    def __init__(self, names, country_code, country, gender, job_title, job_category):
        self.country_code = country_code
        self.country = country
        self.gender = gender
        self.names = names
        self.job_title = job_title
        self.job_title_no_article = " ".join(job_title.split(" ")[1:])
        self.job_category = job_category
        self.tgt_sentences = None
        self.vector = None

        generated_sentences = [
            self._format_template(name, template)
            for template in TEMPLATES for name in self.names]

        self.en_sentences = random.sample(generated_sentences, 40)

    def get_csv(self):
        return (f"{self.country_code},{self.gender}," +
                f"{self.job_title_no_article},{self.job_category}")

    def _format_template(self, name, template):
            return template.replace(
                "<NAME>", name).replace(
                "<OCCUPATION>", self.job_title).replace(
                "<OCCUPATION-NO-ART>", self.job_title_no_article).replace(
                "<COUNTRY>", self.country).replace(
                "<PRONOUN>", "he" if self.gender == "male" else "she")


def generate_data_points(names: Dict[str, List[str]]) -> List[JobDataInstance]:
    data_points = []
    for job_type, jobs in JOBS.items():
        for job in jobs:
            for gender, gender_names in names.items():
                for code, country in COUNTRIES.items():
                    datapoint = JobDataInstance(
                        names=gender_names[code],
                        language=code,
                        country=country,
                        gender=gender,
                        job_title=job,
                        job_category=job_type)
                    data_points.append(datapoint)
    return data_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sbert_model", type=str)
    parser.add_argument("target_lng", type=str)
    parser.add_argument("male_names", type=argparse.FileType("r"))
    parser.add_argument("female_names", type=argparse.FileType("r"))
    args = parser.parse_args()

    process_templates(
        args.sbert_model, args.target_lng, args.male_names, args.female_names,
        generate_data_points)


if __name__ == "__main__":
    main()
