#!/usr/bin/env python3

import argparse
import logging
import random

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
from transformers import (
    AutoTokenizer, FSMTForConditionalGeneration, MarianMTModel,
    MarianTokenizer)
from tqdm import tqdm


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


COUNTRIES = {
    "at": "Austria",
    "ba": "Bosnia nad Herzegovina",
    "be": "Belgium",
    "bg": "Bulgaria",
    "by": "Belarus",
    "ch": "Switzerland",
    "cs": "the Czech Republic",
    "cy": "Cyprus",
    "da": "Denkmark",
    "de": "Germany",
    "el": "Greece",
    "es": "Spain",
    "et": "Estonia",
    "fi": "Finland",
    "fr": "France",
    "hu": "Hungary",
    "hr": "Croatia",
    "ie": "Ireland",
    "is": "Iceland",
    "it": "Italy",
    "lv": "Latvia",
    "lt": "Lithuana",
    "lu": "Luxemburgh",
    "nl": "Netherlands",
    "md": "Moldova",
    "me": "Montenegro",
    "mk": "North Macedonia",
    "mt": "Malta",
    "no": "Norway",
    "pt": "Portugal",
    "pl": "Poland",
    "ro": "Romania",
    "ru": "Russia",
    "sk": "Slovakia",
    "sl": "Slovenia",
    "sq": "Albania",
    "sr": "Serbia",
    "sv": "Sweden",
    "tr": "Turkey",
    "ua": "Ukraine",
    "uk": "Great Britain",
    "xk": "Kosovo"
}


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


TRANSLATORS = {
    "bg": "Helsinki-NLP/opus-mt-tc-big-en-bg",
    "cs": "Helsinki-NLP/opus-mt-tc-big-en-ces_slk",
    "de": "facebook/wmt19-en-de",
    "el": "Helsinki-NLP/opus-mt-tc-big-en-el",
    "en": None,
    "es": "Helsinki-NLP/opus-mt-tc-big-en-es",
    "fi": "Helsinki-NLP/opus-mt-tc-big-en-fi",
    "fr": "Helsinki-NLP/opus-mt-tc-big-en-fr",
    "hu": "Helsinki-NLP/opus-mt-tc-big-en-hu",
    "it": "Helsinki-NLP/opus-mt-tc-big-en-it",
    "pt": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "ro": "Helsinki-NLP/opus-mt-tc-big-en-ro",
    "ru": "facebook/wmt19-en-ru"
}


class Translator:
    def __init__(self, model_name, max_batch_tokens=10000):
        self.model_name = model_name
        self.max_batch_tokens = max_batch_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_name.startswith("facebook"):
            self.model = FSMTForConditionalGeneration.from_pretrained(model_name)
        elif model_name.startswith("Helsinki") or model_name.startswith("gsarti"):
            self.model = MarianMTModel.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def __call__(self, sentences):
        results = []
        batch = []

        def translate_batch():
            max_len = max(len(sent) for sent in batch)
            for sent in batch:
                for _ in range(max_len - len(sent)):
                    sent.append(self.tokenizer.pad_token)
            id_batch = [self.tokenizer.convert_tokens_to_ids(sent) for sent in batch]
            input_ids = torch.tensor(id_batch).to(self.device)
            outputs = self.model.generate(input_ids)
            for sent_out in outputs:
                decoded = self.tokenizer.decode(sent_out, skip_special_tokens=True)
                results.append(decoded)

        for sent in sentences:
            if self.model_name == "Helsinki-NLP/opus-mt-tc-big-en-ces_slk":
                sent = ">>>ces<<< " + sent
            tokens = self.tokenizer.tokenize(sent)
            tokens.append("</s>")
            batch.append(tokens)

            if len(batch) * max(len(sent) for sent in batch) > self.max_batch_tokens:
                translate_batch()
                batch = []

        if batch:
            translate_batch()

        return results


class DataInstance:
    def __init__(self, names, language, country, gender, job_title, job_category):
        self.language = language
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

        self.en_sentences = random.sample(generated_sentences, 20)

    def get_csv(self):
        return (f"{self.language},{self.gender}," +
                f"{self.job_title_no_article},{self.job_category}")

    def _format_template(self, name, template):
            return template.replace(
                "<NAME>", name).replace(
                "<OCCUPATION>", self.job_title).replace(
                "<OCCUPATION-NO-ART>", self.job_title_no_article).replace(
                "<COUNTRY>", self.country).replace(
                "<PRONOUN>", "he" if self.gender == "male" else "she")

    @torch.no_grad()
    def translate(self, translator):
        if translator is None:
            self.tgt_sentences = self.en_sentences
        else:
            self.tgt_sentences = translator(self.en_sentences)

    @torch.no_grad()
    def get_avg_vector(self, sbert_model):
        self.vector = sbert_model.encode(
            self.tgt_sentences, show_progress_bar=False).mean(0)


def load_names(handle):
    names = {}
    for line in handle:
        tokens = line.strip().split(",")
        names[tokens[0]] = tokens[1:]
    handle.close()
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sbert_model", type=str)
    parser.add_argument("target_lng", type=str)
    parser.add_argument("male_names", type=argparse.FileType("r"))
    parser.add_argument("female_names", type=argparse.FileType("r"))
    args = parser.parse_args()

    if not args.target_lng in TRANSLATORS:
        raise ValueError(f"No translator for language '{args.target_lng}'.")

    logging.info("Load typical male names from '%s'.", args.male_names)
    names = {
        "male": load_names(args.male_names),
        "female": load_names(args.female_names)
    }

    logging.info("Generate templated English sentences.")
    data_points = []
    for job_type, jobs in JOBS.items():
        for job in jobs:
            for gender, gender_names in names.items():
                for code, country in COUNTRIES.items():
                    datapoint = DataInstance(
                        names=gender_names[code],
                        language=code,
                        country=country,
                        gender=gender,
                        job_title=job,
                        job_category=job_type)
                    data_points.append(datapoint)

    logging.info("Load Translation model '%s'.", TRANSLATORS[args.target_lng])
    mt_model = None
    if args.target_lng != "en":
        mt_model = Translator(TRANSLATORS[args.target_lng])
    logging.info("Translate sentences.")
    for item in tqdm(data_points):
        item.translate(mt_model)
    logging.info("Translated. Delete the MT model.")

    logging.info("Load SBERT model '%s'.", args.sbert_model)
    sbert = SentenceTransformer(args.sbert_model)
    logging.info("Extract SBERT embeddings.")
    for item in tqdm(data_points):
        item.get_avg_vector(sbert)

    logging.info("Do PCA.")
    embeddings = np.stack([item.vector for item in data_points])
    pca_res = PCA(n_components=10).fit_transform(embeddings)

    logging.info("Save as a dataframe for futher analysis.")
    for item, vector in zip(data_points, pca_res):
        print(item.get_csv() + "," + ",".join(map(str, vector)))

    logging.info("Done.")


if __name__ == "__main__":
    main()
