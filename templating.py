from typing import Dict, Callable, List, TextIO

from abc import ABC, abstractmethod
import logging
import pickle
import os

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
    "ba": "Bosnia and Herzegovina",
    "be": "Belgium",
    "bg": "Bulgaria",
    "by": "Belarus",
    "ch": "Switzerland",
    "cs": "the Czech Republic",
    "cy": "Cyprus",
    "da": "Denmark",
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
    "lt": "Lithuania",
    "lu": "Luxembourg",
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
    "ru": "facebook/wmt19-en-ru",
}


# Taken from http://ukc.disi.unitn.it/index.php/lexsim/
LANGUAGE_SIMILARITES = {
    'bg': {"cs": 5.885, "de": 2.711, "el": 1.271, "en": 2.182, "es": 2.711, "fi": 1.133, "fr": 2.74, "hu": 1.975, "it": 3.05, "pt": 3.324, "ro": 3.138, "ru": 8.461},
    'cs': {"bg": 5.885, "de": 3.733, "el": 1.171, "en": 2.439, "es": 2.639, "fi": 1.341, "fr": 2.749, "hu": 2.869, "it": 3.113, "pt": 3.315, "ro": 2.777, "ru": 6.17},
    'de': {"bg": 2.711, "cs": 3.733, "el": 1.395, "en": 4.72, "es": 3.578, "fi": 1.793, "fr": 4.162, "hu": 2.986, "it": 4.186, "pt": 4.616, "ro": 3.473, "ru": 2.961},
    'el': {"bg": 1.271, "cs": 1.171, "de": 1.395, "en": 1.22, "es": 1.51, "fi": 0.686, "fr": 1.551, "hu": 1.173, "it": 1.923, "pt": 1.882, "ro": 1.411, "ru": 0.937},
    'en': {"bg": 2.182, "cs": 2.439, "de": 4.72, "el": 1.22, "es": 7.907, "fi": 1.734, "fr": 9.665, "hu": 1.844, "it": 6.757, "pt": 7.944, "ro": 6.194, "ru": 2.025},
    'es': {"bg": 2.711, "cs": 2.639, "de": 3.578, "el": 1.51, "en": 7.907, "fi": 1.484, "fr": 11.196, "hu": 2.105, "it": 10.451, "pt": 17.414, "ro": 8.428, "ru": 2.252},
    'fi': {"bg": 1.133, "cs": 1.341, "de": 1.793, "el": 0.686, "en": 1.734, "es": 1.484, "fr": 1.622, "hu": 1.581, "it": 1.825, "pt": 1.785, "ro": 1.511, "ru": 1.02},
    'fr': {"bg": 2.74, "cs": 2.749, "de": 4.162, "el": 1.551, "en": 9.665, "es": 11.196, "fi": 1.622, "hu": 2.086, "it": 9.544, "pt": 11.979, "ro": 8.817, "ru": 2.435},
    'hu': {"bg": 1.975, "cs": 2.869, "de": 2.986, "el": 1.173, "en": 1.844, "es": 2.105, "fi": 1.581, "fr": 2.086, "it": 2.633, "pt": 2.601, "ro": 2.133, "ru": 1.839},
    'it': {"bg": 3.05, "cs": 3.113, "de": 4.186, "el": 1.923, "en": 6.757, "es": 10.451, "fi": 1.825, "fr": 9.544, "hu": 2.633, "pt": 12.172, "ro": 7.824, "ru": 2.816},
    'pt': {"bg": 3.324, "cs": 3.315, "de": 4.616, "el": 1.882, "en": 7.944, "es": 17.414, "fi": 1.785, "fr": 11.979, "hu": 2.601, "it": 12.172, "ro": 9.065, "ru": 2.92},
    'ro': {"bg": 3.138, "cs": 2.777, "de": 3.473, "el": 1.411, "en": 6.194, "es": 8.428, "fi": 1.511, "fr": 8.817, "hu": 2.133, "it": 7.824, "pt": 9.065, "ru": 2.509},
    'ru': {"bg": 8.461, "cs": 6.17, "de": 2.961, "el": 0.937, "en": 2.025, "es": 2.252, "fi": 1.02, "fr": 2.435, "hu": 1.839, "it": 2.816, "pt": 2.92, "ro": 2.509}
}


def load_translate_cache():
    if not os.path.exists("translate_cache.pickle"):
        return {}

    with open("translate_cache.pickle", "rb") as f_pickle:
        return pickle.load(f_pickle)


def save_translate_cache():
    with open("translate_cache.pickle", "wb") as f_pickle:
        return pickle.dump(TRANSLATE_CACHE, f_pickle)


TRANSLATE_CACHE = load_translate_cache()


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

    @torch.no_grad()
    def __call__(self, sentences):
        results = []
        batch = []

        uncached_sources = []
        uncached_targets = []
        for sent in sentences:
            if (self.model_name, sent) in TRANSLATE_CACHE:
                results.append(TRANSLATE_CACHE[(self.model_name, sent)])
            else:
                results.append(None)
                uncached_sources.append(sent)

        def translate_batch():
            max_len = max(len(sent) for sent in batch)
            for sent in batch:
                for _ in range(max_len - len(sent)):
                    sent.append(self.tokenizer.pad_token)
            id_batch = [
                self.tokenizer.convert_tokens_to_ids(sent) for sent in batch]
            input_ids = torch.tensor(id_batch).to(self.device)
            outputs = self.model.generate(input_ids)
            for sent_out in outputs:
                decoded = self.tokenizer.decode(
                    sent_out, skip_special_tokens=True)
                uncached_targets.append(decoded)

        for sent in uncached_sources:
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

        for i, tgt_sent in enumerate(results):
            if tgt_sent is None:
                results[i] = uncached_targets.pop(0)

        return results


class DataInstance(ABC):
    @abstractmethod
    def get_csv(self):
        pass

    @abstractmethod
    def _format_template(self, name: str, template: str):
        pass

    @torch.no_grad()
    def translate(self, translator: Translator) -> None:
        if translator is None:
            self.tgt_sentences = self.en_sentences
        else:
            self.tgt_sentences = translator(self.en_sentences)

    @torch.no_grad()
    def get_avg_vector(self, sbert_model: SentenceTransformer) -> None:
        self.vector = sbert_model.encode(
            self.tgt_sentences, show_progress_bar=False).mean(0)


DataPoinGenerator = Callable[[Dict[str, List[str]]], List[DataInstance]]


def load_names(handle: TextIO) -> List[str]:
    names = {}
    for line in handle:
        tokens = line.strip().split(",")
        names[tokens[0]] = tokens[1:]
    handle.close()
    return names


def extract_vectors(
        sbert_model: str,
        target_lng: str,
        data_point_generator: DataPoinGenerator) -> List[DataInstance]:
    logging.info("Generate templated English sentences.")
    data_points = data_point_generator()

    logging.info("Load Translation model '%s'.", TRANSLATORS[target_lng])
    mt_model = None
    if target_lng != "en":
        mt_model = Translator(TRANSLATORS[target_lng])
    logging.info("Translate sentences.")
    for item in tqdm(data_points):
        item.translate(mt_model)

    logging.info("Load SBERT model '%s'.", sbert_model)
    sbert = SentenceTransformer(sbert_model)
    logging.info("Extract SBERT embeddings.")
    for item in tqdm(data_points):
        item.get_avg_vector(sbert)

    return data_points


def process_templates(
        sbert_model: str,
        target_lng: str,
        data_point_generator: DataPoinGenerator,
        pca: PCA = None) -> None:

    if not target_lng in TRANSLATORS:
        raise ValueError(f"No translator for language '{target_lng}'.")

    data_points = extract_vectors(
        sbert_model, target_lng, data_point_generator)

    logging.info("Do PCA.")
    embeddings = np.stack([item.vector for item in data_points])
    if pca is None:
        logging.info("Fitting PCA.")
        pca = PCA(n_components=.8).fit(embeddings)
    pca_res = pca.transform(embeddings)

    logging.info("Generate CSV for furhter analysis.")
    csv_lines = []
    for item, vector in zip(data_points, pca_res):
        csv_lines.append(item.get_csv() + "," + ",".join(map(str, vector)))

    return pca, csv_lines
