from typing import Dict, Callable, List, TextIO

from abc import ABC, abstractmethod
import logging
import pickle
import random
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
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
            id_batch = [self.tokenizer.convert_tokens_to_ids(sent) for sent in batch]
            input_ids = torch.tensor(id_batch).to(self.device)
            outputs = self.model.generate(input_ids)
            for sent_out in outputs:
                decoded = self.tokenizer.decode(sent_out, skip_special_tokens=True)
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


def load_names(handle: TextIO) -> List[str]:
    names = {}
    for line in handle:
        tokens = line.strip().split(",")
        names[tokens[0]] = tokens[1:]
    handle.close()
    return names



def process_templates(
        sbert_model: str,
        target_lng: str,
        data_point_generator: Callable[[Dict[str, List[str]]], List[DataInstance]],
        pca: PCA = None) -> None:

    if not target_lng in TRANSLATORS:
        raise ValueError(f"No translator for language '{target_lng}'.")

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

