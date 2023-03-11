#!/usr/bin/env python3

"""Get frequent names from WikiData."""

import argparse
from collections import Counter
import math
import logging
import os
import pickle
import re
import time

import requests
from unidecode import unidecode
from pyjarowinkler.distance import get_jaro_distance


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


ENDPOINT_URL = 'https://query.wikidata.org/sparql'

QUERY = """prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
select distinct ?item ?itemLabel ?itemDescription ?firstNameLabel ?surnameLabel where {{
 ?item wdt:P31 wd:Q5;   # Any instance of a human.
       {}; # Condition
       wdt:P21 wd:Q{}.  # Sex or gender is ...
 ?item wdt:P569 ?born.
 ?item wdt:P735 ?firstName.
 ?item wdt:P734 ?surname.
 FILTER( ?born >= "1940-01-01T00:00:00"^^xsd:dateTime )
 SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}}
LIMIT 10000
"""

GENDER = {"M": "6581097", "F": "6581072"}

# code -> (language, country)
LANGUAGES = {
    "at": (None, "40"),
    "ba": (None, "225"),
    "be": (None, "31"),
    "bg": ("7918", "219"),
    "by": (None, "184"),
    "ch": (None, "39"),
    "cs": ("9056", "213"),
    "cy": (None, "229"),
    "da": ("9035", "35"),
    "de": ("188", "183"),
    "el": ("9129", "41"),
    "es": ("1321", "29"),
    "et": ("9072", "191"),
    "fi": ("1412", "33"),
    "fr": ("150", "142"),
    "hu": ("9067", "28"),
    "hr": ("6654", "224"),
    "ie": (None, "27"),
    "is": ("294", "189"),
    "it": ("652", "38"),
    "lv": ("9078", "211"),
    "lt": ("9083", "37"),
    "lu": (None, "32"),
    "nl": ("7411", "55"),
    "md": (None, "217"),
    "me": (None, "236"),
    "mk": (None, "221"),
    "mt": (None, "233"),
    "no": ("9043", "20"),
    "pt": ("5146", "45"),
    "pl": ("809", "36"),
    "ro": ("7913", "218"),
    "ru": ("7737", "159"),
    "sk": ("9058", "214"),
    "sl": ("9063", "215"),
    "sq": ("8748", "222"),
    "sr": ("9299", "403"),
    "sv": ("9027", "34"),
    "tr": ("256", "43"),
    "ua": ("8798", "212"),
    "uk": (None, "21"),
    "xk": (None, "1246")
}


CACHE = {}


def get_wiki_data(query):
    if query in CACHE:
        return CACHE[query], True

    # The endpoint defaults to returning XML, so the Accept: header is required
    r = requests.get(
        ENDPOINT_URL,
        params={'query': query},
        headers={'Accept': 'application/sparql-results+json'},
        timeout=1800)
    data = r.json()
    statements = data['results']['bindings']
    CACHE[query] = statements
    time.sleep(60)
    return statements, False


JARO_CACHE = {}


def jaro_distance(str1, str2):
    str1, str2 = unidecode(str1).lower(), unidecode(str2).lower()

    if (str1, str2) in JARO_CACHE:
        return JARO_CACHE[(str1, str2)]
    if (str2, str1) in JARO_CACHE:
        return JARO_CACHE[(str2, str1)]

    dist = get_jaro_distance(str1, str2)
    JARO_CACHE[(str1, str2)] = dist
    return dist


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("gender", choices=["M", "F"])
    parser.add_argument("name", choices=["first", "surname"])
    args = parser.parse_args()

    if os.path.exists("cache.pickle"):
        with open("cache.pickle", "rb") as f_cache:
            CACHE.update(pickle.load(f_cache))

    per_lng_name_freqs = {}

    for lng, (lng_code, country_code) in LANGUAGES.items():
        logging.info("Download %s", lng)

        lng_condition = f"wdt:P103 wd:Q{lng_code}"
        country_condition = f"wdt:P19/wdt:P131* wd:Q{country_code}"

        if lng is None:
            lng_people = []
        else:
            lng_people, lng_from_cache = get_wiki_data(
                QUERY.format(lng_condition, GENDER[args.gender]))

        if lng in ["de", "fr", "es", "hr"]:
            country_people = []
        else:
            country_people, country_from_cache = get_wiki_data(
                QUERY.format(country_condition, GENDER[args.gender]))
        people = {
            x["item"]["value"]: x
            for x in lng_people + country_people}.values()
        name_counts = Counter()
        for person in people:
            if args.name == "first":
                name = person["firstNameLabel"]["value"]
            elif args.name == "surname":
                name = person["surnameLabel"]["value"]
            else:
                raise ValueError("Unkown name type.")

            if re.match("Q[0-9]+", name):
                continue
            if " / " in name:
                name = name.split(" / ")[0]
            if " " in name:
                continue
            name_counts[name] += 1
        per_lng_name_freqs[lng] = name_counts.most_common(50)
        logging.info(
            "Found %d names: %s", len(name_counts), name_counts.most_common(5))

        if lng_from_cache or country_from_cache:
            logging.info("Cache the SPARQL queries.")
            with open("cache.pickle", "wb") as f_cache:
                pickle.dump(CACHE, f_cache)

    logging.info("Compute TF-IDF for names.")
    name_tf_idfs = {}

    for lng, name_counts in per_lng_name_freqs.items():
        logging.info("Proces %s.", lng)
        name_list = []
        for name, in_lng_freq in name_counts:
            all_lng_freq = in_lng_freq
            lngs_with_name = 1
            for lng2, lng2_name_counts in per_lng_name_freqs.items():
                if lng == lng2:
                    continue
                for name2, name2_count in lng2_name_counts:
                    if jaro_distance(name, name2) <= .90:
                        continue
                    lngs_with_name += 1
                    all_lng_freq += name2_count

            tf = in_lng_freq / all_lng_freq
            idf = math.log(len(LANGUAGES) / lngs_with_name)

            name_list.append((name, tf * idf, in_lng_freq))
        name_list.sort(key=lambda x: (-x[1], -x[2]))
        name_tf_idfs[lng] = name_list
        logging.info("Best scoring: %s", name_list[:5])
        print(f"{lng},{','.join([n for n, _, _ in name_list[:20]])}")


if __name__ == "__main__":
    main()
