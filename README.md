# European Nations in Sentence Embeddings

## Typical names

Extraction of typical names for countries: based on people in
Wikipedia/Wikidata which were born in the country after 1940. For each of the
country, 10 typical names are selected based on the tf-idf score (where a
country corresponds to what normally is a document). Names are considered equal
if they have Jaro distance higher than 0.9.

```bash
python3 names_wikidata.py M > wikidata.male.txt
python3 names_wikidata.py F > wikidata.female.txt
``````

## Extract and analyze sentence embeddings

```bash
mkdir csvs
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers/LaBSE; do for LNG in bg cs de el en es fi fr hu it pt ro ru; do python3 generate_from_templates.py ${MODEL} ${LNG} wikidata.male.txt wikidata.female.txt > csvs/${MODEL/\//-}.${LNG}.csv; done; done
```

## Collect correlation of country attributes, gender and professions

```bash
mkdir correlation-logs correlation-jsons
for F in csvs/*.csv; do  python3 label_correlation.py $F correlation-jsons/$(basename $F | sed -e 's/\.csv/.json/') > correlation-logs/$(basename $F | sed -e 's/\.csv/.log/') ; done
```

Get a CSV report

python3 report.py correlation-jsons/sentence-transformers-LaBSE.{}.json
python3 report.py correlation-jsons/paraphrase-multilingual-mpnet-base-v2.{}.json
python3 report.py correlation-jsons/distiluse-base-multilingual-cased-v2.{}.json
