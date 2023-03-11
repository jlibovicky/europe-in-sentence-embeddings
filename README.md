# European Nations in Sentence Embeddings

## What is in the representation

This is based on three experiments losely based on the Moral Dimensions
Framework.

1. Take just statements about the country and check first PCA dim
2. Take statements about country prestige and check first PCA dim
3. Take statements about job prestige and apply to the country

Experiment (3) shows that the model can distinguish more and less prestigious
jobs, so it means the models know what prestige is. It makes us believe that
experiment (2) gives us relieable information about country prestige encoded in
the models. By comparing results of (1) and (2), we show that the assumed
country prestige is the primary dimension in the representations in general.

### Experiment 1: Country-only PCA

The templated text in this experiment only states that something is someones
country of origin.

```bash
mkdir country-only-csvs
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers/LaBSE; do
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do
        python3 generate_country_only_templates.py ${MODEL} ${LNG} > country-only-csvs/${MODEL/\//-}.${LNG}.csv;
    done
done
```

### Experiment 2: Country prestige PCA

Unlike the previous experiment, here the templated sentences contain statements
that being from a specific country is considered prestigeous.

```bash
mkdir country-prestige-csvs
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers/LaBSE; do
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do
        python3 generate_country_prestige_templates.py ${MODEL} ${LNG} country-prestige-csvs/${MODEL/\//-}.${LNG}.csv;
    done
done
```

### Experiment 3: Job prestige PCA

In this experiment, the templated texts used for PCA only contain statements
about job prestige. Moreover, we specifically select jobs to have particularly
low or high prestige. The PCA is done on vectors originating from these
sentences. In a second step, we apply the same projection on vectors
corresponding

```bash
mkdir job-country-prestige-csvs
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers/LaBSE; do
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do
        python3 generate_job_prestige_templates_and_apply_to_countries.py ${MODEL} ${LNG} job-country-prestige-csvs/${MODEL/\//-}.${LNG}.{job,country}.csv
    done
done
```

### Evaluating the experiments


## From country knowledge to prestige

### Typical names

Extraction of typical names for countries: based on people in
Wikipedia/Wikidata which were born in the country after 1940. For each of the
country, 10 typical names are selected based on the tf-idf score (where a
country corresponds to what normally is a document). Names are considered equal
if they have Jaro distance higher than 0.9.

```bash
for GENDER in M F; do
    for NAME in first surname; do
        python3 names_wikidata.py M first > wikidata.${GENDER}.${NAME}.txt
    done
done
``````

## Extract and analyze sentence embeddings

```bash
mkdir country-csvs
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers/LaBSE; do
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do
        python3 generate_country_only_from_templates.py ${MODEL} ${LNG} wikidata.male.txt wikidata.female.txt > country-csvs3/${MODEL/\//-}.${LNG}.csv
    done
done

mkdir jobs-csv
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers/LaBSE; do for LNG in bg cs de el en es fi fr hu it pt ro ru; do echo $MODEL --- ${LNG}; python3 generate_country_prestige_templates.py ${MODEL} ${LNG} job-csvs/${MODEL/\//-}.${LNG}.{job,country}.csv; done; done
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

## Experiments with nationality classifier

```
python3 name_classifier.py wikidata.male.txt wikidata.male.surnames.txt wikidata.female.txt wikidata.female.surnames.txt paraphrase-multilingual-mpnet-base-v2 country-classifier-paraphrase.pkl
python3 name_classifier.py wikidata.male.txt wikidata.male.surnames.txt wikidata.female.txt wikidata.female.surnames.txt distiluse-base-multilingual-cased-v2 country-classifier-distiluse.pkl
python3 name_classifier.py wikidata.male.txt wikidata.male.surnames.txt wikidata.female.txt wikidata.female.surnames.txt sentence-transformers/LaBSE country-classifier-labse.pkl
```
