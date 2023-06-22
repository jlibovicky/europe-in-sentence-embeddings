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
