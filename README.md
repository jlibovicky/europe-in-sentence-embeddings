# Is a Prestigious Job the same as a Prestigious Country? A Case Study on Multilingual Sentence Embeddings and European Countries

## Extracting the domninant dimesions sentence embeddings with templates

The dimension extraction proceeds as follows (inspired by the Moral Dimension
framework by [Schramowski et al.](https://arxiv.org/abs/2103.11790)).

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
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers/LaBSE xlmr_nliv2_5-langs; do
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do
        python3 generate_country_only_templates.py ${MODEL} ${LNG} country-only-csvs/${MODEL/\//-}.${LNG}.csv;
    done
done
```

### Experiment 2: Country prestige PCA

Unlike the previous experiment, here the templated sentences contain statements
that being from a specific country is considered prestigeous.

```bash
mkdir country-prestige-csvs
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers/LaBSE xlmr_nliv2_5-langs; do
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
corresponding to countries and evaluate how they get ordered.

```bash
mkdir job-country-prestige-csvs
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers/LaBSE xlmr_nliv2_5-langs; do
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do
        python3 generate_job_prestige_templates_and_apply_to_countries.py ${MODEL} ${LNG} job-country-prestige-csvs/${MODEL/\//-}.${LNG}.{job,country}.csv
    done
done
```

## Evaluation

### Interpret the PCA dimensions

Evaluate with what country groups the first PCA dim correlates.

```bash
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers-LaBSE xlmr_nliv2_5-langs; do
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do echo -en "${LNG},"; python3 label_eval.py country-only-csvs/${MODEL}.${LNG}.csv 2> /dev/null; done
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do echo -en "${LNG},"; python3 label_eval.py country-prestige-csvs/${MODEL}.${LNG}.csv 2> /dev/null; done
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do echo -en "${LNG},"; python3 label_eval.py job-country-prestige-csvs/${MODEL}.${LNG}.country.csv 2> /dev/null; done
done
```

Evalute correlation of first PCA dim with country GDP.

```bash
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers-LaBSE xlmr_nliv2_5-langs; do
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do echo -en "${LNG},"; python3 country_gdp_eval.py country-only-csvs/${MODEL}.${LNG}.csv 2> /dev/null; done
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do echo -en "${LNG},"; python3 country_gdp_eval.py country-prestige-csvs/${MODEL}.${LNG}.csv 2> /dev/null; done
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do echo -en "${LNG},"; python3 country_gdp_eval.py job-country-prestige-csvs/${MODEL}.${LNG}.country.csv 2> /dev/null; done
done
```

Evalute how well the extracted dimension separates low- and high- profile jobs.

```bash
for MODEL in paraphrase-multilingual-mpnet-base-v2 distiluse-base-multilingual-cased-v2 sentence-transformers-LaBSE xlmr_nliv2_5-langs; do
    for LNG in bg cs de el en es fi fr hu it pt ro ru; do
        python3 eval_job_separation.py job-country-prestige-csvs/${MODEL}.${LNG}.job.csv 2> /dev/null
    done
    echo
done
```

### Evaluate cross-lingal similarity

Compute how the cross-language correlation depends on country distance, GDP
difference and lexical similarity of the languages.

```bash
python3 compare_languages.py \
    job-country-prestige-csvs/paraphrase-multilingual-mpnet-base-v2.{}.country.csv \
    job-country-prestige-csvs/distiluse-base-multilingual-cased-v2.{}.country.csv \
    job-country-prestige-csvs/sentence-transformers-LaBSE.{}.job.csv \
    job-country-prestige-csvs/xlmr_nliv2_5-langs.{}.job.csv \
    --is-job --labels "Mul. Par. MPNet" "Dist. mUSE" "LaBSE" "XLM-R-NLI"
```

## Citation

```bibtex
@misc{libovicky2023prestigious,
  title={Is a Prestigious Job the same as a Prestigious Country? A Case Study on Multilingual Sentence Embeddings and European Countries},
  author={Libovick{\'y}, Jind{\v{r}}ich},
  journal={arXiv preprint arXiv:2305.14482},
  year={2023}
}
```
