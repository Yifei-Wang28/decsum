# Decision-Focused Summarization

Partial replication (failthfulness) of Decision-Focused Summarization [paper link](https://arxiv.org/abs/2109.06896).

## Env
Create env with conda:
```
conda create -n yelp python=3.7.6
```
Then install packages with:

```
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 python -m pip install
# download spacy package
python -m spacy download en_core_web_sm

# If you are using RTX3090, try the following step to install pytorch
pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data Preprocessing
Here, we only select reviews from restaurants to build our dataset.

Download raw dataset from https://www.yelp.com/dataset/download and uncompress it to `YELP_DATA_DIR`. 
Then, run scrip at the base directory.
```
python -m preprocess.yelp_preprocess [--yelp_data_dir YELP_DATA_DIR] [--output_dir OUTPUT_DIR]
```

## Train Longformer model
>Remeber to change env variables in `scripts/train_transformer.sh` before running the training script. It takes about three hours to train longformer on RTX3090 with half precision.
```
bash scripts/train_transformer.sh
```
You can check training log here `${OUTPUT_DIR}/logs/` with `tensorboard`.
Trained model will be saved to path like this `${OUTPUT_DIR}/version_27-12-2021--16-59-15/checkpoints/epoch=1-val_loss=0.12.ckpt`.

## Run DecSum
> Change env variables in `scripts/sentence_select.sh`  before running DecSum. This step takes about 10 hours on RTX3090.

```
# at base Directory
bash scripts/sentence_selection.sh
```
The DecSum summaries will be saved at `${RES_DIR}/models/sentence_select/selected_sentence/yelp/50reviews/test/Transformer/window_1_DecSum_WD_sentbert_50trunc_1_1_1/best/1/text_.csv`.

*_MSE with True Label_* metric will be store at `${RES_DIR}/models/sentence_select/results/yelp/50reviews/test/Transformer/window_1_DecSum_WD_sentbert_50trunc_1_1_1/best/1/text_.csv`.

## Baseline methods
> Change env variables in `scripts/sentence_select_rand.sh`  before running Random Selection. This step takes about 1.5 hours on RTX3090.

```
# at base Directory
bash scripts/sentence_selection_rand.sh
```

# Citation
```
@inproceedings{hsu-tan-2021-decision,
    title = "Decision-Focused Summarization",
    author = "Hsu, Chao-Chun  and
      Tan, Chenhao",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.10",
    doi = "10.18653/v1/2021.emnlp-main.10",
    pages = "117--132",
}
```