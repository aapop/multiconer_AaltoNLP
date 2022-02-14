#!/bin/bash
#SBATCH --job-name=koala_dev_preds
#SBATCH --account= ACCOUNT
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
##SBATCH --mail-type=BEGIN,END,FAIL #uncomment to enable mail

module load pytorch

# Set cache to a location with more memory.
export TRANSFORMERS_CACHE=


# KOALA
python3 predict_tags.py --test "data/DE-German/de_dev.conll" --out_dir "predictions/dev/koala" --file_name "de_best_koala" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-de" --model ""

python3 predict_tags.py --test "data/BN-Bangla/bn_dev.conll" --out_dir "predictions/dev/koala" --file_name "bn_best_koala" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-bn" --model ""

python3 predict_tags.py --test "data/FA-Farsi/fa_dev.conll" --out_dir "predictions/dev/koala" --file_name "fa_best_koala" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-fa" --model ""

python3 predict_tags.py --test "data/KO-Korean/ko_dev.conll" --out_dir "predictions/dev/koala" --file_name "ko_best_koala" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-ko" --model ""

python3 predict_tags.py --test "data/EN-English/en_dev.conll" --out_dir "predictions/dev/koala" --file_name "en_best_koala" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-en" --model ""


# BASELINE
python3 predict_tags.py --test "data/DE-German/de_dev.conll" --out_dir "predictions/dev/base" --file_name "de_baseline" --batch_size "30" --max_length "500" --encoder_model "xlm-roberta-large" --model ""

python3 predict_tags.py --test "data/BN-Bangla/bn_dev.conll" --out_dir "predictions/dev/base" --file_name "bn_baseline" --batch_size "30" --max_length "500" --encoder_model "xlm-roberta-large" --model ""

python3 predict_tags.py --test "data/FA-Farsi/fa_dev.conll" --out_dir "predictions/dev/base" --file_name "fa_baseline" --batch_size "30" --max_length "500" --encoder_model "xlm-roberta-large" --model ""

python3 predict_tags.py --test "data/KO-Korean/ko_dev.conll" --out_dir "predictions/dev/base" --file_name "ko_baseline" --batch_size "30" --max_length "500" --encoder_model "xlm-roberta-large" --model ""

python3 predict_tags.py --test "data/EN-English/en_dev.conll" --out_dir "predictions/dev/base" --file_name "en_baseline" --batch_size "30" --max_length "500" --encoder_model "xlm-roberta-large" --model ""


# E2E
python3 predict_ensemble_e2e.py --test "data/DE-German/de_dev.conll" --out_dir "predictions/dev/e2e" --file_name "de_e2e" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-de" --model ""

python3 predict_ensemble_e2e.py --test "data/BN-Bangla/bn_dev.conll" --out_dir "predictions/dev/e2e" --file_name "bn_e2e" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-bn" --model ""

python3 predict_ensemble_e2e.py --test "data/FA-Farsi/fa_dev.conll" --out_dir "predictions/dev/e2e" --file_name "fa_e2e" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-fa" --model ""

python3 predict_ensemble_e2e.py --test "data/KO-Korean/ko_dev.conll" --out_dir "predictions/dev/e2e" --file_name "ko_e2e" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-ko" --model ""

python3 predict_ensemble_e2e.py --test "data/EN-English/en_dev.conll" --out_dir "predictions/dev/e2e" --file_name "en_e2e" --batch_size "30" --max_length "500" --encoder_model "koala/xlm-roberta-large-en" --model ""