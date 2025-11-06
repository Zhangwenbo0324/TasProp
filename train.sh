# ================== Configuration ==================
BASE_DIR="your_experiment_dir/TasProp" # use your desired default path
EXPERIMENTS=("anest" "bbbp" "excit" "sider" "SR")

VOCAB_PATH="${BASE_DIR}/data/vocab_all.txt"
PRETRAINED_MODEL="${BASE_DIR}/model/vae_model_zinc/model.iter-4"

AUG_SCRIPT="${BASE_DIR}/molvae/augment.py"
TRAIN_SCRIPT="${BASE_DIR}/molvae/p_vae_train.py"

export CUDA_VISIBLE_DEVICES=3

# ================== Data Augmentation Stage ==================
echo "=== Starting Data Augmentation Stage ==="

for exp in "${EXPERIMENTS[@]}"; do
        mkdir -p "${BASE_DIR}/data/gen"

        echo "Augmenting task: ${exp}"
        python "${AUG_SCRIPT}" \
            --task "${exp}" \
            --delta 3 \
            --write_path "${BASE_DIR}/data/gen/${exp}_gen.csv" \
            --vocab "${VOCAB_PATH}" \
            --model "${PRETRAINED_MODEL}" \
            --train_path "${BASE_DIR}/data/${exp}_train.csv"
done

# ================== Model Training Stage ==================
echo "=== Starting Model Training Stage ==="
    for exp in "${EXPERIMENTS[@]}"; do
        model_dir="${BASE_DIR}/model/${exp}"
        mkdir -p "${model_dir}/best_model"
        touch "${model_dir}/errors.txt"
        touch "${model_dir}/log.txt"

        echo "Training model for task: ${exp}"
        python "${TRAIN_SCRIPT}" \
            --task "${exp}" \
            --aug 1 \
            --vocab "${VOCAB_PATH}" \
            --hidden 450 \
            --depth 3 \
            --latent 56 \
            --batch 40 \
            --lr 0.0007 \
            --alpha 1.0 \
            --beta 0.008 \
            --model "${PRETRAINED_MODEL}" \
            --origin_train_path "${BASE_DIR}/data/${exp}_train.csv" \
            --gen_path "${BASE_DIR}/data/gen/${exp}_gen.csv" \
            --test_path "${BASE_DIR}/data/${exp}_test.csv" \
            --save_dir "${model_dir}"
    done

echo "=== All jobs completed ==="
