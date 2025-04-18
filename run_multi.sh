#!/usr/bin/env bash

# 複数のパラメータを組み合わせて実行する例。
# GPU が専有されている(= nvidia-smi により GPU を使用中のプロセスが検出される)場合は
# 一定時間スリープして、GPU が空くのを待ってから次のジョブを実行します。
#
# 実行前:
#   chmod +x run_multi_gpu_wait.sh     # 実行権限を付与
#   ./run_multi_gpu_wait.sh           # または bash run_multi_gpu_wait.sh で実行

# 1. GPUチェック関数
check_gpu_free() {
  # nvidia-smiコマンドで "C" という文字が含まれる行を grep し、
  # そこにプロセス(ユーザープロセス)が見つかったかどうかを確認する簡易的な例。
  # 行例: "  0    6862      C   python3 ... " など
  # 何らかのプロセスが存在したら return 1 (GPU使用中)、無ければ return 0 (空き)
  local gpuprocs
  gpuprocs="$(nvidia-smi | grep " C ")"

  if [ -z "$gpuprocs" ]; then
    # grepの結果が空 => プロセスなし => GPUは空いている
    return 0
  else
    # 何らかのプロセスが検出された => GPU使用中
    return 1
  fi
}

# 2. パラメータ設定
BASE_SAVE_DIR="./data/illumination_multi"   # 結果を保存するベースディレクトリ
mkdir -p "${BASE_SAVE_DIR}"                # 必要に応じて作成

SEEDS=("0" "1" "2")
SIGMAS=("0.05" "0.1" "0.2")
K_NBRS=("1" "2")

# 3. ループで実験を回す
for seed in "${SEEDS[@]}"; do
  for sigma in "${SIGMAS[@]}"; do
    for k in "${K_NBRS[@]}"; do

      # 実験結果の保存先ディレクトリをユニークにする
      SAVE_DIR="${BASE_SAVE_DIR}/seed${seed}_sigma${sigma}_k${k}"
      mkdir -p "${SAVE_DIR}"

      echo "-----------------------------------------------------------"
      echo "Trying seed=${seed}, sigma=${sigma}, k_nbrs=${k}"
      echo "Save directory: ${SAVE_DIR}"
      echo "Checking GPU availability..."
      echo "-----------------------------------------------------------"

      # GPU使用中の場合は待機 (30秒ごとに再チェックする例)
      while true; do
        if check_gpu_free; then
          echo "GPU is free! Starting the job..."
          break
        else
          echo "GPU busy. Waiting 30 seconds..."
          sleep 30
        fi
      done

      # 実験コマンド
      python main_illuminate.py \
        --seed="${seed}" \
        --save_dir="${SAVE_DIR}" \
        --substrate="plenia" \
        --n_child=32 \
        --pop_size=256 \
        --n_iters=1000 \
        --sigma="${sigma}" \
        --k_nbrs="${k}"

      echo "Done. Results saved in ${SAVE_DIR}"
      echo
    done
  done
done
