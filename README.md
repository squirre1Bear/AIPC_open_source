# AIPC Peptide Rescoring Pipeline

当前模型整体流程如下：

```text
原始谱图搜索结果
  -> 统一 parquet 预处理
  -> 手工 PSM / peptide / fragment / group 特征
  -> 按仪器训练 LightGBM v1，生成 lgbm_v1_score
  -> top1 consensus peptide 重排
  -> sequence n-gram target/decoy 模式分数
  -> global peptide support 分数
  -> clean-peptide prior 分数
  -> 生成 all_pred.zip
```

最终重打分分数为：

```text
final_score =
  top1_consensus_score
  + 4.5 * sequence_ngram_score
  + 4 * global_peptide_support_score
  + 4 * clean_peptide_prior_score
```

## 目录结构

```text
/root/aipc                                      项目目录
/root/autodl-tmp/datasets/aipc                  数据目录
/root/autodl-tmp/datasets/aipc/processed        预处理后数据
/root/autodl-tmp/datasets/aipc/processed_split  train/valid 划分
/root/aipc/models                               模型目录
/root/aipc/eval                                 验证与中间结果
/root/aipc/submissions                          提交结果
```

## 1. 数据预处理

### 1.1 处理训练数据

不同仪器的数据先被转成统一的 parquet 格式：

```bash
cd /root/aipc

python src/preprocess/process_mzml.py
python src/preprocess/process_wiff.py
python src/preprocess/process_tims.py
```

这些脚本负责读取原始搜索结果，统一字段命名，保留 PSM、peptide、precursor、scan、score、label、instrument 等后续训练需要的列。

### 1.2 处理 Basic 测试集

```bash
python src/preprocess/process_bas_data.py
```

输出目录：

```text
/root/autodl-tmp/datasets/aipc/processed/bas_merged
```

该目录中的每个 parquet 对应一个 Basic 测试文件。后续 LightGBM 预测和最终提交都基于这个目录。

### 1.3 添加手工特征

```bash
AIPC_WORKERS=8 python src/preprocess/run_feature_1+2.py
```

主要特征类型：

- peptide / precursor 组成特征：序列长度、氨基酸组成、修饰数量、修饰类型等。
- 质量误差特征：理论质量、实验质量、ppm 误差、绝对误差。
- scan / RT / rank 特征：谱图内候选排序、文件内统计、保留时间相关特征。
- fragment ion 匹配特征：10/20/50 ppm 容差下 b/y 离子匹配数量、匹配比例、匹配强度统计。
- group 内竞争特征：同一 scan 或同一候选组内的相对排名、分数差、top 候选间隔。

以上特征来自谱图搜索结果、PSM 字段、peptide 字符串和文件内部统计，用于增强模型信息提取能力。

### 1.4 划分 train/valid

```bash
python src/preprocess/split_train_valid.py \
  --data-root /root/autodl-tmp/datasets/aipc \
  --valid-ratio 0.1 \
  --seed 42
```

输出：

```text
/root/autodl-tmp/datasets/aipc/processed_split/train
/root/autodl-tmp/datasets/aipc/processed_split/valid
```

`processed_split/train` 用于训练 sequence n-gram 模型、构建 clean-peptide prior；`processed_split/valid` 用于本地 offline leaderboard 验证。

## 2. LightGBM v1 基础 PSM 模型

后续 rescoring 的基础分数列是：

```text
lgbm_v1_score
```

该分数由按仪器训练的 LightGBM v1 产生。由于不同仪器的分数尺度、fragment 特征可用性和噪声不同，按仪器分开训练可以增强模型性能，因此按 `mzml / tims / wiff` 分开训练三个仪器的模型。

训练命令：

```bash
cd /root/aipc

for inst in mzml tims wiff; do
  python src/model/train_lightGBM_v1_folds_withoutRT.py \
    --data-root /root/autodl-tmp/datasets/aipc \
    --out-dir /root/aipc/models/exp_by_instrument/v1/$inst \
    --only-instrument $inst \
    --n-folds 5 \
    --balance-by-rows \
    --train-max-rows 3000000 \
    --valid-max-rows 1000000 \
    --max-rows-per-file 80000 \
    --neg-pos-ratio 2.0 \
    --num-threads 8 \
    --seed 20260519
done
```

参数说明：

- `--only-instrument`：每次只训练一个仪器的数据，降低仪器分布差异对模型的影响。
- `--n-folds 5`：训练 5 折模型，用于 OOF 验证和 fold ensemble 预测。
- `--balance-by-rows`：按文件行数做采样平衡，避免超大文件主导训练。
- `--train-max-rows` / `--valid-max-rows`：限制训练和验证采样规模，控制内存和训练时间。
- `--max-rows-per-file`：限制单文件贡献的最大行数，避免个别文件过拟合。
- `--neg-pos-ratio 2.0`：控制 decoy/target 采样比例，缓解类别不平衡。
- `--num-threads 8`：LightGBM 使用 8 线程。

训练完成后，为 Basic 测试集写入 `lgbm_v1_score`：

```bash
python src/preprocess/add_group_feature_test.py \
  --parquet-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
  --fold-model-dir /root/aipc/models/exp_by_instrument/v1 \
  --force \
  --mask-rt-qvalue-anomaly
```

这一步会用各仪器的 fold ensemble 对测试 parquet 预测，并把 `lgbm_v1_score` 写回 parquet。

## 3. 主体 Rescoring 模型

主体模型位于：

```text
src/eval/rescore_sequence_support_grid.py
```

当前 benchmark 参数：

```text
alpha = 12
beta = 0.5
seq_weight = 4.5
support_weight = 4.0
score_transform = logit
```

最终主体分数为：

```text
seq_support_score =
  top1_consensus_score
  + seq_weight * sequence_ngram_score
  + support_weight * global_peptide_support_score
```

### 3.1 top1 consensus score

输入基础分数：

```text
base = logit(lgbm_v1_score)
```

使用 `logit` 是因为 LightGBM 输出更接近概率尺度，直接相加不稳定。映射到 log-odds 空间后，后续的 rank penalty、sequence score、support score 更适合线性融合。

对同一文件内的同一 clean peptide，模型计算：

```text
top1_consensus_score =
  base
  + beta * (peptide_top1_score - base)
  - alpha * log(1 + peptide_rank0)
```

其中参数含义如下：

- `peptide_top1_score`：同一文件内，同一 clean peptide 的最高 base score。
- `peptide_rank0`：同一文件内，同一 clean peptide 按 base score 从高到低排序后的 0-based rank。
- `beta=0.5`：让同一 peptide 的多个 PSM 适度向最高证据靠拢。
- `alpha=12`：强惩罚重复 PSM 的高排名，减少同一 peptide 的多个谱图占据 FDR 前排。

这种设计是因为竞赛按唯一 peptide 计数，而不是按总 PSM 计数。原始 PSM 模型容易把同一个高置信 peptide 的多个 PSM 都排得很靠前，这对唯一 peptide 数量的边际贡献有限。top1 consensus 保留每个 peptide 的最佳证据，同时降低重复 PSM 的排序优先级，让更多不同 peptide 有机会进入 1% FDR 阈值。

### 3.2 sequence n-gram score

`sequence_ngram_score` 使用 peptide 字符串和 target/decoy label，学习序列层面的 target/decoy 统计差异。

处理流程：

```text
precursor_sequence / modified_sequence
  -> 清洗修饰
  -> 加边界符：^PEPTIDE$
  -> 字符 3-5 gram
  -> CRC32 hash 到 2^21 维稀疏空间
  -> SGDClassifier 训练 target/decoy 分类器
```

当前参数：

```text
max_sequences = 1500000
max_sequences_per_file = 5000
n_hash_folds = 16
ngram_min = 3
ngram_max = 5
n_features_power = 21
sgd_alpha = 1e-5
sgd_max_iter = 8
seq_weight = 4.5
seed = 20260605
```

参数解释：

- `max_sequences=1500000`：最多抽取 150 万条唯一 peptide 序列训练，控制内存和训练时间。
- `max_sequences_per_file=5000`：限制单文件贡献，避免大文件主导序列模型。
- `n_hash_folds=16`：按 peptide 文本 hash 到 16 个 fold。
- `ngram_min=3, ngram_max=5`：3-gram 捕捉局部氨基酸模式，5-gram 捕捉更长的序列上下文。
- `n_features_power=21`：哈希维度为 `2^21`，尽量平衡收敛速度和内容占用。
- `sgd_alpha=1e-5`：SGD 线性模型的 L2 正则强度。
- `sgd_max_iter=8`：训练轮数，兼顾速度和收敛。
- `seq_weight=4.5`：sequence 分数在最终主体模型中的融合权重。

验证阶段使用 hash OOF：

```text
某个 peptide 属于第 k 个 hash fold
  -> 预测它时只使用不包含第 k fold 的模型
```

这样可以避免同一个 peptide 在训练和验证中被直接记忆，降低离线验证泄漏。测试阶段没有 label，因此使用全部 fold 模型的 ensemble 生成 sequence 分数。

### 3.3 global peptide support score

`global_peptide_support_score` 是预测集内部的无标签支持分数，衡量一个 clean peptide 在当前预测集合中是否有稳定证据。

构建流程：

1. 对每个文件内的每个 clean peptide，只保留 top-k PSM。
2. 统计该 peptide 在多少文件中出现。
3. 统计它被 top-k 支持的次数。
4. 聚合 top PSM 的 base score，形成 noisy-or 风格的支持概率。
5. 加入较弱的 file count / top-k count 奖励。
6. 对所有 peptide 的 support score 做标准化。

当前参数：

```text
support_topk_per_file = 1
support_count_scale = 0.25
support_weight = 4.0
```

参数解释：

- `support_topk_per_file=1`：每个文件中每个 peptide 只取最高分 PSM，避免同一文件内大量重复谱图带来过强支持。
- `support_count_scale=0.25`：出现次数奖励的强度，只作为弱补充，避免变成简单按频次排序。
- `support_weight=4.0`：support 分数在主体模型中的融合权重。

真实 peptide 往往会在多个 scan、多个文件或多个候选中形成一致证据。单个 PSM 分数可能有噪声，但跨文件、跨谱图的重复支持能提升 peptide 层面的可信度。

## 4. clean-peptide prior

`clean_peptide_prior` 使用训练集中 clean peptide 的 target/decoy 统计，为测试集中已在训练数据出现过的 peptide 提供统计先验。

构建流程：

1. 从 `processed_split/train` 读取训练 PSM。
2. 对 `precursor_sequence / modified_sequence / peptide / peptide_key` 做统一清洗，去除或还原修饰标记。
3. 按 clean peptide 聚合 `target_count` 和 `decoy_count`。
4. 对每个 clean peptide 计算 target/decoy log-odds。
5. 测试集预测时，按 clean peptide 查表得到 prior 分数；未命中训练集的 clean peptide 分数为 0。

当前参数：

```python
LEXICON_PRIOR_WEIGHT = 4.0
```

分数定义：

```text
clean_peptide_prior_score =
  log((target_count + 1) / (decoy_count + 1))
```

参数解释：

- `target_count`：该 clean peptide 在训练集中 label 为 target 的 PSM 数量。
- `decoy_count`：该 clean peptide 在训练集中 label 为 decoy 的 PSM 数量。
- `+1`：Laplace smoothing，避免只有 target 或只有 decoy 时出现除零，同时降低极低计数 peptide 的不稳定性。

最终融合：

```text
final_score =
  top1_consensus_score
  + 4.5 * sequence_ngram_score
  + 4.0 * global_peptide_support_score
  + 4.0 * clean_peptide_prior_score
```

## 5. 生成提交结果

```bash
cd /root/aipc

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=/root/aipc/eval/seq_support_clean_prior_w4_cpu8_${STAMP}
SUB=/root/aipc/submissions/0607/seq_support_clean_prior_w4_cpu8_${STAMP}
LEXWORK=/root/autodl-tmp/aipc_work/clean_peptide_prior/seq_support_clean_prior_w4_cpu8_${STAMP}

python src/eval/rescore_sequence_support_grid.py \
  --train-root /root/autodl-tmp/datasets/aipc/processed_split/train \
  --out-dir "$OUT" \
  --max-sequences 1500000 \
  --max-sequences-per-file 5000 \
  --n-hash-folds 16 \
  --ngram-min 3 \
  --ngram-max 5 \
  --n-features-power 21 \
  --sgd-alpha 1e-5 \
  --sgd-max-iter 8 \
  --best-alpha 12 \
  --best-beta 0.5 \
  --best-seq-weight 4.5 \
  --best-support-weight 4 \
  --support-topk-per-file 1 \
  --support-count-scale 0.25 \
  --score-transform logit \
  --cpu-threads 8 \
  --workers 8 \
  --lexicon-prior-min-count 1 \
  --lexicon-prior-work-dir "$LEXWORK" \
  --test-parquet-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
  --test-score-col lgbm_v1_score \
  --submission-out-dir "$SUB" \
  --expected-test-rows 10768114 \
  --seed 20260605
```

## 6. 模型思路总结

1. LightGBM v1 负责 PSM 层面的基础 target/decoy 区分。
2. top1 consensus 把 PSM 排序转成更适合唯一 peptide 指标的排序，保留每个 peptide 的最佳证据，惩罚重复 PSM。
3. sequence n-gram 从 peptide 字符串中学习 target/decoy 的序列模式，使用 hash OOF 降低验证泄漏风险。
4. global peptide support 利用预测集内部的重复证据，提高跨文件、跨谱图支持稳定的 peptide。
5. clean-peptide prior 使用训练集中 clean peptide 的 target/decoy log-odds，提供 peptide-level 统计先验。

最终主体模型可以概括为：

```text
final_score =
  top1_consensus_score
  + 4.5 * sequence_ngram_score
  + 4.0 * global_peptide_support_score
  + 4.0 * clean_peptide_prior_score
```
