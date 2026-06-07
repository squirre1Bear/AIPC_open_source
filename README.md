  # AIPC Peptide Rescoring Pipeline

  本项目用于 AIPC（人工智能蛋白质组学系列竞赛：利用人工智能揭示蛋白质组学的“暗物质”）提交结果生成。

  大致打分流程为：
  LightGBM PSM score
    -> peptide top1 consensus rescoring
    -> sequence n-gram target/decoy pattern score
    -> global peptide support score
    -> optional full-train clean peptide lexicon boost

  其中主要模型是词表重打分之前的 seq_support_strong，词表只作为最后一层较轻量的后处理。

  ## 1. 数据预处理

  训练原始数据位于：

  /root/autodl-tmp/datasets/aipc/original

  Basic 测试集位于：

  /root/autodl-tmp/datasets/aipc/bas_data

  先将不同仪器数据处理为统一 parquet：

  python src/preprocess/process_mzml.py
  python src/preprocess/process_wiff.py
  python src/preprocess/process_tims.py

  Basic 测试集预处理：

  python src/preprocess/process_bas_data.py

  输出目录：

  /root/autodl-tmp/datasets/aipc/processed/bas_merged

  随后为训练集和测试集添加基础手工特征：

  AIPC_WORKERS=8 python src/preprocess/run_feature_1+2.py

  这一步生成的特征主要包括：

  peptide / precursor 组成特征
  修饰相关特征
  理论质量误差特征
  RT / scan 排序特征
  fragment ion 匹配特征（分割 b/y 离子，在 10/20/50 ppm 容差下进行匹配）
  group 内竞争特征

  这些特征均来自谱图、PSM、peptide 字符串和仪器内部统计，不使用物种信息。

  划分训练集和验证集：

  python src/preprocess/split_train_valid.py \
    --data-root /root/autodl-tmp/datasets/aipc \
    --valid-ratio 0.1 \
    --seed 42

  输出：

  /root/autodl-tmp/datasets/aipc/processed_split/train
  /root/autodl-tmp/datasets/aipc/processed_split/valid

  ## 2. LightGBM v1 生成基础 PSM 打分

  后续 rescoring 的基础列是：

  lgbm_v1_score

  该分数由按仪器训练的 LightGBM v1 模型提供。在训练模型时，对三种仪器采用了“分仪器训练”的方式，共训练了3组模型。
  
  按仪器分别训练的原因是 mzML、tims、wiff 的特征分布和打分尺度不同，分开训练可以减少仪器间分布漂移。

  训练命令：

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

  主要参数含义：

  --only-instrument
    每次只训练一个仪器的数据，降低仪器分布差异带来的影响。

  --n-folds 5
    使用 5 折训练，便于生成 OOF 分数和 fold ensemble。

  --balance-by-rows
    按行数平衡不同文件，避免大文件主导训练。

  --neg-pos-ratio 2.0
    控制 decoy/target 采样比例，缓解类别不平衡。

  --train-max-rows / --valid-max-rows
    控制训练和验证采样规模，保证训练速度和内存稳定。

  --max-rows-per-file
    限制单个文件贡献的最大行数，避免某些文件过度影响模型。

  给 Basic 测试集生成 lgbm_v1_score：

  python src/preprocess/add_group_feature_test.py \
    --parquet-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
    --fold-model-dir /root/aipc/models/exp_by_instrument/v1 \
    --force \
    --mask-rt-qvalue-anomaly

  这一步会把 fold ensemble 的 lgbm_v1_score 写回测试 parquet，并生成 group 内排序统计特征。

  ## 3. 基础 Benchmark

  最终提交的主要模型是：

  seq_support_a12_b0p5_sw4p5_pw4

  该模型由三类信号组成：

  benchmark_score =
    top1_consensus_score
    + 4.5 * sequence_ngram_score
    + 4.0 * global_peptide_support_score

  其中参数含义如下：

  ### 3.1 top1 consensus score

  输入基础分数：

  base = logit(lgbm_v1_score)

  logit 转换的作用是把概率型分数映射到无界 log-odds 空间，使后续加性融合更稳定。

  top1 consensus 的公式为：

  top1_consensus_score =
    base
    + beta * (peptide_top1_score - base)
    - alpha * log(1 + peptide_rank0)

  当前参数：

  alpha = 12
  beta  = 0.5

  参数含义：

  peptide_top1_score
    同一个文件内，同一个 clean peptide 的最高 PSM 分数。

  peptide_rank0
    同一个文件内，同一个 clean peptide 按 base 分数从高到低排序后的排名。
    最高分 PSM 的 rank 为 0，第二个为 1，依此类推。

  beta = 0.5
    控制当前 PSM 向同 peptide 最高证据靠拢的程度。
    beta 越大，同一 peptide 的多个 PSM 分数越接近 top1 PSM。

  alpha = 12
    控制重复 PSM 惩罚强度。
    比赛指标是唯一 peptide 数，因此同一 peptide 的第 2、第 3 个 PSM 对最终唯一 peptide 数贡献有限。
    较大的 alpha 会抑制重复 PSM 占据高排名，让更多不同 peptide 有机会进入 1% FDR 阈值。


  由于原始 PSM 模型容易把同一个高置信 peptide 的多个谱图都排在前面。
  但官方指标按唯一 peptide 计数，重复 PSM 不会线性增加分数。
  因此需要保留每个 peptide 的最高证据，同时压低重复 PSM 的排序优先级。

  ### 3.2 sequence n-gram score

  sequence_ngram_score 用 peptide 序列本身训练 target/decoy 区分模型。

  核心处理流程：

  precursor_sequence

    -> clean peptide sequence

    -> 去除修饰

    -> 添加边界符：^PEPTIDE$

    -> 字符 3-5 gram

    -> HashingVectorizer

    -> SGDClassifier(log_loss)

  当前参数：

  ngram_min = 3
  ngram_max = 5
  n_features_power = 21
  sgd_alpha = 1e-5
  sgd_max_iter = 8
  n_hash_folds = 16
  seq_weight = 4.5
  seed = 20260605

  参数含义：

  ngram_min=3, ngram_max=5

    使用长度为 3 到 5 的氨基酸字符片段。
    3-gram 能捕捉局部组成模式，5-gram 能捕捉更长的局部序列上下文。

  n_features_power=21

    HashingVectorizer 的特征维度为 2^21。
    维度足够大，可以降低 hash collision，同时保持内存可控。

  sgd_alpha=1e-5

    SGDClassifier 的 L2 正则强度。
    较小 alpha 允许模型学习较丰富的 n-gram 模式。

  sgd_max_iter=8

    训练迭代轮数。当前数据规模较大，8 轮在速度和收敛之间较平衡。

  n_hash_folds=16

    按 peptide 文本 hash 到 16 个 fold。
    验证时每个 peptide 使用不包含该 hash fold 的模型打分，避免同一 peptide 在训练和验证中直接记忆导致离线泄漏。

  seq_weight=4.5

    sequence n-gram 分数在最终 benchmark 中的融合权重。

  
  由于decoy peptide 通常由构造规则生成，其氨基酸局部模式、边界模式、修饰清洗后的序列形态，可能与真实 target peptide 存在统计
  差异。
  sequence n-gram 模型尝试学习这种序列层面的通用规律。模型只观察 peptide 字符串和 target/decoy label，不引入蛋白来源、物种标签、FASTA 注释或外部数据
  库。

  为了避免变成简单的 peptide 查表，验证阶段使用 hash OOF：

  同一个 peptide 所在 hash fold 不参与该 peptide 的验证预测。

  测试阶段没有 label，因此使用全部 fold 的 ensemble 生成 sequence_ngram_score。

  ### 3.3 global peptide support score

  global_peptide_support_score 是无标签的预测集内部支持分数。

  它不使用 label，也不区分 target/decoy，而是只看一个 peptide 在当前预测集合中的重复证据强度。

  构建流程：

  1. 对每个文件内的每个 clean peptide，保留 top-k PSM。
  2. 统计该 peptide 在多少文件中出现。
  3. 统计 top-k PSM 数量。
  4. 聚合 base score 概率，计算 noisy-or 支持概率。
  5. 加入文件数和 top-k 次数的弱计数奖励。
  6. 对所有 peptide 的 support 分数做标准化。

  当前参数：

  support_topk_per_file = 1
  support_count_scale   = 0.25
  support_weight        = 4.0

  参数含义：

  support_topk_per_file=1
    每个文件中每个 peptide 只取最高分 PSM。
    这样可以避免同一文件内大量重复谱图让某个 peptide 获得过高支持。

  support_count_scale=0.25
    对 file_count 和 topk_count 的计数奖励强度。
    只作为弱信号，避免简单按出现次数排序。

  support_weight=4.0
    global support 分数在最终 benchmark 中的融合权重。


  真实 peptide 往往会在多个 scan、多个文件或多个候选中形成一致证据。
  单个 PSM 分数可能有噪声，但跨文件/跨谱图的重复支持可以提高 peptide 层面的可信度。
  该模块不读取 label，而是让证据更稳定的 peptide 提升，证据弱或重复冗余的 peptide 不额外提升。

  ## 4. 生成 seq_support_strong 提交

  运行命令：

  STAMP=$(date +%Y%m%d_%H%M%S)
  OUT=/root/aipc/eval/seq_support_strong_cpu8_${STAMP}
  SUB=/root/aipc/submissions/seq_support_strong_cpu8_best_${STAMP}

  python src/eval/rescore_sequence_support_grid.py \
    --train-root /root/autodl-tmp/datasets/aipc/processed_split/train \
    --pred-root /root/aipc/models/exp_by_instrument/v1 \
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
    --test-parquet-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
    --test-score-col lgbm_v1_score \
    --submission-out-dir "$SUB" \
    --expected-test-rows 10768114 \
    --seed 20260605

  输出包括：

  $OUT/leaderboard.csv
  $OUT/leaderboard.json
  $OUT/details.json
  $OUT/metadata.json

  $SUB/all_pred.tsv
  $SUB/all_pred.zip
  $SUB/rescore_metadata.json

  其中 $SUB/all_pred.zip 就是基础 benchmark 提交文件。

  ## 5. 词表重打分后处理

  在 seq_support_strong 基础上，最后加入 full-train clean peptide lexicon boost。

  但词表在 Basic 测试集中的命中覆盖率较低，仅有：

  578,626 / 10,768,114 = 5.3735%

  因此它只影响少量测试行，大部分排序仍由前面的 seq_support_strong 模型决定。 用于在少量训练集中已出现过的 clean peptide 上补充 target/decoy 统计先验。

  词表构建自全训练集中的：

  precursor_sequence
  label

  肽段清洗规则：

  n[42]     -> 空
  N[.98]   -> N
  Q[.98]   -> Q
  M[15.99] -> M
  C[57.02] -> C

  每个 clean peptide 统计：

  target_count = label == 1 的次数
  decoy_count  = label != 1 的次数

  词表分数：

  lex_logodds = log((target_count + 1) / (decoy_count + 1))

  最终提交分数：

  final_score = benchmark_score + 4 * lex_logodds

  构建词表：

  python src/eval/build_full_clean_lexicon.py \
    --train-root /root/autodl-tmp/datasets/aipc/processed_split/train \
    --out-dir /root/aipc/eval/lexicon_full_clean_train_$(date +%Y%m%d_%H%M)

  生成最终提交：

  python src/submit/make_lexicon_boost_submission.py \
    --test-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
    --benchmark-dir /root/aipc/submissions/0605/seq_support_strong_cpu16_best_20260605_204201 \
    --lexicon /root/aipc/eval/lexicon_full_clean_train_20260606_1052/lexicon_clean_full_train.parquet \
    --out-dir /root/aipc/submissions/0606/lexicon_full_clean_benchmark_add_w64_20260606_1100 \
    --weight 4 \
    --key-mode clean \
    --score-mode add \
    --base-source benchmark \
    --workers 8 \
    --expected-rows 10768114

  输出：

  /root/aipc/submissions/0606/lexicon_full_clean_benchmark_add_w64_20260606_1100/all_pred.tsv
  /root/aipc/submissions/0606/lexicon_full_clean_benchmark_add_w64_20260606_1100/all_pred.zip
  /root/aipc/submissions/0606/lexicon_full_clean_benchmark_add_w64_20260606_1100/rescore_metadata.json

  ## 7. 最终模型总结

  最终模型可以概括为：

  final_score =
    seq_support_strong_score
    + 4 * full_train_clean_peptide_lexicon_logodds

  其中主体模型 seq_support_strong_score 计算公式为：

  seq_support_strong_score =
    top1_consensus_score
    + 4.5 * sequence_ngram_score
    + 4.0 * global_peptide_support_score
