# Checklist

- [x] gen_data.py 支持 `--p-ground`，内部电极按概率标 1（接地），外框环恒为 1，每个布局 ≥1 HV（全接地时强制改一个为 HV）——`_colorize` 实现，实测 121 ground / 474 HV
- [x] 电极类型（ground/hv）逐电极写入 gen_report.json，且同 seed 复跑类型分配逐位一致（Task 2.3 PASS）
- [x] 工况电极数分层轮转：n_target = 1 + (cond % N_max)，实测目标分布 rect 各 5、circle 8–9、polygon 12–13、blob 16–17，均 ∈ {⌊50/N⌋, ⌈50/N⌉}
- [x] variant_ok 同时检查 HV 与内部接地连通域碎片（≥20px，外框环豁免）
- [x] 校验关⑥显式断言落盘 region 唯一值 ⊆ {1,2,3}，出现 0 拒写盘——全量 2000 个 region CSV 零文件含 0
- [x] 验证批 16 对六关全过；含内部接地电极样本（circle_c001）BC 精确（内部接地 5517px 全 0V、HV 3947px 全 60000V）
- [x] verify_data.py 对验证批反向闭环 rel L2 max 2.874e-08 < 1e-3
- [x] 正式生成 2000/2000 对通过六关（4 族 × 50 工况 × 10 变体），gen_report.json 四族齐备（各 500 written / 0 failed）
- [x] verify_data.py 全量 2000 对反向校验通过率 100%（rel L2 mean 2.66e-08 / p90 3.02e-08 / max 6.01e-08，求解残差 max 3.6e-15）
- [x] dataset.py 根目录递归加载发现 2000 对 / 200 工况；单工况路径加载行为回归不变（one-hot 互斥、掩码一致）
- [x] data/ 下阶段二 16 对验证批已被新数据替换；临时目录 data_valbatch 已清理
- [x] retangle_data 未删除、未修改，且不再被任何新生成/训练流程引用
- [x] proceedings/5. 阶段三记录.md 落盘，含标注方案 v2、工况设计、生成与校验实测、旧数据弃用声明
