# Tasks

- [x] Task 1: 扩展 gen_data.py —— 标注方案 v2 + 内部接地电极 + 分层工况 + 校验加固
  - [x] SubTask 1.1: 重构四族生成器为"电极掩码列表 + 统一类型采样上色"：各生成器返回电极像素掩码列表，公共函数按 `--p-ground`（默认 0.25）采样接地/HV 类型并上色（保证 ≥1 HV），外框环恒为 1；电极类型写入 electrodes 元数据
  - [x] SubTask 1.2: 工况电极数分层轮转：`n_target = 1 + (cond % N_max)`（rect 10 / circle 6 / polygon 4 / blob 3），保持最佳努力放置
  - [x] SubTask 1.3: `variant_ok` 扩展：裁剪后 HV 连通域与内部接地连通域均 ≥20px（外框环豁免）
  - [x] SubTask 1.4: 新增校验关⑥：落盘前显式断言 `set(unique(region)) ⊆ {1,2,3}`，失败拒写盘；docstring 更新为标注方案 v2 与六关校验说明
  - [x] SubTask 1.5: `gen_report.json` 记录 p_ground、各电极类型、各工况目标/实际电极数
- [x] Task 2: 验证批（每族 2 工况 × 2 变体 = 16 对，输出至临时目录）
  - [x] SubTask 2.1: 四族验证批生成，六关全过；抽查含内部接地电极的样本 BC/E=0 精确
  - [x] SubTask 2.2: verify_data.py 对验证批反向闭环，rel L2 < 1e-3
  - [x] SubTask 2.3: 同 seed 复跑一族，电极类型分配与几何逐位一致（可复现性）
- [x] Task 3: 正式生成 2000 对（4 族 × 50 工况 × 10 变体 → `data/`，替换阶段二 16 对验证批）
  - [x] SubTask 3.1: 清理 `data/` 下旧验证批四族目录
  - [x] SubTask 3.2: 四族正式生成（--conditions 50 --variants 10），2000/2000 对通过六关
- [x] Task 4: 全量反向闭环与加载回归
  - [x] SubTask 4.1: `verify_data.py --data-root data` 全量 2000 对校验，rel L2 < 1e-3 通过率 100%
  - [x] SubTask 4.2: dataset.py 根目录递归加载回归：2000 对 / 200 工况；单工况路径行为不变
- [x] Task 5: 阶段三数据记录文档（proceedings/5. 阶段三记录.md）：标注方案 v2、工况设计、生成与校验实测数字、retangle_data 弃用声明

# Task Dependencies

- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 4
