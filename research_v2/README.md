# ancserAPX Research v2：隔離、可續跑、可驗證的研究入口

這套 CLI 用於完整訓練 Ridge、GBDT、GRU、Transformer，並在凍結的 OOS
預測上做交易成本後的策略搜尋。它與 daily run 隔離：不匯入 scheduler、OMS、
Alpaca adapter 或 production store，不送單，也不修改 production 設定。

CLI 本身沒有訓練時間上限。終端、作業系統或人工中止後，使用相同命令重跑；
已完整驗證的 stage 會跳過，未發布的 partial stage 會重新計算。

## 最短用法

在專案根目錄執行：

```powershell
# 1. 第一次才需要：把 mutable data/store 做成實體、逐檔 SHA-256 驗證快照
python -m research_v2.cli snapshot --snapshot-id canonical

# 2. 全部執行：features -> tabular -> sequence -> search
python -m research_v2.cli all

# 3. 完成後重新雜湊所有結果
python -m research_v2.cli verify --stage all
```

現有 repository 已有 `canonical-20260709`。未指定 `--snapshot` 時，
`canonical` 會依序解析：

1. `research_v2/snapshots/canonical`；
2. 若不存在，選擇名稱排序最新的 `canonical-*`，目前即
   `canonical-20260709`。

預設設定是 `research_v2/default_config.json`，預設 run id 是 `canonical`。
三者都是明確、可覆寫的 canonical 預設，不會偷偷使用 production config。

如果資料或設定改變，請使用新的 immutable run id：

```powershell
python -m research_v2.cli all `
  --snapshot canonical-20260710 `
  --config research_v2/default_config.json `
  --run-id full-20260710-v2
```

同一個 run id 若對應到不同的 data hash 或 config hash，CLI 會拒絕混用。

## 命令

### `snapshot`

```powershell
python -m research_v2.cli snapshot \
  --snapshot-id canonical-20260710
```

PowerShell 換行請把 `\` 換成反引號。這個命令：

- 在複製前後分別檢查來源 manifest 與所有 Parquet；
- 建立實體 copy，不使用 symlink；
- 每檔保存 size、mtime、SHA-256；
- 全部成功才用同 volume rename 原子發布；
- 已存在且驗證成功時預設回傳 `skipped_verified`。

如需指定測試來源，可傳 `--store-dir` 和 `--manifest-path`。這兩者只影響
snapshot 的 read source，所有輸出仍被限制在指定的 `research_v2` root。

### `features`

```powershell
python -m research_v2.cli features --run-id canonical
```

建立 point-in-time feature panel：

```text
close(t) 已知資訊 -> open(t+1) 執行 -> open(t+1+h) 標籤
```

輸出包含 legacy factors、精確 6-1/9-1/12-1 momentum、reversal、volatility、
overnight/intraday、liquidity、breadth/regime、截面 rank、label 與資料品質報告。

### `tabular`

```powershell
python -m research_v2.cli tabular --run-id canonical
```

若 features 尚未完成會先建立；若已完成則重新雜湊後跳過。模型包括：

- deterministic Ridge；
- deterministic histogram GBDT；
- 只平均每日截面 calibrated rank 的 Ensemble；
- purged + embargoed chronological walk-forward；
- selection OOS 決定超參數與 ensemble weights，之後才評估 lockbox。

訓練 universe 固定為 snapshot 中全期間 complete-case symbols，避免 held bar
缺失被默默當成 0 return。這仍無法消除今日股票池造成的 survivorship bias；
結果中會保留這項限制。

### `sequence`

```powershell
python -m research_v2.cli sequence --device auto
```

訓練小型 GRU 與 Transformer：

- 直接讀取同一組 point-in-time feature 序列；
- sequence endpoint 嚴格位於 train/validation/test whitelist；
- 每 fold 使用固定 seed，validation 按時間在 train 之後；
- 選模結束後鎖定 sequence ensemble，再評估 lockbox；
- `--device auto`、`cpu`、`cuda` 或 `cuda:0` 都會寫入 artifact parameters。

GPU 數值在不同 CUDA、driver 或硬體之間不保證 bit-for-bit 相同，所以 marker
同時保存 Python、PyTorch、所有核心套件版本與平台。要求最強跨機器可重現性時
使用 `--device cpu`，並使用新的 run id，避免與 `auto` artifact 混用。

### `search`

```powershell
python -m research_v2.cli search --device auto
```

`search` 會先確保 tabular 與 sequence 都完整，然後把兩者的 frozen OOS scores
合併。它不重看模型內部訓練資料來選 portfolio。

搜尋分階段進行：

1. 在 selection OOS 選 score；
2. 選 top-N、equal/inverse-vol、rebalance/staggered cadence 與 risk overlay；
3. 在預先聲明的 10 bps 額外 friction 下選 leverage；
4. 固定 champion 後才做 lockbox；
5. 報告 0/5/10/20 bps sensitivity、fold、neighborhood 與 rebalance-offset
   sensitivity。

完整成本還包含配置中的固定 friction、range-derived spread proxy、平方根
market impact、ADV participation、funding；所以 10 bps sensitivity 是額外壓力，
不是成本的全部。

### `all`

```powershell
python -m research_v2.cli all --device auto
```

依賴圖如下：

```text
verified snapshot
       |
    features
    /      \
tabular   sequence
    \      /
      search
        |
   all completion marker
```

每個節點都先驗證再決定 skip/run。`all` 不是一次性黑箱；每個 stage 可獨立重跑
與審計。

### `verify`

```powershell
python -m research_v2.cli verify --stage snapshot
python -m research_v2.cli verify --stage run
python -m research_v2.cli verify --stage features
python -m research_v2.cli verify --stage tabular
python -m research_v2.cli verify --stage sequence
python -m research_v2.cli verify --stage search
python -m research_v2.cli verify --stage all
```

`verify` 不訓練。它會重算 snapshot 或 artifact 中每個檔案的 SHA-256，檢查檔案
集合、size、stage/run/data/config identity。`--stage all` 要求 features、tabular、
sequence、search、all marker 全部存在且通過。

## Artifact 佈局

```text
research_v2/
  snapshots/
    canonical-*/
      snapshot.json
      manifest.json
      store/*.parquet
  runs/
    <run-id>/
      run.json
      config.json
      features/
        panel.parquet
        report.json
        feature_columns.json
        _SUCCESS.json
      tabular/
        selection_oos_predictions.parquet
        lockbox_predictions.parquet
        fold_records.json
        locked_settings.json
        folds.json
        summary.json
        _SUCCESS.json
      sequence/
        selection_oos_predictions.parquet
        lockbox_predictions.parquet
        fold_records.json
        locked_settings.json
        settings.json
        folds.json
        summary.json
        _SUCCESS.json
      search/
        search_summary.json
        champion.json
        stage_a.csv
        stage_b.csv
        stage_c.csv
        configuration_neighborhood.csv
        rebalance_offset_sensitivity.csv
        fold_metrics.csv
        _SUCCESS.json
      all/
        pipeline.json
        _SUCCESS.json
```

每個 `_SUCCESS.json` 保存：

- `snapshot_data_sha256`：manifest 與所有 snapshot file content 的聚合 hash；
- `config_sha256` 與完整 canonical config copy；
- `code_sha256`：該 stage 執行前後一致的 Research v2 source hash；
- Python/OS 與 NumPy、Pandas、Polars、PyArrow、scikit-learn、PyTorch 版本；
- base seed 與每 fold/refit 的 seed 公式；
- stage parameters hash；
- upstream `_SUCCESS.json` hashes；
- 所有 output 的相對路徑、size 與 SHA-256。

只有 writer 完成、code 未在途中改變、marker 自我驗證成功後，partial directory
才會 rename 成正式 stage。正式 stage 視為 immutable：若被改動，resume 會失敗，
不會把殘缺資料當成成功結果。

## Resume 與失敗處理

預設就是 `--resume`：

```powershell
python -m research_v2.cli all --run-id full-20260710-v2 --resume
```

若要確認 stage 尚不存在，而不是續跑：

```powershell
python -m research_v2.cli tabular --run-id new-audit-run --no-resume
```

規則：

- 正式 stage + 完整 marker + hashes 一致：跳過；
- 正式 stage 缺 marker、檔案缺失或 hash 改變：fail closed；
- 上游 artifact 改變：下游 identity 不符，拒絕沿用；
- process 在發布前中止：正式 stage 不會出現，下次重新執行該 stage；
- config/snapshot 與既有 run id 不同：拒絕混合，應改用新 run id。

CLI 目前的 checkpoint 粒度是 stage，而不是單一 epoch。也就是完整的 tabular 或
sequence stage 一旦成功便永久跳過；若 sequence 在正式發布前中止，會從該 stage
開頭重跑，以免把不同 code/version/seed 的 folds 拼在一起。

## 對 daily run 的隔離保證

所有 CLI command 都在 process-local `offline_context` 中執行：

- 暫時移除本 research process 的 `APCA_*`、`ALPACA_*`、
  `PAPER_TRADING*` 環境變數；
- 阻擋 Alpaca、data fetch/store、backtest production engine、execution、OMS、
  scheduler/server imports；
- output path 必須位於一個名為 `research_v2` 的 root；
- 不改其他 OS process 的環境，不停止 daily runner；
- 結束時精確恢復本 process 原有環境。

snapshot 是唯一會讀 mutable `data/store` 的命令，而且只做 verified physical copy。
後續 features、training、search 只讀已完成 snapshot。

## 如何解讀結果

`tabular/summary.json` 和 `sequence/summary.json` 是 signal-level OOS/lockbox IC
診斷；它們不是可交易淨績效。真正用來比較候選的是 `search/champion.json`、
`search_summary.json` 和成本敏感度 CSV。

即使 champion 通過目前資料，也只能稱為當前 snapshot/universe/成本假設下的
最佳候選，不能稱為永久最優。特別要保留以下限制：

- 今日 universe 回看歷史仍有 survivorship bias；
- sector map 與本地 benchmark 覆蓋有限；
- 日線只能近似 next-open，不能精確重建 09:35 fill；
- spread/impact proxy 尚需用真實成交校準；
- lockbox 一旦被查看，就不能反覆調參後仍宣稱 untouched。

下一個真正乾淨的證據是持續 shadow/paper forward test，而不是再度優化已看過的
2026 lockbox。

## 行業中性化研究

固定現有 OOS 預測、只改每日橫截面轉換並跑完整成本／offset A/B：

```powershell
python -m research_v2.run_neutralization_study_v3
```

完成後輸出位於：

```text
research_v2/runs/20260710_full_v1/neutralization_study_v3/
```

其中 `study_plan.json` 在計算結果前鎖定規格，`manifest.json` 固定輸入、程式、
live-clock semantic inputs、sector snapshots 與全部輸出 hash，`report.md` 是人讀結論；
CSV／Parquet 保存 IC、fold、成本敏感度、offset、paired bootstrap 與平均持倉分配。
完成目錄不可由 runner refresh／overwrite，發布後還會核對 exact file set。

Claude weekly 場景是 prior-close→最後交易日 official-open 的 480 檔 complete-case
proxy，不是 exact Main parity：Main config 目前為 510 檔，且 production 約 09:35 ET
成交。首次建倉、stale catch-up 與 calendar fallback 也不在此正常週期 A/B。
這個研究只測 static broad-sector neutralization；缺少歷史 point-in-time market cap
與 shares 時，不會輸出或冒稱市值中性化結果。

## LETF universe rotation

跨 sector 的 Seed-30 LETF rotation 使用獨立的 research-only runner；它不修改 live
universe，也不匯入 broker execution。Universe 身分事件、proxy-first 訊號、next-open
成交、成本／容量、同實現波動 SPY control 與 robustness gate 的完整規格及目前結論見
[`docs/LETF_ROTATION_RESEARCH.md`](../docs/LETF_ROTATION_RESEARCH.md)。

```powershell
python -m research_v2.run_letf_rotation `
  --snapshot research_v2/snapshots/letf-sip-clean-20260717-v2 `
  --run-id letf-seed30-sip-20260720-v4 `
  --bootstrap-repetitions 5000
```

目前正式結果分類為 `PIT_APPROX_INVALID_FOR_ALPHA_CLAIM`：主策略相對同實現
波動 SPY evaluation control 的 5,000 次 paired block bootstrap 勝出機率只有
10.16%，5 個
rebalance offsets 與 16 個預先聲明鄰域皆無正 excess。這是可重現的否證結果，
不是 live deployment candidate；完整數字、Seed-30 survivorship 限制與兩週實盤
cash-flow 對帳見上述研究報告。
