# LETF Universe Rotation：Alpha、回測與穩定性驗證

## 結論先行

圖片中的 `+4,906.7% / CAGR 52.0% / Sharpe 2.28` 應被視為研究假說，不是已證實的 alpha。ancserAPX 現在有一條完全隔離於 live 的 LETF 研究鏈：point-in-time 產品身分、proxy-first 訊號、曝險群組與相關性限制、`close(t) -> open(t+1)` 成交、成本／容量模型、基準比較與穩定性壓力測試。

目前透明重建的結論是「不 match」：在可交易評估期 2017-01-09 至 2026-07-17、乾淨 SIP／`Adjustment.ALL` snapshot 上，保守主策略的 net CAGR 為 3.69%、Sharpe 0.45、MaxDD -27.94%；在完全相同的共同交易日，把 15% target-vol SPY 的 daily return 以 0.6444 倍縮放到與策略相同的 8.85% 實現波動後，evaluation-only SPY control 的 CAGR 為 7.69%、Sharpe 0.88、MaxDD -15.29%。即使改成較接近圖片的「產品自身動能、Top-5 等權、移除風控與群組限制」，CAGR 也只有 7.66%、Sharpe 0.40、MaxDD -77.78%。因此現有可公開描述的規則無法重現圖片曲線。

| 同一資料與交易時鐘 | CAGR | Sharpe | MaxDD | 累積報酬 |
| --- | ---: | ---: | ---: | ---: |
| LETF rotation 主策略 | 3.69% | 0.45 | -27.94% | 41.13% |
| 同實現波動 SPY control（ex-post） | 7.69% | 0.88 | -15.29% | 102.05% |
| 15% target-vol SPY mandate | 11.80% | 0.88 | -22.97% | 188.48% |
| 圖片近似 ablation | 7.66% | 0.40 | -77.78% | 101.57% |

主策略相對同實現波動 SPY control 的 paired 21-session moving-block bootstrap（5,000 次）年化平均 excess 為 -3.78%，95% CI 為 [-9.57%, +2.14%]，勝出機率只有 10.16%。16 個預先聲明的參數／風控鄰域與 5 個 rebalance offsets 都沒有正 excess；經驗 gate 只通過 3/13，結構 gate 通過 6/14。因此目前分類固定為 `PIT_APPROX_INVALID_FOR_ALPHA_CLAIM`，不可部署。

這不是說原圖一定錯，而是表示原圖必定還有未被明確描述的訊號、事件定義、交易語義或選樣過程；在那些規則被版本化以前，不能把曲線接到 live。

## 什麼才是 alpha

Alpha 不是「最後賺了多少美元」，也不是「因為 3x LETF 複利後曲線很高」。在這個系統裡，alpha 至少要同時滿足：

1. 使用同風險、同可交易時鐘的 benchmark 後仍有正超額報酬；
2. 扣除 spread、impact、額外 friction 與容量限制後仍成立；
3. 不靠 market beta、科技集中、產品槓桿或單一幸存標的解釋；
4. 在未參與選參數的 OOS／lockbox、不同 rebalance offset、成本與參數鄰域仍穩定；
5. 統計不確定性足夠小，而且 live／paper fill 能對上回測假設。

本研究同時列出 15% target-vol SPY mandate 與 scenario-level control。alpha gate 使用後者：先在精確共同交易日把 SPY daily returns 縮放到每個策略情境的同實現波動，再比較 compounded return、fold、offset、鄰域與 paired bootstrap。這是 ex-post evaluation control，不是可交易或無前視的 SPY 策略；另列 SPY、QQQ、UPRO 與 point-in-time Seed-30 月度等權。最終美元曲線只用來說明複利，不用來定義 alpha。

## Universe 的強化

固定 Seed-30 來自圖片：

`FNGU, DPST, SOXL, KORU, ERX, TNA, AGQ, UTSL, GUSH, CWEB, EURL, EDC, RETL, BOIL, TECL, DFEN, CURE, YINN, DRN, FAS, TQQQ, LABU, DUSL, NAIL, NUGT, UPRO, TPOR, SPXL, WANT, MIDU`

系統不再把 30 個 ticker 當成 30 個獨立 sector。下列重疊群組同時最多一檔：

- Technology：`FNGU / SOXL / TECL / TQQQ`
- Financials：`DPST / FAS`
- Energy：`ERX / GUSH`
- China：`CWEB / YINN`
- US large cap：`UPRO / SPXL`
- Consumer cyclical：`RETL / WANT`

每個產品都有 `instrument_id`、有效日期、結構（ETF／ETN）、日目標槓桿、theme、macro bucket、1x proxy 與官方來源。重要歷史事件也分段保存：

- 現在的 note 於 2025-02-20 以 `FNGB` 推出，2025-06-24 才改 ticker 為 `FNGU`；不得與 2018 年的舊 FNGU／後來的 FNGA 串接。資料商若把更名前歷史全部正規化成今天的 `FNGU`，eligibility 會 fail closed，而不是把它冒充為當時可交易的 FNGU。
- ERX、GUSH、NUGT 在 2020-04-01 由日目標 3x 改為 2x。
- SOXL、FAS、TPOR、NUGT 的 underlying index 變更有獨立 regime 邊界。

每日 eligibility 必須同時符合產品身分有效、當日存在、最近 252 個 global market sessions 每天都有真實 bar。禁止 synthetic pre-inception history、forward-fill、缺 bar 當 0 報酬，以及用今天 active 的 broker API 倒推歷史成分。

兩個保守限制會刻意少算、而不是偷補資料：這份 snapshot 沒有獨立 `FNGB` keyed bars，所以 formal runner 在 2025-06-24 改名邊界重新累積 FNGU warm-up，沒有把資料商正規化的舊 `FNGU` rows 當成 FNGB continuity；另外，本次 `retrieved_at_utc` 是從當時保留的 cache／檔案時間補錄，不是原始抓取 request 當下由 fetcher 原生寫入。v4 的 aggregate hash 能證明之後 58 個檔案未變，不能倒過來獨立證明當時資料商回應的真實性。未來 snapshot 必須由抓取流程原生寫 provider/feed/adjustment/request timestamp 才能消除此 provenance 限制。

仍需誠實保留一個限制：Seed-30 是今天選出的存續產品，沒有包含歷史上已清算的 LETF。即使每日 eligibility 完全 non-lookahead，整個 seed 仍有產品 survivorship／selection bias；因此目前 artifact 固定標為 `PIT_APPROX_INVALID_FOR_ALPHA_CLAIM`。

## 預設策略與執行語義

訊號只使用 close(t) 當下已知的 1x proxy：

- 126 日與 63 日 momentum，跳過最近 5 sessions；
- momentum 除以 trailing-63 日 proxy volatility；
- 21／63 日 acceleration；
- absolute gate：proxy `close(t) > SMA200(t)` 且 63 日報酬為正；
- 三項做當日 cross-sectional percentile 後，以 `45% / 35% / 20%` 合成。

每 5 個 global sessions 決策，依分數 greedy 選 Top-5；同 theme 最多 1、同 macro bucket 最多 2，任兩個候選 trailing-126 日絕對相關係數不得高於 0.70。不足五檔時保留 cash，不放寬條件硬補滿。

所有 target 仍由既有 `research_v2.portfolio.construct_portfolio()` 生成，沒有第二套隱藏權重真相。主策略使用 inverse-vol、單檔／theme 25%、外層 gross 上限 1.0、15% target volatility、SPY trend + breadth hysteresis 與 drawdown governor。

考慮帳戶曾發生 margin call，LETF 已內含 2x／3x 日槓桿，研究預設禁止再疊 broker margin。若 ADV clipping 使減倉無法完成，系統保留實際未成交路徑並讓 gross／容量 gate 失敗，不會假裝全部成交。

成交時鐘為：

1. close(t) 產生訊號；
2. open(t+1) 依 lagged ADV、spread proxy 與 square-root impact 成交；
3. 缺 held bar 立即 raise；
4. 每日對 cash、position、P&L 與成本做會計恒等式核對。

LETF 的管理費、產品內融資、每日重置、期貨 roll 與 volatility decay 已反映在產品真實價格路徑，不重複扣除，也不以「underlying 多日報酬 × 3」合成。

## 長期穩定性測試

一個候選只有通過以下 battery 才能進 untouched lockbox：

1. **資料與身分**：100% PIT eligibility/event audit；FNGU 等 ticker reuse 不串接；held 缺 bar fail closed。
2. **時間外推**：rolling 3 年 train + 1 年 validation + 1 年 test，purge／embargo 各 21 sessions，最後 12 個月只開一次 lockbox。
3. **Offset**：5 個 weekly offsets 全測，避免恰好從某一天開始。
4. **參數鄰域**：Top-K、lookback、cadence、相關性、equal／inverse-vol、regime 與 target-vol 的小型預先聲明 grid；至少 70% 鄰域有正同實現波動 SPY-control excess。
5. **成本／成交**：額外 0／5／10／20／40 bps、延遲與 next-close stress；20 bps 後 Sharpe 至少 0.7。
6. **集中度**：leave-one-symbol-out、leave-one-theme-out、移除最大貢獻者後仍有正 excess。
7. **統計**：5,000 次 paired circular moving-block bootstrap；另需 Deflated Sharpe、PBO 與多重測試修正。
8. **市場狀態**：COVID、2022 bear、科技 bull、最近修正期分段；報 worst day／week／month、CVaR、time under water。
9. **容量**：每筆預計成交不超過 lagged ADV 1%，並測不同 AUM；不可讓巨大複利曲線逃避容量限制。
10. **Live alignment**：untouched lockbox 通過後仍需至少三個月 shadow／paper，且用真實 fills 校準成本，才可討論 live。

目前已完成 snapshot provenance、資料／身分、next-open、成本、offset、參數／風控 ablation、年度 folds、paired bootstrap 與容量 gate。Leave-one-out、purged walk-forward、DSR/PBO、真正 dead-fund-inclusive master、untouched lockbox 和三個月 shadow 尚未完成，所以結構 gate 必定失敗。因為候選已在同實現波動 excess、bootstrap、offset 與所有鄰域明確失敗，沒有花更多算力替一個失敗候選完成 leave-one-out；這不是「未測所以可能通過」，而是明確記為 gate fail。

## 兩週 baseline 與入金對帳

Broker ledger 顯示 2026-07-08 入金 $2,000，2026-07-17 再入金 $1,000。舊 tracker 把第二筆入金當成策略獲利，因此曾顯示約 `+$935 / +50.85%`；這個數字不成立。修正後，截至 2026-07-20：

- equity：$2,833.03；
- 淨投入：$3,000；
- money-weighted endpoint P&L：-$166.97；
- 排除外部現金流的 linked-return 估計：-9.4164%（假設 transfer 發生於 tracker observation 期末）。

`-$166.97` 是由 ending equity 減淨投入得到的確定美元損益；`-9.4164%` 不是精確 TWR。現有 broker activity 只有日期／午夜時間，沒有入金前一刻的 account valuation，所以現階段只能依明示的期末現金流假設估計。adapter 現在會優先保存 `transaction_time`，UI 也把 observation P&L 與 broker prior-close、排除當日外部現金流的 calendar-day dollar P&L 分開，不再把同日兩次 snapshot 間的變化誤標為整日損益。

兩週 baseline 目前只能判定為「不 match，且不能做正式 attribution」：回測與 live 的 rebalance clock 不同，歷史 daily account snapshots 也不完整。可比的臨時窗口內，回測 7/09→7/14 約 -2.04%，live 7/09→7/15 約 +0.83%，差約 +2.87 個百分點；但 target 只有 15/20 重疊，不能把差異解讀為 live alpha。

已修復三個可確定的 execution 問題：

1. tracker 以 broker cash activity 排除入／出金，activity 不可取得時 fail closed，不再把 deposit 當 P&L；
2. 7/17 的 partial rebalance（planned 29、submitted 9、failed 20）不再推進 cadence，OMS 先處理 cancel／sell、刷新 account 後才 preflight buys，讓下一個合資格時段可恢復；
3. account／trading blocked、非 `ACTIVE`、NaN／Inf 權重或 equity 會直接 hard-block，即使 target 不超過 1x；帳戶可交易但 equity 低於 $2,000 或 multiplier 不允許 margin 時，才把外層 gross 上限縮到 1.0。槓桿目標缺少完整 margin 欄位也會 fail closed。

現有 broker 資料能證明 7/16 equity 為 $1,868.42、buying power 為 $0、cash 為 -$963.99，但 repository 沒有正式 margin-call notice，所以報告不冒稱已核實券商發出 margin call。使用者補入的 $1,000 已依外部現金流處理。7/20 tracker 的實際 gross 約 0.522，而保存的 target gross 是 1.5，亦證明 partial batch 沒有達到計畫曝險。live config 的 1.5x 選擇沒有被靜默修改；guard 只在帳戶不具 margin 資格時阻止外層槓桿。正式 parity 還需要把 backtest 與 live 鎖成同一個 calendar clock，這是策略語義變更，不能用事後最佳結果替使用者偷偷選擇。

## 如何重跑

```powershell
python -m research_v2.run_letf_rotation `
  --snapshot research_v2/snapshots/letf-sip-clean-20260717-v2 `
  --run-id letf-seed30-sip-20260720-v4 `
  --bootstrap-repetitions 5000
```

完整 leave-one-symbol/theme 研究可加 `--full-robustness`，但會逐情境重跑 event ledger，耗時显著增加。

主要程式與 artifact：

- `research_v2/letf_universe.py`：instrument/event registry 與 PIT eligibility。
- `research_v2/letf_rotation.py`：純 close(t) signal／selector，不產生權重。
- `research_v2/letf_experiment.py`：next-open、成本、風險、benchmark 與 bootstrap。
- `research_v2/run_letf_rotation.py`：offline runner 與 immutable artifact。
- `research_v2/runs/letf-seed30-sip-20260720-v4/`：正式研究輸出；`_SUCCESS.json` 固定 58 個 Parquet 的 aggregate hash、snapshot manifest、config、registry、10 個 source 與每個 output 的 hash。

此流程不修改 `config/live_strategy.json`，不新增 live universe，也不送出任何 broker 訂單。
