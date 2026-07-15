# ancserAPX v1.0.0 策略升級報告

**主題**：非線性機器學習、序列模型、集成、交易成本與多層風險管理  
**日期**：2026-07-10  
**文件狀態**：設計與實施路線；除「現況」明確標示的功能外，本報告其餘內容均為待實作／待驗證方案  
**核心目標**：在市場波動放大、量化交易擁擠與 alpha 衰減加速的環境下，提高策略的淨收益穩健性與存活率，而不是只提高樣本內 CAGR。

---

## 1. 執行摘要

結論不是「立刻用 Transformer 取代因子」，而是建立一個可以持續更新的信號生產與風險控制系統：

1. **先讓回測誠實**：目前多 sleeve 回測只扣借貸成本，沒有佣金、bid-ask spread、滑價、market impact 與換手成本。未先補齊成本模型，ML 很容易把高換手誤認成高 alpha。
2. **GBDT 作為第一個 ML 主模型**：因子與未來收益通常不是單一線性斜率。GBDT 可用分段切分與特徵交互捕捉非線性，適合先承接現有 `factor_*` 特徵庫。
3. **GRU／Transformer 作為獨立序列信號**：它們可學習量價路徑與時序表徵，但不應在第一階段刪除手工因子，也不應直接吃未正規化的原始價格。序列模型是補充，不是預設的替代品。
4. **Ensemble 是正式產線方向**：先把不同模型輸出轉成同一天、同股票池的可比截面 rank，再以 walk-forward 樣本外、扣成本後的結果決定權重。不可直接平均量綱不同的原始預測值。
5. **風控必須與 ML 同步升級**：加入持倉真實波動目標、drawdown governor、行業與單名上限、beta/gross 約束、真正的擁擠監測、換手懲罰與固定壓力測試。
6. **模型是消耗品，不是永久資產**：採每 1–3 個月候選重訓、champion/challenger、漂移監控與可回退版本。只有通過樣本外淨成本門檻的模型才可升級到實盤。

建議順序：**成本與驗證框架 → 小幅風控修復 → GBDT → 序列模型 → 集成／meta-labeling**。

---

## 2. ancserAPX v1.0.0 現況與缺口

以下判斷以目前程式碼為準。

| 層級 | 現有能力 | 主要缺口 |
|---|---|---|
| 因子 | 動量、反轉、波動、Amihud、spread proxy、Alpha101、EMA、KDJ/RSI、sector rank 等量價因子 | 資訊來源仍高度集中於 OHLCV；缺 point-in-time 基本面、財報事件、overnight/intraday 拆分及 ML 特徵快照 |
| 組合 | 多 sleeve、top-N、winner lock、槓桿、等權核心、部分 sector neutralization | 核心持倉仍為等權；缺 inverse-vol、行業硬上限、beta/風險預算與成本感知優化 |
| 風控 | QQQ/SPY 200EMA cash/throttle、20EMA 重啟、vol throttle、價格／成交額過濾、異常區間過濾 | vol 以股票池等權收益估算，不是實際 target weights；現有 `crowding_shock_guard` 實際是 range filter，不是真正的 crowding gauge；缺回撤分級降槓桿 |
| 動態因子權重 | MWU 依 rank IC 更新 | 使用單日 IC 立即更新；`window` 尚未真正用於平滑；缺 EWMA、收縮、換手抑制與權重穩定門檻 |
| 回測 | 與實盤共享 `combined_target_weights()`，具 parity 基礎；支援持有期與風控 overlay | P&L 只扣 borrow，未扣交易摩擦；缺 turnover 明細、容量模型、成本敏感度掃描與固定壓力測試 |
| ML | 尚無正式依賴、資料集、模型 registry 或 OOS 預測儲存層 | 必須先建立 point-in-time 特徵／標籤、purged walk-forward、模型版本與離線預測管線 |

目前最有價值的資產是 **回測／實盤共用投資組合邏輯**。ML 不應繞過這條 parity 路徑；模型應輸出新的、具版本與時間戳的 factor score，再由相同組合層使用。

---

## 3. 為什麼先做 GBDT

### 3.1 GBDT 是什麼

GBDT（Gradient Boosting Decision Trees，梯度提升決策樹）是一組按順序建立的淺層決策樹。每一棵新樹修正前面模型尚未解釋好的誤差，最後把多棵樹的輸出相加。

它對截面選股特別有用的原因是：

- **分段反應**：模型可以讓某個因子的中段近似平坦，只在極端區間提高或降低分數。
- **非線性交互**：例如「強動量」只有在低特異波動、流動性充足且市場 breadth 健康時才有效。
- **不要求線性量綱**：樹依切分排序工作，通常不需像線性／神經網路一樣先做標準化。
- **可控制複雜度**：淺樹、較大 leaf、feature/bagging 抽樣、L1/L2 與 early stopping 可降低金融低信噪比下的過擬合。

LightGBM 與 XGBoost 都是 GBDT 家族的成熟實作。ancserAPX 第一版可選 LightGBM，主要是訓練速度與大截面資料處理方便；最終仍需以本系統自己的 walk-forward 淨成本結果決定，而不是以套件名氣決定。

### 3.2 「中段躺平、尾部發力」應視為可檢驗假說

以動量為例，可以提出以下假說：

> 動量值位於截面中段時預測力弱；真正有資訊的可能是極強／極弱尾部，而且效果取決於波動、流動性與市場 regime。

線性模型只能用單一斜率近似整段關係，確實可能由中段噪音稀釋尾部信號；GBDT 則可建立尾部分支。但「±1σ 中段沒有預測力」不是所有市場與時期都成立的定律，必須用以下方法確認：

1. 僅在 training window 內建立 factor decile／ventile 圖。
2. 在 validation/test 觀察各分箱未來 5 日的截面超額收益、信賴區間與單調性。
3. 以 out-of-sample partial dependence、SHAP 分布及跨期穩定性檢查模型是否真的在尾部學到可重複關係。
4. 對尾部樣本數設最低門檻，避免模型只記住少數極端事件。

Gu、Kelly 與 Xiu（2020）在美股的大型比較中發現，樹模型與神經網路的優勢與非線性及 predictor interaction 有關；其重要信號也包含動量、流動性與波動。這支持「測試非線性 ML」的方向，但不代表任何資料集上的 GBDT 都必然勝出，也不能直接外推為特定私募的實際模型配置。

### 3.3 樹模型不是資料治理的豁免權

GBDT 能原生處理缺失值，且對特徵縮放不敏感；這不等於可以忽略資料品質：

- 缺失本身可能洩漏上市時間、資料供應商覆蓋或財報發布狀態。
- 極端值仍可能來自拆股、錯價、成交量單位或 corporate action 錯誤。
- point-in-time 基本面若使用事後修訂值，任何模型都會產生 look-ahead bias。
- 橫截面 universe 必須避免 survivor bias，且每個 prediction date 只能使用當時可交易的股票。

所以仍需 winsorization 規則、資料品質旗標、缺失原因分類與 point-in-time 快照。

---

## 4. GBDT 的標籤、特徵與驗證設計

### 4.1 標籤：預測「相對選股能力」，不是大盤漲跌

若週調倉／持有 5 個交易日，第一版標籤建議為：

1. 計算股票未來 5 日總收益。
2. 減去同期基準、行業或截面均值，得到相對收益；中性化方式必須與最終組合目標一致。
3. 在每個 prediction date 做截面 percentile rank 或常態分數轉換。
4. 可選擇將預估交易成本先從標籤扣除，或在 portfolio optimizer 中獨立懲罰 turnover；兩者不可重複扣除。

推薦第一版輸出 `factor_ml_gbdt`：每天每檔股票的截面分數，而不是宣稱精確的美元收益預測。

原始收益標籤並非必然錯誤，但它會混入 market beta、波動尺度與 regime。若目標是選出相對強勢股票，rank／residual label 更貼近策略目的；同時保留 raw-return benchmark，讓結果以實證比較決定。

### 4.2 第一版特徵

**現有截面特徵**：所有已驗證的 `factor_*` 欄位，包括動量、反轉、波動、Amihud、spread proxy、EMA distance、Alpha101 與 sector rank。

**多尺度信號**：

- 6-1、9-1、12-1 momentum；
- 5/10/21/63 日 reversal 與 volatility；
- volume、dollar volume、range 與 liquidity 的短／長窗口比率；
- factor rank 的一階變化與穩定度。

**regime 特徵**：

- QQQ/SPY 趨勢、波動與 drawdown；
- universe breadth（站上 50/200 日均線比例）；
- 截面 dispersion、平均相關性；
- 成長／價值、大小盤或 sector leadership 的可取得 proxy。

**正交資料源（後續）**：

- overnight 與 intraday return 拆分；
- 財報日前後事件旗標，避免在未知跳空風險前盲目換倉；
- point-in-time 基本面與 earnings surprise；
- 經授權且能重現的新聞／另類資料。

### 4.3 時間切分

禁止隨機 train/test split。採用 purged + embargoed walk-forward：

```text
訓練窗口 ── purge/embargo(至少等於 label horizon) ── 驗證窗口 ── 測試窗口
                       5 日標籤 → 至少隔離 5 個交易日
```

具體規則：

- 只用過去資料訓練，按月或季向前滾動。
- 超參數只在 train/validation 選擇；test 只使用一次。
- 重疊的 5 日標籤不可跨越資料切點。
- 所有 winsorize、標準化、缺失填補、特徵選擇都只能在 training window 擬合。
- 記錄 `model_version`、`train_start`、`train_end`、特徵 schema 與資料 hash。

### 4.4 第一版 GBDT 的保守限制

- 淺深度／少 leaves；
- 較大的 `min_data_in_leaf`；
- row/feature subsampling；
- L1/L2；
- early stopping；
- 限制每次調參的 search budget；
- 與等權因子、線性／Elastic Net baseline 同場比較。

在金融資料中，「模型更複雜」不是升級標準；**扣成本後、跨窗口穩定且風險不惡化**才是。

---

## 5. GRU／Transformer 直接吃量價序列？

### 5.1 可以，但不是直接吃未處理的價格

序列模型可學習手工因子沒有完整描述的路徑，例如波動聚集、量價先後順序、跳空後修復及不同時間尺度的形狀。建議輸入不是絕對 OHLCV，而是：

- adjusted OHLC returns／相對前收的 gap；
- log volume、dollar volume 與相對自身歷史的 z-score；
- high-low range、realized volatility；
- 相對 QQQ/SPY 與 sector 的收益；
- 缺失 mask、可交易性與 corporate-action flag。

候選序列長度可用 60、126、252 日做對照，標籤仍對齊未來 5 日截面 rank／residual return。

### 5.2 GRU 與 Transformer 的角色

| 模型 | 優點 | 主要風險 | 建議定位 |
|---|---|---|---|
| 小型 GRU | 參數較少，適合先驗證序列信號是否存在 | 長期依賴與跨資產互動較弱 | 第一個 deep baseline |
| Temporal CNN／TCN | 訓練穩定、並行、局部 pattern 明確 | receptive field 設計影響大 | 與 GRU 同場比較的低成本 baseline |
| Transformer | 可處理長距離與跨特徵互動 | 資料與算力需求高、過擬合及不穩定風險高 | 僅在小模型 baseline 通過後投入 |

現有資料的相鄰樣本高度重疊，有效樣本量遠小於「日期 × 股票數」的表面數字。深度模型必須小型化、強正則、按時間驗證，不能因為可生成大量 sliding windows 就認為資料已足夠。

### 5.3 不建議完全跳過手工因子

比較合理的三路架構是：

```text
手工因子 ───────────────┐
                         ├─ 截面校準 ─ Ensemble ─ 成本感知組合 ─ 風控 ─ 下單
GBDT(因子+regime) ───────┤
                         │
GRU/Transformer(量價序列)┘
```

手工因子提供：穩定 baseline、可解釋性、模型失效時的 fallback，以及 GBDT 的高品質結構化輸入。序列模型提供不同錯誤結構的邊際信號；只有當它的 OOS residual IC 在控制 GBDT 後仍為正，才值得進入 ensemble。

---

## 6. Ensemble：不是原始輸出的直接平均

### 6.1 正確的輸出對齊

GBDT、GRU 與 Transformer 的 raw score 量綱不同。每個 rebalance date 應先：

1. 使用同一個可交易 universe；
2. 將各模型輸出轉為截面 percentile rank 或 winsorized z-score；
3. 依 validation set 做方向、校準與缺失處理；
4. 再做加權平均。

公式示例：

```text
score_i,t = w_gbdt × rank(gbdt_i,t)
          + w_seq  × rank(sequence_i,t)
          + w_base × rank(rule_factor_i,t)
```

第一個 benchmark 可比較：

- 等權平均；
- `0.6 GBDT + 0.2 sequence + 0.2 baseline` 的保守先驗；
- 依 rolling OOS ICIR 配權，但向等權強烈收縮。

權重不可用 test period 倒推。若兩模型輸出高度相關，平均帶來的方差縮減有限；應同時監控 model score correlation、持倉 overlap 與 incremental IC。

### 6.2 為什麼 ensemble 通常更穩，但不是「必然更好」

只要模型錯誤不完全相關，平均可以降低預測方差；市場漂移時，多來源信號也較不容易同時失效。但差模型、同質模型或錯誤校準仍會拖累集成。因此新增模型需滿足：

- 單模型 OOS 淨成本績效不顯著惡化；
- 對既有 ensemble 有正的 incremental IC／Sharpe；
- 壓力期不增加不可接受的 tail risk；
- 換手與容量仍在限制內。

### 6.3 Meta-labeling：低風險的 ML 第二用途

除了直接選股，可讓 ML 判斷「既有信號這次是否值得交易」：

- 輸入：regime、breadth、波動、liquidity、crowding、基礎信號強度；
- 標籤：這次 top-N 信號扣成本後是否為正，或是否超過最低 alpha 門檻；
- 輸出：0–1 confidence；
- 用途：縮放 gross、提高換倉門檻或暫停某個 sleeve，而不是取代持倉排序。

這通常比一開始讓深度模型完全控制選股更容易驗證與治理。

---

## 7. 回測誠實化與交易成本

### 7.1 必須新增的成本分解

每次調倉先計算：

```text
turnover_t = Σ_i |target_weight_i,t - pretrade_weight_i,t|

cost_t = commission_t
       + half_spread_t
       + slippage_t
       + market_impact_t
       + borrow/financing_t
```

第一版可用固定 bps；第二版再按股票估計：

- `factor_spread` 作 spread proxy；
- dollar volume／ADV 作容量與 impact denominator；
- `factor_amihud` 作價格衝擊 proxy；
- 單筆交易占 ADV 比例作非線性 impact；
- 空頭另計 borrow availability 與 fee。

### 7.2 成本敏感度與容量

所有候選策略固定跑 0 / 5 / 10 / 20 bps（或依實際券商與成交紀錄校準）的雙向敏感度，輸出：

- gross vs net CAGR/Sharpe；
- 年化 turnover；
- 每次 rebalance 成本；
- alpha decay curve；
- 交易金額占 ADV；
- break-even cost bps。

「A 股雙邊千三」只能作其他市場的示例，不能直接套用 ancserAPX 的美股執行。ancserAPX 應以自身成交紀錄、spread 與流動性校準。

### 7.3 組合層加入換手懲罰

比事後硬截換手更一致的目標為：

```text
maximize  α̂ᵀw
        - λ_turnover × Σ|w - w_prev|
        - λ_risk × wᵀΣw
        - λ_concentration × concentration(w)
```

並加入 gross、單名、sector、beta、流動性與最小交易門檻。只有新信號的預期邊際 alpha 足以覆蓋成本時才換倉。

---

## 8. 對抗擁擠、bot 與 alpha 衰減

### 8.1 真正的 crowding gauge

目前 `crowding_shock_guard` 以 20 日平均 high-low range 過濾異常股票，較接近 tradeability／shock filter。建議保留原功能但重新命名，另建真正擁擠指標：

- top-N 持倉平均 pairwise correlation；
- sector／single-name／factor exposure concentration；
- top-N 名單跨模型與跨期 overlap；
- momentum 多空端的估值、波動與流動性 spread；
- 組合預計交易占 ADV 與 spread 惡化；
- 截面 dispersion 急降、相關性急升。

crowding 分數升高時，採漸進反應：提高換倉門檻 → 降低 momentum sleeve → 降 gross，而非第一時間全數清倉。

### 8.2 參數集成

把單一 `12-1` 動量擴為 6-1 / 9-1 / 12-1 的 rank 平均或交由受限 GBDT 組合。這降低單一窗口的時點風險，但仍需測試三者相關性，不能把高度相同的信號當成三份獨立 alpha。

### 8.3 錯開調倉

將週持倉拆成 5 個 tranche，每天只更新 1/5：

- 降低單一調倉日運氣；
- 降低市場 footprint 與集中滑價；
- 讓持有期更接近 rolling 5-day signal。

回測必須逐 tranche 保存 pre-trade weights 與成本，不能只把總權重除以五。

---

## 9. 多層風險管理升級

### 9.1 持倉層

1. **Inverse-vol／risk-adjusted weighting**：以實際可得、lagged 的波動估計取代純等權；設 weight floor/cap，避免低估波動的股票取得過大權重。
2. **單名上限**：core 與 winner-lock 合併後再次檢查，而非只在單一 sleeve 內限制。
3. **sector 上限**：例如 25%–35% 作研究範圍；超額權重按規則重新分配。
4. **流動性與容量**：min price、ADV、participation rate、spread 與預計清倉天數。
5. **財報事件風險**：財報前 N 日剔除、降權或單獨配置 event budget，需以 point-in-time calendar 實作。

### 9.2 組合層

1. **以實際權重估計波動**：用 lagged holdings return 或 `wᵀΣw`，取代 universe 等權平均收益的 volatility proxy。
2. **Drawdown governor**：示例研究網格：策略自身回撤 -10% → gross 上限 1.0x；-20% → 0.5x；恢復時採 hysteresis 與分級回升，避免來回切換。
3. **Beta 上限**：以 rolling SPY/QQQ beta 或 covariance estimate 控制 net beta；估計不足時使用保守 fallback。
4. **風險預算**：限制 sector contribution、single-name contribution、top-5 concentration 與 gross leverage。
5. **相關性危機模式**：當平均相關性與組合波動同時急升，額外降低 gross。

### 9.3 市場 regime 與極端情境

1. **三檔狀態**：breadth 先惡化時降槓桿；QQQ/SPY 跌破慢速趨勢再進 cash；恢復使用不同門檻形成 hysteresis。
2. **固定壓力窗口**：至少固定測 2020-03、2022 熊市、2024-08 波動衝擊，另加全樣本 rolling worst-N windows，避免只對已知事件調參。
3. **尾部對沖 sleeve**：OTM put／collar 可研究為 1.5x 帳戶的結構性保護，但不是免費保險。必須納入 premium bleed、roll、skew、到期與 gap risk；在完整期權資料與執行能力建立前不可把它描述成「保證不爆倉」。

---

## 10. MWU 升級

目前 MWU 以單日 rank IC 立即乘法更新，容易被噪音驅動；建議：

1. 計算與持有期一致的 5 日 forward rank IC。
2. 對 IC 使用 63 日 EWMA，且只使用已完整實現的 label。
3. 將動態權重向等權或經濟先驗收縮：

```text
w_final = ρ × w_dynamic + (1 - ρ) × w_equal
```

4. 設每日／每次 rebalance 最大權重變動、turnover penalty 與 minimum evidence threshold。
5. 讓 `window` 真正控制估計窗口，並保存 OOS 權重歷史供 parity 與審計。

這一步不需要先上 ML，卻能直接降低 factor-weight whipsaw。

---

## 11. 模型漂移、重訓與生產治理

### 11.1 重訓節奏

- 每月產生 challenger；每季作為第一版正式升級節奏。
- 若資料量／運算允許，可比較 1、2、3 個月，但頻率由 OOS 淨成本決定。
- 不因短期失效自動重訓並立即上線；先完成 drift diagnosis 與 champion/challenger gate。

### 11.2 每日／每期監控

- rank IC、ICIR、top-bottom spread；
- score 分布、missing rate、feature PSI／分布漂移；
- 模型間 score correlation、持倉 overlap；
- turnover、spread、slippage、ADV participation；
- realized vs forecast volatility、beta、sector concentration；
- gross/net P&L attribution：signal、cost、financing、risk overlay。

### 11.3 回退與版本

每筆預測至少保存：

```text
as_of_date, symbol, score, model_name, model_version,
train_start, train_end, feature_schema_hash, data_snapshot_id
```

訓練在離線流程完成；回測與實盤只讀取當時已存在的 frozen prediction。若模型檔、schema、資料日期不一致，回退到上一個 champion 或手工因子 baseline，不可現場重新訓練。

---

## 12. 評估指標與上線門檻

模型預測指標：

- OOS rank IC、ICIR；
- decile／top-N spread；
- calibration 與分箱單調性；
- incremental IC（控制現有因子與既有模型後）；
- 不同 regime、sector、size、liquidity 分組穩定性。

投資組合指標：

- net CAGR、Sharpe、Sortino、Calmar；
- max drawdown、回撤恢復時間、CVaR；
- turnover、break-even cost、capacity；
- beta、gross/net exposure、sector/top-5 concentration；
- 最差窗口與固定壓力期表現。

建議上線 gate：

1. 所有結果均為 purged walk-forward OOS；
2. 主要成本情境下仍有正的增量價值；
3. 不以顯著惡化 drawdown／CVaR 換取少量平均收益；
4. 至少跨多個 market regime，且不是單一年份貢獻全部績效；
5. parity、資料版本、模型版本與 fallback 測試全部通過；
6. 先 paper／shadow，再以小額 gross 漸進上線。

---

## 13. 建議實施順序

### Phase 0 — 誠實回測與基準（最高優先）

- 加入 turnover、固定／動態成本與成本敏感度掃描；
- 建立 purged walk-forward harness；
- 固定壓力窗口與 regression tests；
- 輸出 gross/net attribution 與容量指標。

**完成定義**：每個策略都能回答「扣除什麼成本、換手多少、在哪些窗口失效」。

### Phase 1 — 不依賴 ML 的風控修復

- MWU 改 5 日 IC + 63 日 EWMA + shrinkage；
- inverse-vol、單名與 sector cap；
- target-weight portfolio volatility；
- drawdown governor、beta/gross guard；
- 現有 range guard 更名，新增真正 crowding gauge。

### Phase 2 — GBDT 截面模型

- 建 point-in-time dataset 與 5 日 rank/residual labels；
- LightGBM 與線性／等權 baseline 比較；
- 只生成 OOS `factor_ml_gbdt`；
- frozen prediction 接入 parity-critical portfolio path。

### Phase 3 — 小型序列模型

- GRU／TCN baseline；
- 確認 incremental IC、成本與模型相關性；
- 通過後才測小型 Transformer。

### Phase 4 — Ensemble 與 meta-labeling

- rank 校準後平均；
- OOS 決定權重並向等權收縮；
- confidence 控制 gross／換倉門檻；
- champion/challenger 與 drift automation。

---

## 14. 本版架構決策

1. **採用 GBDT 優先，序列模型後置。**
2. **不刪除手工因子；它們同時是 baseline、GBDT 特徵與 fallback。**
3. **Ensemble 平均的是校準後的截面分數，不是原始模型輸出。**
4. **ML 預測以新的 factor score 接入共用 portfolio module，維持回測／實盤 parity。**
5. **交易成本與風控 gate 優先於模型架構升級。**
6. **任何「業界普遍使用某模型」的說法只作觀察，不作投資決策證據。**
7. **最終目標是可維護的信號系統：可重訓、可監控、可回退、可審計。**

---

## 15. 參考資料

- Shihao Gu, Bryan Kelly, Dacheng Xiu, “Empirical Asset Pricing via Machine Learning,” *The Review of Financial Studies*, 33(5), 2020, 2223–2273. DOI: [10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009)
- [NBER Working Paper 25398 — Empirical Asset Pricing via Machine Learning](https://www.nber.org/papers/w25398)
- [LightGBM 官方文件](https://lightgbm.readthedocs.io/en/latest/)

---

## 16. 下一個可執行交付

下一個開發迭代應只做 **Phase 0：交易成本模型 + turnover 記錄 + 0/5/10/20 bps sweep**。這是後續 MWU、GBDT、GRU／Transformer 與 Ensemble 能否被可信評估的共同地基。

---

## 17. 行業中性化實證更新（2026-07-14）

已在隔離的 Research v2 中完成 immutable-by-runner、hash-verifiable 的 Corrective v3。v1／v2 保留作審計歷史，但已被 v3 取代。這是 2026-07 static broad-sector 研究，不是歷史 point-in-time GICS 或市值中性化：資料沒有歷史 market cap／shares，因此沒有用價格、ADV、成交額或今日市值倒灌歷史。

測試固定既有 OOS 分數與投資組合設定，只改變訊號轉換：

- 每日 sector dummy OLS residual（等價於行業內減均值）；
- 每日行業內 Z-score；
- 每日行業內 percentile rank；
- Claude #1 另區分 factorwise（先處理 Momentum／RSI，再固定 70/30 合成）與 scorewise（合成後處理）。

驗證範圍包括 selection OOS 2023-11-13 至 2025-11-14、已開封的 2026 描述期、8 個 walk-forward folds、Claude #1 正常成熟週期的 prior-close weekly proxy、另行標示的 5-session offset 診斷、Hybrid 的 21 個 monthly offsets、0/5/10/20 bps reference 成本敏感度，以及 5,000 次 circular moving-block paired bootstrap。Claude proxy 的一般定義是「前一個完整市場日收盤訊號→該週最後市場日 official-open 成交」；130 次中 125 次在週五、5 次在週四執行，休市安排也可能形成週三→週五或週三→週四。它不是 exact live parity：Main config 有 510 檔，研究 complete-case universe 為 480 檔，而且 official open 不能精確代表 09:35 fill。

主要結果：

1. **中性化確實降低行業暴露與平均持倉集中。** Claude weekly proxy 的 selection 平均最大行業占 gross 由 34.70% 降至 factor-residual 的 25.80%，有效行業數由 4.80 升至 6.16；within-sector rank 為 22.78%／7.05。每檔仍是約 7.5% gross 的 Top-20 等權，改變的是入選股票與行業分布，不是 sizing 規則。
2. **訊號層沒有證明 alpha 增強。** Claude raw Rank IC 0.0250，factor residual 0.0156（差 -0.0094）；Hybrid raw 0.0385，residual 0.0267（差 -0.0118，Newey–West t -2.19）。Hybrid raw 的 between-sector IC 在 selection 為正，說明被刪掉的不全是污染，其中包含當期有效但可能漂移的 sector timing。
3. **Claude score-level residual 值得新的 shadow-forward，但不能上線。** 在 10 bps weekly proxy 中，selection Sharpe 由 1.039 升至 1.311、CAGR 37.59% 升至 45.74%、MaxDD -36.85% 改善至 -35.48%、gross turnover 147.47 降至 132.91；0–20 bps reference sweep 都優於 raw。然而其 Rank IC 仍下降 0.0052，paired bootstrap 年化算術報酬差為 +4.20%，95% CI [-12.53%, +20.52%]，而且 2026 已開封。它是新假設，不是已證明升級。
4. **三個預先指定 residual 主 gate 全部未通過。** Claude factor residual 雖把 weekly proxy Sharpe 提高 0.236，Sector R² 只降低 72.29%且 IC 損失超標；Hybrid true-no-risk residual 的 21-offset 中位 Sharpe 1.243→1.165；legacy champion 行為則 1.126→1.003。Hybrid residual 可改善 MaxDD與換手，但明顯犧牲 CAGR／IC。
5. **舊冠軍已得到精確重現。** Selection final equity 150,612.4486、CAGR 22.724%、Sharpe 1.571817、MaxDD -19.332%、gross turnover 38.4202、總成本 7,756.9988 均通過 hard parity。審核也確認舊 artifact 所謂 `risk_variant=none` 實際含隱式 trend overlay；Research v2 現改為必須顯式 `trend_filter=True`。
6. **行業限制、中性化與市值中性化是三件事。** Long-only Top-N 的 score Sector R² 接近零，不等於持倉相對 benchmark 完全行業中性；硬 sector cap 也可能切掉 sector momentum。市值中性化必須等 PIT market cap／shares 資料到位後才測 joint OLS。

目前決策：**不修改 Main daily 的 Claude #1，不把任何中性化版本直接上線。** 保留 `scorewise sector residual` 作為 shadow challenger；下一步預先鎖定 partial shrinkage `score_final=(1-λ)·raw+λ·residual` 的少量 λ，使用新的 paper／forward 資料判斷，而不是繼續調已看過的 2026。

完整可重現結果：[`research_v2/runs/20260710_full_v1/neutralization_study_v3/report.md`](../research_v2/runs/20260710_full_v1/neutralization_study_v3/report.md)。

---

## 18. PE 100% 排序、sector-relative 市值篩選與平均 sector 配置（2026-07-14）

本次新增的是隔離在 `research_v2` 的研究策略，沒有接入 Main daily，也沒有修改 `config/live_strategy.json`。正式定義如下：

1. 只保留有效且大於零的 trailing PE；虧損公司的負 PE 不視為「便宜」。
2. 在每個 sector 的 contemporaneous 可交易股票內計算 market-cap 中位數，只保留市值不低於該 sector 中位數者。
3. 對所有通過者按原始 PE 由低至高做全市場 Top-22；PE 是唯一選股排序信號。
4. 選股完成後，才把已入選 sector 的 gross budget 設成相同，再在 sector 內等分。這不會為湊齊 sector 而改寫 Top-22。

第 2 步是 hard eligibility filter，**不是聯合回歸**。真正的 joint OLS challenger 是 `earnings_yield ~ z(log_market_cap) + sector dummies` 並使用 residual；它會改變 value signal，因此不能再稱為「原始 PE 100% 排序」。兩種方法已分開輸出，沒有混用。

### 18.1 當前截面結果

使用同一份 2026-07-14 凍結快照重播，510 檔中有 478 檔 positive PE、507 檔 positive market cap、478 檔兩者同時有效。全市場中位市值篩選有明顯 sector retention bias：Energy 72.00%、Technology 64.94%、Communication 61.90%，但 Real Estate 30.00%、Utilities 32.14%、ConsStaples 33.33%。改成 sector 內中位數後，各 sector 約保留 48%–52%。

| 當前截面版本 | Top-N | Sector 數 | 中位 PE | 中位市值 | 最大 sector / gross | 有效 sector 數 |
|---|---:|---:|---:|---:|---:|---:|
| 原始 PE、每股等權 | 22 | 9 | 8.08 | $23.65B | 31.82% | 5.90 |
| 原始 PE、已入選 sector 等 gross | 22 | 9 | 8.08 | $23.65B | 11.11% | 9.00 |
| sector 中位市值篩選、每股等權 | 22 | 10 | 12.54 | $82.59B | 40.91% | 4.65 |
| sector 中位市值篩選、已入選 sector 等 gross | 22 | 10 | 12.54 | $82.59B | 10.00% | 10.00 |
| joint OLS value residual、每股等權 | 22 | 10 | 8.08 | $21.77B | 22.73% | 7.81 |

重要發現：sector-relative 市值篩選成功消除「同一絕對市值門檻對不同行業不公平」的問題，但它本身不是 sector concentration control。本次它令每股等權 Top-22 的最大 sector 權重由 31.82% 增至 40.91%；真正把最大 sector gross 壓到 10% 的是後置權重層。市值篩選後與 raw Top-22 只重疊 5 檔，表示這不是微小調整，必須有多期 PIT 回測才能升級。

目前 Main Claude #1 配置沒有 `risk_management` 區塊，所以現行 daily 沒有明示 market-cap filter，也沒有啟用 liquidity filter 或 sector balance。既有 `liquidity_filter` 若啟用，實際是 price ≥ $5 與 20 日平均 dollar volume ≥ $20M，**不是市值篩選**；最新截面通過率由 Real Estate 26.67%、Materials 27.27%、Utilities 32.14% 到 Energy 79.17%、Technology 78.95%、Communication 76.19%，因此全球固定流動性門檻本身也有 sector bias。

### 18.2 單次 forward 敏感度診斷

唯一較早的 frozen fundamental snapshot 是 2026-02-13，但沒有 PIT market cap／shares、filing-availability provenance 或時區。因此只能測「相同 low-PE Top-22 的每股等權 vs 已入選 sector 等 gross」，不能測 sector 中位市值規則，也不能稱為完整歷史回測。為避免同日洩漏，於下一完整交易日 2026-02-17 open 建倉，持有到 2026-07-10 close，共 100 個共同交易日；假設每邊 10 bps。

| One-shot 版本 | 淨總報酬 | 短樣本年化波動 | Sharpe（0% rf） | MaxDD |
|---|---:|---:|---:|---:|
| Low-PE Top-22 每股等權 | +0.14% | 12.16% | 0.09 | -5.09% |
| 同一 Top-22、sector 等 gross | -2.97% | 13.62% | -0.49 | -6.66% |
| 468 檔 positive-PE 等權內部基準 | +5.41% | 11.80% | 1.18 | -7.65% |

Sector 等 gross 把 Financials 由 40.91% 降至 10%，但有六個 sector 只有一檔入選，令這些股票各自升至 10%。所以它降低 sector 集中，卻提高單名集中；本次報酬落後 raw 3.11 個百分點，波動與 MaxDD 也較差。0／5／10／20 bps 每邊時，raw 分別為 +0.34%／+0.24%／+0.14%／-0.06%，sector 等 gross 為 -2.78%／-2.87%／-2.97%／-3.17%。單一路徑不足以證明長期無效，但足以否決「等 sector 必然更安全」以及直接升級 Main daily。

目前決策：保留 PE 純排序與 sector-relative size screen 作 shadow-forward 研究元件；硬性等 sector 不升級。下一個較合理的 challenger 是 sector cap 或 volatility-scaled sector budget，並限制單名權重；待取得具 filing timestamp 的 PIT EPS、shares、market cap 與歷史 sector effective dates 後，再做 weekly walk-forward、成本／容量與 0/5/10/20 bps 測試。

完整可重現結果：

- [當前 PE／市值截面 Corrective v3](../research_v2/runs/20260714_pe_sector_current_v3/report.md)
- [2026-02-13 單次 forward Corrective v2](../research_v2/runs/20260213_pe_sector_forward_v2/report.md)

---

## 19. Delta-PE 與 sector PE 共同重估研究（2026-07-14）

本次在 Research v2 增加 Delta-PE shadow factor，Main daily 未修改。只有兩張基本面快照：legacy 2026-02-13 與 current 2026-07-14 LA／2026-07-15 UTC；ticker 交集 499 檔，兩端均有正 PE、正 EPS 與 sector 的完整樣本 464 檔。由於 current Delta-PE 必須等 7 月快照才知道，2 月至 7 月價格只能作 contemporaneous attribution，不能作預測回測；而且兩點相隔約五個月，尚不能稱為「突然」變化。

定義採固定 matched constituents 的 robust sector median：

```text
d_i = log(PE_current / PE_previous)
s_g = median(d_i | sector g)
r_i = d_i - s_g
```

沒有使用 arithmetic mean sector PE，因為接近零的 EPS 會把平均 PE 拉到極端；舊快照也沒有 PIT market cap／aggregate earnings，無法計算真正 cap-weighted sector PE。保留兩個不混淆的 AND：

- literal AND：`min(rank(stock d_i), rank(sector s_g))`，要求 stock `d_i>0`、sector `s_g>0` 及 sector positive breadth ≥50%，精確對應「股票與行業 PE 同升」。
- relative AND：`min(rank(sector s_g), within-sector rank(r_i))`，另要求 `r_i>0`，所以個股還必須跑贏自己的 sector 中位數。

Naive Delta-PE 不能直接視為正向信號。Raw Top-22 的中位 PE 上升 154.55%，但 reported EPS 中位下降 48.20%；68.18% 的股票 EPS 跌逾20%。MOS 的 PE 約由 7.8 升至163.9，主要原因卻是 EPS 約由3.86降到0.14。類似污染亦出現在 FANG、GPC、NRG、COIN、APO 等，因此新增初步 quality guard：reported EPS 不得下降超過10%，implied price change 必須為正。這個 guard 尚未經 corporate-action share-basis 認證，只是敏感度，不是 production-approved 財報品質因子。

實際 sector matched-median Delta-PE 只有 Energy +4.36%（breadth 54.17%）及 Utilities +0.39%（57.14%）為正，其餘9個 sector 均為負。因此：

| Shadow variant | 候選數 | Sector 數 | 最大 sector（每股等權） | 中位 PE 變化 | EPS 跌逾20% |
|---|---:|---:|---:|---:|---:|
| Naive stock Delta-PE | 22 | 7 | 31.82% | +154.55% | 68.18% |
| Literal stock+sector AND | 22 | 2 | 54.55% | +17.76% | 13.64% |
| EPS-guarded literal AND | 21 | 2 | 66.67% | +7.81% | 0.00% |
| EPS-guarded relative AND | 18 | 2 | 66.67% | +10.75% | 0.00% |

EPS-guarded literal AND 目前為21檔，仍只集中 Energy／Utilities。改成兩個 sector 等 gross 可把 sector 權重改成50%／50%，但不會創造真正跨行業分散。因此它只被凍結為 forward shadow：最早有效成交點是 current snapshot 可用後的下一個市場開盤；待新資料到位後固定測5／21／63日 forward Rank IC、相對 sector IC、turnover、sector cap 與0／5／10／20 bps。至少累積36–60個 PIT monthly snapshots 後，才能用 rolling MAD z-score 定義「突然暴漲」。

目前決策：Delta-PE 值得保留為研究特徵，但 raw 高 Delta-PE 方向未獲批准，兩個 AND 版本亦不得因同期價格相關漂亮而宣稱 alpha。優先研究 `price-led rerating` 與 `EPS-collapse expansion` 的分離，sector surge 只作受限 tilt，必須配 sector／單名上限。

完整可重現結果：[Delta-PE two-snapshot Corrective v2](../research_v2/runs/20260714_delta_pe_study_v2/report.md)。

---

## 20. Production safety、成本與 UI 可追溯性更新（2026-07-14）

本次沒有把 GBDT／GRU／Transformer、PE、Delta-PE 或中性化 challenger
直接接入 Main daily；Production MODEL registry 目前只暴露已實作的
`factor_composite`，未知或未完成模型會被拒絕，不能在 UI 中「看似可選但實際
fallback」。本次真正改變的是資料、執行、風控、成本與審計地基：

1. **Live 下單改成 fail-closed 單一路徑。** Windows task 與網站 Force 都走
   `broker eligibility → sync → manifest gate → physical parquet gate → stateful
   risk → targets → exact as-of/effective-universe gate → OMS`。每個 effective
   symbol 與 QQQ／SPY 必須在最新完整 NYSE session 有唯一且合法的 OHLCV 實體
   row；coverage／freshness 固定 100%，live config 不能降級。Inactive、不可交易、
   非 fractional 或 broker 明確 not-found 的代號會帶原因排除；任何不確定的 API
   回應仍 fail-closed。
2. **修復資料截止日 P0 bug。** Daily bar 即使 timestamp 是 04:00，也會包含在
   同一 `end_date`；Parquet 與 manifest 改為跨程序鎖定及 atomic replace，降低
   web sync 與 Windows runner 同時寫入的損壞／lost-update 風險。
   2026-07-14 的真實 dry sync 中，configured 510 檔有 506 檔 active／tradable／
   fractionable；ANSS、CTRA、HOLX 為 inactive，BK 為 broker not-found。506 檔加
   QQQ／SPY 共 508 個實體檔的 freshness、physical coverage 與 history coverage
   均為 100%，全程沒有進入策略或 OMS。
3. **Daily risk 真正在非調倉日執行。** 200EMA 跌破退出、20EMA 重入、vol／
   throttle 每日檢查；一般縮槓桿只縮放 broker 真實 drifted holdings，不每日重選
   alpha，只有 regime transition／re-entry 才重算。部分清倉會依隔日真實剩餘持倉
   重試。
4. **審計不再覆蓋。** 每次資料 gate、目標、order plan、submitted／failed、
   broker order/fill 與 account state 都有 append-only JSONL；tracker 使用 broker
   `equity-last_equity` 記錄 day P&L。網站的 Final P&L 是可用 fills 的 gross FIFO
   realized gain，並明示 fees、cashflows、dividends 與 pre-history lots 的限制。
5. **Windows 安裝自動化。** `install.bat` 會安裝 09:25 America/New_York 的 task；
   California 對應 06:25。`daily.bat` 同步等待 Python 完成、保存 exit/output，並
   每日重算 DST。程式內 daemon 與 Windows one-shot 的差異是：前者必須常駐，
   後者由 Windows 喚醒程序；實際交易安全邏輯相同。帳戶級跨程序 execution
   lock 阻止網站連點、daemon 與 Windows task 對同一帳戶重複送單；sync 若拖過
   09:29，non-force runner 會在 OMS 前再次阻擋。Manual Force 只可越過 cadence／
   時窗，不能越過資料、as-of 或 broker eligibility gate。
6. **回測成本分離。** `commission_bps`、`slippage_bps`、
   `regulatory_sell_bps` 分開按實際 traded notional 計算。預設 broker commission
   為 0 bps、slippage 為單邊 5 bps；監管費預設 0，因實際 SEC／TAF／CAT 並非
   一個永久固定 bps，需按當期 fee schedule 或成交資料校準。
7. **設定與歷史可追溯。** 任一 edit 顯示 `UNSAVED MODIFIED SET`；full preset
   的 sleeve 定義可保留並讓 Top-N／hold／leverage／risk／universe 真正 override，
   改 factors／MODEL／MWU 才切換 custom scoring。自訂 factor weights 已接入 live，
   不再默認等權忽略。新完整結果、75筆 legacy backtest summary 與有效 tracker
   snapshots 會在同一 history 清單重載。MWU dashboard／snapshot 優先顯示該次
   真正計算的動態權重，而不是 config 初始值；手動 Fetch 與 Coverage 亦跟隨畫面
   所選 Universe。Browser title 與 header 均為 `ancserAPX 1.0.0`。

8. **驗證結果。** 隔離式測試為 research safety 16 項加其餘 128 項，共 144 項
   全通過；JavaScript syntax、Python compile、diff check 及 Claude #1 同一 as-of
   的 backtest/live target parity 均通過（Top-20、gross 1.5、最大權重差 0）。

仍有一項已知 cadence 語義差異：現有 backtest 的 5D 是由模擬起點每 5 個
session，live 的 5D preset 是每週最後 NYSE session。因兩者錨點不同，不能把
「相同因子 target parity」誤稱為整條成交路徑完全相同；修正它會重定義既有
歷史回測，故本次沒有在未重跑全部 benchmark 的情況下偷偷改寫結果。

### 20.1 固定延遲 5 sessions 的實證

Claude #1、截至 2026-07-13、單邊 5 bps：

| 指標 | 正常完整訊號 | 延遲 5 sessions | 差異 |
|---|---:|---:|---:|
| Final equity | $653,565 | $592,306 | -$61,259 |
| CAGR | 45.48% | 42.65% | -2.83pp |
| Sharpe | 1.18 | 1.11 | -0.07 |
| MaxDD | -31.95% | -36.34% | -4.39pp |

252 次調倉平均 Top-20 只重疊 10.71 檔（53.53%），最差只重疊 1 檔。因此
「延遲一週」不是每次完全不同，但平均約一半持倉已變，且回撤與長期複利明顯
惡化。零摩擦 CAGR 50.82%，加入 5 bps 後為 45.48%；年化 gross turnover 約
72.5x，說明 turnover penalty 具有研究價值，但它會改變持倉決策，不能只在
Production OMS 單邊加入。本次只加入成本觀測，未把 penalty 擅自上線。

ADV participation cap 同樣未加入 Production：目前約數千美元、流動性大盤股時
通常不會 binding，但資金放大或進入小型／低流動性股票後很重要。正式接入前需
同時完成 ADV point-in-time 資料、partial fills、order slicing、未完成訂單狀態及
backtest/live parity。

完整結果：

- [資料延遲實驗](experiments/claude1_signal_staleness/report.md)
- [成本、turnover penalty 與 participation cap](backtest_execution_costs.md)
