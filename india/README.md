# 🇮🇳 India digital payments — PhonePe Pulse

Phase 1 of adding India as a source alongside Eurostat and CBS. PhonePe Pulse
transactions, registered users and registered merchants, at country / state /
district level, quarterly **2018 Q1 → 2026 Q2**.

| | |
|---|---|
| source | [github.com/PhonePe/pulse](https://github.com/PhonePe/pulse), branch `main` |
| commit mirrored | `257bf38e3095020f9f516c4c72c10bdc348781b1` |
| licence | **CDLA-Permissive-2.0** ([text](https://cdla.dev/permissive-2-0/)) — free reuse incl. commercial, attribution notice retained |
| attribution | PhonePe Pulse (https://www.phonepe.com/pulse/), CDLA-Permissive-2.0 |
| coverage | 1 country · 36 states/UTs · 783 districts · 34 quarters |

Branch `main`, never `master` — master is stale (stops at 2024, still carries the
retired `insurance` category).

## Reproduce

```bash
python -m india.pulse_fetch         # mirror 11,184 JSON files -> data/raw/india/ (~2.4 min)
python -m india.pulse_transform     # flatten -> data/processed/india/ (~30 s)
python -m india.validate_pulse      # 11 acceptance checks, exit 1 on failure
```

Both steps are resumable and idempotent: the fetch skips cached files, and the
transform is byte-identical on re-run. The raw mirror is git-ignored (13 MB);
only `data/processed/india/` (7.0 MB) is committed, per the
[data policy](../data/README.md).

The download is driven by the repo's **git tree listing**, not by a
year × quarter × state product. Those products are mostly wrong — 2026 has only
Q1–Q2, and `merchant` starts later than `transaction` — so enumerating what the
repo actually contains turns a missing quarter into a fact rather than a 404.

## Output

```
data/processed/india/
  fact_transactions.parquet   31,613 rows  geo_id, year, quarter, category, instrument, txn_count, txn_amount_inr
  fact_users.parquet          27,880       geo_id, year, quarter, registered_users, app_opens
  fact_merchants.parquet      25,949       geo_id, year, quarter, registered_merchants
  fact_top.parquet            32,173       scope_geo_id, metric, rank_within, entity_raw/_norm, count, amount_inr
  dim_geo.parquet                820       geo_id, level, state_slug, state_name, district_name
  dim_time.parquet                34       year, quarter, quarter_start, quarter_end
  dim_category.parquet             4       category, category_norm
  source_meta.json                         source, commit, licence, counts, caveats
  csv/                                     7 tidy exports for pandas / the chat agent
```

`geo_id` is a synthetic stable key: `IN`, `IN-<state_slug>`,
`IN-<state_slug>-<district>`.

> ⚠️ **A district is a (state, district) pair, never a bare name.** Three names
> are shared across two states each — `bilaspur` (Chhattisgarh / Himachal
> Pradesh), `hamirpur` (Himachal Pradesh / Uttar Pradesh), `pratapgarh`
> (Rajasthan / Uttar Pradesh). Counting names alone gives 780 and looks like
> three missing districts. Always group on `geo_id`.

## The source has three JSON shapes, and they drift

Every file is `{success, code, data, responseTimestamp}`; the shape of `data`
depends on **both** the view and the metric:

| view / metric | shape |
|---|---|
| `aggregated` / `transaction` | `data.transactionData[]` → `{name, paymentInstruments[]}` |
| `aggregated` / `user`,`merchant` | `data.aggregated` → `{registeredCount}` |
| `map` / `transaction` | `data.hoverDataList[]` → `{name, metric[]}` — a **list** |
| `map` / `user`,`merchant` | `data.hoverData` → `{name: {...}}` — a **dict** |
| `top` / `transaction` | `data.{states,districts}` → `{entityName, metric}` |
| `top` / `user`,`merchant` | `data.{states,districts}` → `{name, registeredCount}` |

Measured over all 11,184 files at this commit, rather than assumed — four points
where the data departs from what the hand-off brief expected:

1. **`amount` never appears in `aggregated/transaction`** (0 of 3,773 entries).
   Rupee amounts live **only** in the `map`/hover view. So category rows
   (country/state) carry a count with a null amount; the `TOTAL` rows sourced
   from `map` carry both.
2. **`usersByDevice` and `appOpens` appear in zero files.** The brief expected
   both in older years; the restructure removed them.
   `fact_users_by_device.parquet` is therefore **not written** — an empty table
   would imply the measure exists and is zero. `app_opens` is kept as an
   all-null column, and the parser still handles `usersByDevice` if it returns.
3. **Only three transaction categories exist, in every year:** P2P, Retail,
   Utility. The older "Recharge & bill payments" / "Financial Services" buckets
   are gone from `main`. The raw `name` is stored verbatim so a new category is
   never silently bucketed.
4. **`top/<metric>/.../state/...` has `states: null`**, not a missing key — a
   state file ranks its own districts. Null, hence `or []` throughout.

## Caveats for anyone reporting a number from this

- **PhonePe only** — roughly half of UPI, not the whole market. Not a national
  payments statistic. Market share needs an RBI/NPCI denominator (Phase 2).
- **District rows are `TOTAL` only.** The P2P/Retail/Utility split exists at
  state and country level, never below.
- **Quarterly**, not monthly. **No pincode level** — removed in the restructure.
- Amounts are **INR**, as published, unadjusted for inflation.

## Validation

`india.validate_pulse` asserts the brief's §4 criteria; all 11 pass at the
mirrored commit. The two load-bearing numbers were measured independently of
this code, so they catch a flattener that silently drops or double-counts:

- **26,616** district-transaction rows · 783 districts · 36 states/UTs · 2018Q1–2026Q2
- **38,664,285,691** district transaction counts in 2026 Q2, equal to the
  national aggregate (P2P 11,889,781,621 + Retail 24,691,181,013 +
  Utility 2,083,323,057)

Check 3 reconciles across **two views** — district counts from `map`/hover
against the national total from `aggregated` — so agreement means the two shapes
were parsed consistently, not merely parsed.

## Not done (Phases 2–4, deliberately out of scope)

RBI DBIE series, data.gov.in catalogue, NPCI monthly stats, and the LGD /
boundary / Census-population geo bridge. See the hand-off brief; Phase 2 needs a
human decision on scope before it starts.
