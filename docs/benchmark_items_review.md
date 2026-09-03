# Benchmark items — for review

Everything below is **draft and unvalidated**. The point of this page is to
let a domain expert check the question, the dataset it was paired with, the
query, and the answer — and reject any of them. Method: [methodology.md](methodology.md).

## 1. Verified items — Tier A (1)

Gold SQL executed against the live table; the answer below is the pinned
snapshot. Re-running the query reproduces it, and a perturbed query fails.

### `cbs-0001` — Did the post-WWII baby boom in the Netherlands compensate for the births that did not occur during WWII?

*Original:* Heeft de naoorlogse geboortegolf in Nederland de tijdens WOII niet plaatsgevonden geboorten gecompenseerd?

*Clarified (what the query actually pins down):* Compare average annual live births (x1000) for 1935-1939, 1940-1944 and 1946-1950, and report the 1946-1950 surplus relative to five pre-war years.

*Answer type:* `yes_no` · *publisher:* CBS

**Dataset(s):**
- [`85524NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85524NED/table) — Bevolking, huishoudens en bevolkingsontwikkeling; vanaf 1899

**Gold SQL**

```sql
WITH b AS (
  SELECT CAST(Perioden_label AS INTEGER) AS year, value AS live_births_thousands
  FROM t_85524NED
  WHERE measure = 'Levend geboren kinderen' AND unit = 'x 1 000'
    AND Perioden_label ~ '^[0-9]{4}$'
)
SELECT
  ROUND(AVG(CASE WHEN year BETWEEN 1935 AND 1939 THEN live_births_thousands END), 1) AS pre_war_1935_39,
  ROUND(AVG(CASE WHEN year BETWEEN 1940 AND 1944 THEN live_births_thousands END), 1) AS war_1940_44,
  ROUND(AVG(CASE WHEN year BETWEEN 1946 AND 1950 THEN live_births_thousands END), 1) AS boom_1946_50,
  ROUND(SUM(CASE WHEN year BETWEEN 1946 AND 1950 THEN live_births_thousands END)
      - 5*AVG(CASE WHEN year BETWEEN 1935 AND 1939 THEN live_births_thousands END), 1) AS surplus_vs_prewar
FROM b
```

**Answer**

| pre_war_1935_39 | war_1940_44 | boom_1946_50 | surplus_vs_prewar |
|---|---|---|---|
| 174.2 | 197.2 | 253.0 | 1659.0 |

<sub>retrieved 2026-09-03 · table last updated 2025-10-17T00:00:00+02:00</sub>

*Question attested in:* <https://papers.tinbergen.nl/>

**Review:** is the question well-posed? is this the dataset you would use? does the SQL express the question? is the answer right?

---

## 2. Deferred items — Tier B, microdata (1)

Gold SQL is authored against the documented CBS register schemas and is
**not executable** outside the CBS secure environment. Kept so the microdata
questions stay in the benchmark as a *write the query you would run* task,
to be validated with CBS.

### `cbs-micro-0001` — What is the difference in labour market participation between educational groups?

*Registers:* HOOGSTEOPLTAB, POLIS

```sql
-- Authored against the documented CBS microdata register schemas.
-- NOT executable outside the CBS secure environment; to be validated
-- with CBS as a partner. Kept so the microdata arm is not lost.
SELECT o.opleidingsniveau, COUNT(DISTINCT p.rinpersoon) AS employed
FROM POLIS p JOIN HOOGSTEOPLTAB o USING (rinpersoon)
GROUP BY 1
```

---

## 3. Candidate pool from triage

*Triage has not been run against the current corpus.*

## How to review

1. **Section 1** — check each verified item end to end. A wrong dataset or a
   query that answers a different question is the failure to look for.
2. **Section 3a** — check the question↔dataset pairing before SQL is written;
   this is where the earlier document-level attribution bug did its damage.
3. **Section 3c** — check we are not throwing away good questions.

*Regenerate:* `python scripts/make_benchmark_review.py`
