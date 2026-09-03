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

Data availability is taken from the **classifier** pass (`data_needed`).
The triage `benchmark_status` field is deliberately ignored — see
[methodology §9](methodology.md#9-known-defects-and-how-they-were-caught).

| data needed (classifier) | n |
|---|---:|
| `microdata` | 1,202 |
| `other_source` | 1,054 |
| `public_aggregate` | 1,008 |
| `none` | 12 |

### 3a. Candidates awaiting gold SQL (5)

Three independent signals agree: the classifier called it answerable from
public aggregates, it marked it verifiable, and triage attributed a specific
dataset **to this question** (not merely to its publication). These are next
in line to be authored — and the first thing worth a human check.

| question | dataset | confidence |
|---|---|---|
| Which areas in the Netherlands have a structural pressure on their livability, and are there more or fewer of… | [`85067NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85067NED/table) | high |
| Did the post-WWII baby boom in the Netherlands compensate for the births that did not occur during WWII and t… | [`85524NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85524NED/table) | high |
| What is the development of the total estimated procurement volume of the Dutch governments in the years 2017,… | [`84122NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84122NED/table) | medium |
| What is the development of the procurement volume of awarded contracts below and above the threshold in the y… | [`84122NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84122NED/table) | medium |
| What is the procurement volume of the Dutch government in the period 2017–2019? | [`84122NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84122NED/table) | high |

### 3b. Answerable from open data but no dataset attributed (25 shown)

The classifier judged these answerable from published tables, but no
dataset could be attributed — because the source publication cited none
that we could link. Retrieval, not question quality, is the blocker.

| question | what would have to be pinned |
|---|---|
| How visible is aging in health care expenditures? | population, period, and measure |
| Which conditions lead to the highest healthcare expenditures? | population, period, and measure |
| What is the sports participation of Dutch people since 2012? | population, period, measure |
| What is the percentage of Dutch people who are members of a sports club? | population, period, and measure |
| What are the differences in sports participation between age groups? | population, period, and measure |
| What is the development of membership in sports associations? | population, period, and type of sports associations |
| What is the number of employers that fall directly under the scope of the collective labor agreemen… | The scope of the collective labor agreement is not defined, and the population … |
| What is the total number of employers that fall within the scope of the collective labor agreement? | scope of the collective labor agreement |
| What is the total number of employers that fall within the scope of the collective labor agreement? | scope of the collective labor agreement |
| How many persons are directly or on the basis of article 14 of the Cao Act bound to the Cao? | The question is about legal status under a specific law (Cao Act, article 14), … |
| What is the situation of holders of status who have established themselves in the municipality of G… |  |
| What is the household situation, socio-economic position, education participation, integration, inc… |  |
| How do the numbers of status holders in Groningen compare to those in The Hague and the rest of the… | The term 'status holders' is not clearly defined, and the time period or specif… |
| What percentage of migrant workers work at 100 percent, 105 percent, 110 percent, 115 percent, and … | The question is about a percentage of migrant workers relative to a 'Wml' (poss… |
| How large are the differences in labor market outcomes between persons with and without a migration… | population, period, and specific labor market outcomes (e.g., employment rate, … |
| Small households. Based on the population forecast, provide an overview of the development of the n… | Definition of 'small households', time period, and source of the population for… |
| How many Dutch people go on vacation annually? | period |
| Where do Dutch people go on vacation? | population (e.g., Dutch residents), period (e.g., 2023), and measure (e.g., mos… |
| What is the number and distribution of holidays in the Netherlands in 2023? |  |
| How did the number of holidays and overnight stays change in 2023 compared to 2022? | Which specific country or region, and which specific time period within 2023 an… |
| What are the most popular months to go on vacation in the Netherlands? | period |
| What is the distribution of holidays by age group in the Netherlands? | The question conflates 'holidays' (events) with 'age group' (demographic), and … |
| Which areas are rising or declining in terms of livability, and which dimensions of the Leefbaarome… | The specific dimensions of the Leefbaarometer to be analyzed, as well as the ti… |
| How many women had an unintended pregnancy in 2022 based on general practitioner registrations? | The question does not specify a clear population or geographic scope beyond 'ge… |
| What is the number of teenage mothers in 2022? | population (e.g., country, region), and possibly demographic details (e.g., age… |

### 3c. Microdata questions — the Tier B pool

The largest single bucket (1,202). Not answerable from public tables;
candidates for gold SQL authored against register schemas and validated
later with CBS.

| question | classifier reason |
|---|---|
| What is the percentage of non-use of the supplementary allowance among first-year higher education … | The question requires access to individual-level data on first-year higher educ… |
| What is the average amount of additional allowance that non-users miss out on? | The question requires access to individual-level data on allowance usage to cal… |
| What is the percentage of non-users of the supplementary allowance who do use the loan facilities o… | The question requires individual-level data on non-users of the supplementary a… |
| What is the predictive power of the model based on current income compared to more comprehensive mo… | The question requires comparing the predictive power of different models, which… |
| How does the predictive power change when the current model is supplemented with background charact… | The question requires evaluating changes in predictive power using model perfor… |
| How do the models and their predictive power differ between different target groups? | The question requires comparing predictive power across models and target group… |
| Is there a shift in the inflow to outpatient youth care towards households with higher incomes in E… | The question requires analyzing trends in inflow to outpatient youth care by ho… |
| What are the causes of the large and persistent differences in labor participation between persons … | The question asks for an explanation of causal factors behind labor participati… |
| What are the structural effects of measures on the labor participation and budgetary balance of per… | The question requires causal analysis of structural effects on labor participat… |
| To what extent are the data on pre-school education that schools provide to DUO representative? | The question concerns the representativeness of data provided by schools to DUO… |
| Are the data on early childhood education that schools provide to DUO suitable for making national … | The question requires an evaluation of the quality and suitability of the data,… |
| What is the size of the group of people who may be entitled to PAWW but do not register? | The question requires access to individual-level administrative data to identif… |
| Do borrowing constraints hamper self-employed individuals more than wage-employed individuals? | The question requires comparing the impact of borrowing constraints on self-emp… |
| Is there a change in criminal behavior of offenders in the 3 years before and after the start of re… | The question requires comparing individual-level criminal behavior data before … |
| What is the change in criminal behavior of offenders, translated into safety care costs, in the 3 y… | The question requires comparing safety care costs before and after rehabilitati… |

## How to review

1. **Section 1** — check each verified item end to end. A wrong dataset or a
   query that answers a different question is the failure to look for.
2. **Section 3a** — check the question↔dataset pairing before SQL is written;
   this is where the earlier document-level attribution bug did its damage.
3. **Section 3c** — check we are not throwing away good questions.

*Regenerate:* `python scripts/make_benchmark_review.py`
