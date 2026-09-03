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

## 4. Two candidate sources disagree — please adjudicate

Two independent ways of finding a dataset for a question:

- **cited** — the dataset the *publication itself named*, resolved lexically.
  Provenance-faithful: right if the task is *recompute the paper's number*.
- **retrieved** — nearest datasets by embedding over all 12,308 enriched
  catalogue entries. Task-faithful: right if the task is *find the data that
  answers this question*.

They agree on **none** of the cases below. That is not a retrieval failure:
papers cite reference tables (region definitions, classifications) alongside
the table carrying the measure, so 'what the paper cited' and 'what answers
the question' genuinely differ. **Which one is gold is a decision, not a
computation** — hence this section.

**Q:** Which areas in the Netherlands have a structural pressure on their livability, and are there more or fewer of them compared to the previous measurement?

<sub>scope: period=2020-2022 · geography=Netherlands · measure=areas with structural pressure on livability</sub>

| source | dataset | title |
|---|---|---|
| **cited** | [`85067NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85067NED/table) | *(named in the publication)* |
| retrieved #1 (0.6745) | [`71137NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/71137NED/table) | Census; Morally Unfavorable Housing Situations, 1930 |
| retrieved #2 (0.673) | [`81924NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/81924NED/table) | Liveability and Nuisance in Neighborhoods; Region (2012-201… |
| retrieved #3 (0.6587) | [`80168ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/80168ned/table) | Neighbourhood Livability by Police Region (IVM), 2008-2011 |

*Pick one, both, or neither.*

**Q:** To what extent have the employment prospects of youth been restored after the coronavirus pandemic in the Netherlands?

<sub>scope: period=2020-2021 · geography=Netherlands · population=youth entering the labor market · measure=employment prospects, measured as the chance of obtaining a substantial job (minimum three days per week)</sub>

| source | dataset | title |
|---|---|---|
| **cited** | [`83031NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83031NED/table) | *(named in the publication)* |
| retrieved #1 (0.668) | [`85178NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85178NED/table) | ICT Usage in Small Businesses by Company Size, 2021 |
| retrieved #2 (0.6561) | [`86087NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/86087NED/table) | Youth Labour Market Situation (15-27 years); Region 2024, 2… |
| retrieved #3 (0.6557) | [`82915NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82915NED/table) | Youth Labor Participation, 2003-2022 |

*Pick one, both, or neither.*

**Q:** What is the impact of the coronavirus pandemic on the transition of young people from education to the labor market in the Netherlands between 2018 and 2021?

<sub>scope: period=2018-2021 · geography=Netherlands · population=young people · measure=transition from education to the labor market</sub>

| source | dataset | title |
|---|---|---|
| **cited** | [`83031NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83031NED/table) | *(named in the publication)* |
| retrieved #1 (0.686) | [`85178NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85178NED/table) | ICT Usage in Small Businesses by Company Size, 2021 |
| retrieved #2 (0.6805) | [`80220ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/80220ned/table) | Labour Market Mobility: Changes in Labour Market Position (… |
| retrieved #3 (0.6794) | [`iss_21covt`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/iss_21covt/table) | Enterprises by Impact of COVID-19 on Turnover and NACE Acti… |

*Pick one, both, or neither.*

**Q:** Which groups of young people have been more affected by the coronavirus pandemic in the labor market, specifically those without a starting qualification, with at most an mbo-2 education, w…

<sub>scope: population=young people · measure=labor market chances</sub>

| source | dataset | title |
|---|---|---|
| **cited** | [`83031NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83031NED/table) | *(named in the publication)* |
| retrieved #1 (0.6326) | [`85696NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85696NED/table) | MBO Graduates: Labour Market Position After Leaving Educati… |
| retrieved #2 (0.6317) | [`71827ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/71827ned/table) | MBO Graduates by Origin Group and Cohort (1990/91-2015/16) |
| retrieved #3 (0.631) | [`iss_21covb`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/iss_21covb/table) | Enterprises affected by trade barriers due to COVID-19 by b… |

*Pick one, both, or neither.*

**Q:** Did the post-WWII baby boom in the Netherlands compensate for the births that did not occur during WWII and the interwar period?

<sub>scope: period=post-WWII · geography=Netherlands · measure=fertility rates</sub>

| source | dataset | title |
|---|---|---|
| **cited** | [`85524NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85524NED/table) | *(named in the publication)* |
| retrieved #1 (0.6638) | [`80749ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/80749ned/table) | Fertility Rates by Women's Birth Cohorts, 1935-2020 |
| retrieved #2 (0.6513) | [`37422ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/37422ned/table) | Birth Statistics, Key Figures, 1950-2022 |
| retrieved #3 (0.651) | [`37520`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/37520/table) | Births and Mother's Exact Age (1950-2014) |

*Pick one, both, or neither.*

**Q:** What is the development of the total estimated procurement volume of the Dutch governments in the years 2017, 2018, and 2019?

<sub>scope: period=2017-2019 · geography=Netherlands · measure=total estimated procurement volume</sub>

| source | dataset | title |
|---|---|---|
| **cited** | [`84122NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84122NED/table) | *(named in the publication)* |
| retrieved #1 (0.7117) | [`60050`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/60050/table) | Netherlands Government Expenditures and Revenues by Budget … |
| retrieved #2 (0.6991) | [`84089NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84089NED/table) | Contribution to GDP Volume Growth; National Accounts, 2016-… |
| retrieved #3 (0.692) | [`84114NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84114NED/table) | Government Finances; Key Figures 1995-2023 |

*Pick one, both, or neither.*

**Q:** What is the development of the procurement volume of awarded contracts below and above the threshold in the years 2017, 2018, and 2019 in the Netherlands for the Dutch government?

<sub>scope: period=2017-2019 · geography=Netherlands · population=Dutch government · measure=procurement volume of awarded contracts below and above the threshold</sub>

| source | dataset | title |
|---|---|---|
| **cited** | [`84122NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84122NED/table) | *(named in the publication)* |
| retrieved #1 (0.6775) | [`60050`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/60050/table) | Netherlands Government Expenditures and Revenues by Budget … |
| retrieved #2 (0.6564) | [`60012`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/60012/table) | Netherlands Government Sector ESR 95 Transactions, 1996-2010 |
| retrieved #3 (0.6525) | [`82605NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82605NED/table) | Imports and Exports by Property Transfer; Volume Developmen… |

*Pick one, both, or neither.*

**Q:** What is the procurement volume of the Dutch government in the period 2017–2019?

<sub>scope: period=2017-2019 · geography=Netherlands · measure=procurement volume</sub>

| source | dataset | title |
|---|---|---|
| **cited** | [`84122NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84122NED/table) | *(named in the publication)* |
| retrieved #1 (0.7182) | [`60050`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/60050/table) | Netherlands Government Expenditures and Revenues by Budget … |
| retrieved #2 (0.7026) | [`84114NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84114NED/table) | Government Finances; Key Figures 1995-2023 |
| retrieved #3 (0.7022) | [`81242ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/81242ned/table) | Government Finances; Key Figures 1987 - 2013 |

*Pick one, both, or neither.*

**Q:** Does the duration of social benefits differ among specific subgroups (based on gender and/or age) among Syrian and Eritrean status holders in the Netherlands?

<sub>scope: geography=Netherlands · population=Syrian and Eritrean status holders · measure=duration of social benefits</sub>

| source | dataset | title |
|---|---|---|
| **cited** | [`83102NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83102NED/table) | *(named in the publication)* |
| retrieved #1 (0.7512) | [`81368ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/81368ned/table) | People Receiving Social Assistance (Bijstandsuitkering); Du… |
| retrieved #2 (0.7469) | [`81367ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/81367ned/table) | People Receiving Social Security Benefits: Duration and Cha… |
| retrieved #3 (0.7468) | [`85585NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85585NED/table) | People on Social Assistance by Duration of Benefit Situation |

*Pick one, both, or neither.*

### 4a. Retrieval-only candidates — sample of 20

Questions whose publication cited nothing we could link. These are the
~1,000 that lexical matching lost entirely; retrieval gives them a
candidate for the first time. Judge whether the top hit is usable.

| question | top retrieved | score |
|---|---|---|
| How visible is aging in health care expenditures in the Netherlands in 2019? | [`83075NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83075NED/table) Health Care Expenditure and Funding in … | 0.758 |
| Which conditions lead to the highest healthcare expenditures in the Netherlands in 2019? | [`84047NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84047NED/table) Health and Welfare Expenditure Key Figu… | 0.6958 |
| What is the extent of non-use of the supplementary allowance among students in higher educatio… | [`ilc_ats11`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/ilc_ats11/table) Non-participation in Professional Train… | 0.5988 |
| How will the quantitative and qualitative housing needs develop in Zutphen until 2030, with a … | [`37127ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/37127ned/table) Household Projections: Key Figures, 200… | 0.6833 |
| How does the demand for residential care develop for different target groups in Zutphen in the… | [`81645NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/81645NED/table) Household Projections by Type, Age, and… | 0.7225 |
| What is the additional housing need in Zutphen until 2030 and 2040? | [`86054NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/86054NED/table) Housing Stock: Additional Additions and… | 0.6598 |
| How does the demand for social rental housing develop until 2030, with a look ahead to 2040? W… | [`84823NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84823NED/table) Rental Development of Dwellings by Type… | 0.6861 |
| What is the sports participation of Dutch people aged 6 years or older since 2012? | [`80909ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/80909ned/table) Active and Passive Sports Participation… | 0.7434 |
| What is the percentage of Dutch people aged 6 years or older who are members of a sports club? | [`82869NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82869NED/table) Caribbean Netherlands: Vacation and Spo… | 0.719 |
| How has the corona pandemic influenced sports participation among people aged 6 years or older… | [`82869NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82869NED/table) Caribbean Netherlands: Vacation and Spo… | 0.6692 |
| What are the differences in sports participation between age groups in the Netherlands based o… | [`7082SPBE`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/7082SPBE/table) Sports Participation by Type of Sport | 0.7335 |
| What is the development of membership in sports associations among the Dutch population aged 6… | [`84109NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84109NED/table) Water Sports Clubs: Membership and Acti… | 0.7435 |
| What is the number of employers that fall directly under the scope of the collective labor agr… | [`tour_lfsq4r2`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/tour_lfsq4r2/table) Employed persons in tourism industries … | 0.6522 |
| What is the total number of employers that fall within the scope of the collective labor agree… | [`tour_lfsq4r2`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/tour_lfsq4r2/table) Employed persons in tourism industries … | 0.6579 |
| What is the total number of employers that fall within the scope of the collective labor agree… | [`tour_lfsq4r2`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/tour_lfsq4r2/table) Employed persons in tourism industries … | 0.6579 |
| How many persons are directly or on the basis of article 14 of the Cao Act bound to the Cao as… | [`83862NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83862NED/table) Hospitality Sector Turnover Development… | 0.6179 |
| What is the situation of holders of status who have established themselves in the municipality… | [`71191ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/71191ned/table) Persons with Employment or Benefits; Du… | 0.6405 |
| What is the household situation, socio-economic position, education participation, integration… | [`37670WON`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/37670WON/table) Housing situation of private households… | 0.6626 |
| How do the numbers of status holders in Groningen compare to those in The Hague and the rest o… | [`70147ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/70147ned/table) Household Income in Neighborhoods of th… | 0.6723 |
| What is the expected growing demand for cybersecurity personnel in the Netherlands? | [`85409NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85409NED/table) Business ICT Usage by Company Size, 2022 | 0.6679 |

## How to review

1. **Section 1** — check each verified item end to end. A wrong dataset or a
   query that answers a different question is the failure to look for.
2. **Section 3a** — check the question↔dataset pairing before SQL is written;
   this is where the earlier document-level attribution bug did its damage.
3. **Section 3c** — check we are not throwing away good questions.
4. **Section 4** — decide which notion of gold we want: the dataset the paper
   cited, or the dataset that best answers the question. This choice shapes the
   whole benchmark and cannot be settled automatically.

*Regenerate:* `python scripts/make_benchmark_review.py`
