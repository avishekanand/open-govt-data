# CBS questions with executed answers

40 questions whose gold SQL runs against the live table and returns a
result that passes an automated sanity check. Each answer below is the pinned
snapshot; re-running the query reproduces it.

3 further queries ran but were rejected by QA (an unfiltered dimension
repeats every grouping key) — listed in [benchmark_items_review.md](benchmark_items_review.md).

### How did the number of holidays and overnight stays change in 2023 compared to 2022 in the Netherlands?

*Dataset:* [`85302NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85302NED/table) · *type:* `comparison` · *id:* `cbs-a0001`

*What the query pins down:* How did the total number of holidays and overnight stays change in the Netherlands in 2023 compared to 2022, using the 'Totaal vakanties' and 'Totaal overnachtingen' measures and filtering to the tot…

```sql
SELECT 
  Perioden_label AS period,
  measure AS measure_type,
  SUM(value) AS count
FROM t_85302NED
WHERE measure IN ('Totaal vakanties', 'Totaal overnachtingen')
  AND Perioden_label IN ('2022', '2023')
  AND BestemmingEnSeizoen_label = 'Totaal vakanties'
GROUP BY Perioden_label, measure;
```

| period | measure_type | count |
|---|---|---|
| 2023 | Totaal vakanties | 882262.0 |
| 2022 | Totaal overnachtingen | 6511.999999999998 |
| 2023 | Totaal overnachtingen | 6871.500000000001 |
| 2022 | Totaal vakanties | 843926.0 |

<sub>question attested in https://www.landelijkedataalliantie.nl/nl/home/download/cvo-basisrapport-2024?disposition=inline</sub>

### Where do Dutch residents go on vacation in 2024?

*Dataset:* [`84367NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84367NED/table) · *type:* `distribution` · *id:* `cbs-a0005`

*What the query pins down:* What percentage of Dutch residents went on vacation to different destinations in 2024?

```sql
SELECT Bestemming_label AS destination, value AS percentage_of_dutch_residents FROM t_84367NED WHERE measure = 'Percentage Nederlanders' AND Perioden = '2024JJ00' AND Marges = 'MW00000';
```

| destination | percentage_of_dutch_residents |
|---|---|
| Buitenland | 63.8 |
| Afrika totaal | 4.8 |
| Noord-Afrika | 3.4 |
| Overig Afrika | 1.0 |
| Azië totaal | 6.7 |
| West-Azië | 2.7 |
| Zuidoost-Azië | 2.0 |
| Overig Azië | 2.2 |
| … | 31 more rows |

<sub>question attested in https://www.landelijkedataalliantie.nl/nl/home/download/download-rapportage-vakantiegedrag-inwoners-nederland-2024-nl?disposition=inline</sub>

### Which groups are risk groups for work accidents in the Netherlands?

*Dataset:* [`84433NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84433NED/table) · *type:* `ranking_or_list` · *id:* `cbs-a0006`

*What the query pins down:* Which occupational groups are at the highest risk for work accidents with four or more days of absence in the Netherlands in 2022?

```sql
SELECT Beroep_label AS risk_group FROM t_84433NED WHERE measure = 'Werknemers ongeval 4 dgn of meer verzuim' AND unit = 'In % van alle werknemers' AND Perioden = '2022JJ00' AND Marges = 'MOG0095' AND Beroep_label NOT LIKE 'Totaal%' ORDER BY value DESC LIMIT 10;
```

| risk_group |
|---|
| 0632 Politie en brandweer |
| 1214 Vrachtwagenchauffeurs |
| 0732 Timmerlieden |
| 073 Bouwarbeiders |
| 12 Transport en logistiek beroepen |
| 121 Bestuurders voertuigen en bediene... |
| 0771 Productiemachinebedieners |
| 063 Beveiligingswerkers |
| … | 2 more rows |

<sub>question attested in https://monitorarbeid.tno.nl/wp-content/uploads/sites/16/2023/10/Arbeidsongevallen-in-Nederland-2011.pdf</sub>

### Where do Dutch people go on vacation?

*Dataset:* [`85302NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85302NED/table) · *type:* `distribution` · *id:* `cbs-a0001`

*What the query pins down:* What are the vacation destinations of Dutch people in 2023, measured as the average per person per vacation?

```sql
SELECT BestemmingEnSeizoen_label AS destination, value AS average_per_person FROM t_85302NED WHERE measure = 'Gemiddeld per persoon per vakantie' AND Perioden_label = '2023' AND Vakantiekenmerken_label = 'Totaal vakanties';
```

| destination | average_per_person |
|---|---|
| Totaal vakanties | 608.0 |
| Vakantiebestemming: Nederland | 260.0 |
| Vakantiebestemming: buitenland | 887.0 |
| Zomerseizoen | 654.0 |
| Winterseizoen | 541.0 |

<sub>question attested in https://www.landelijkedataalliantie.nl/nl/home/download/cvo-basisrapport-2024?disposition=inline</sub>

### What do Dutch people spend on a vacation in 2023?

*Dataset:* [`85302NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85302NED/table) · *type:* `distribution` · *id:* `cbs-a0002`

*What the query pins down:* What is the total spending on vacations by Dutch people in 2023, specifically for trips to destinations outside the Netherlands?

```sql
SELECT measure, unit, value, Perioden_label FROM t_85302NED WHERE measure = 'Totaal vakantie-uitgaven' AND Perioden_label = '2023' AND Vakantiekenmerken_label = 'Totaal vakanties' AND Marges_label = 'Waarde' AND BestemmingEnSeizoen_label = 'Vakantiebestemming: buitenland';
```

| measure | unit | value | Perioden_label |
|---|---|---|---|
| Totaal vakantie-uitgaven | mln euro | 18497.0 | 2023 |

<sub>question attested in https://www.landelijkedataalliantie.nl/nl/home/download/cvo-basisrapport-2024?disposition=inline</sub>

### What is the number of teenage mothers in 2022 in the Netherlands?

*Dataset:* [`85722NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85722NED/table) · *type:* `single_number` · *id:* `cbs-a0003`

*What the query pins down:* What is the number of teenage mothers in the year 2022 in the Netherlands?

```sql
SELECT value_text AS number_of_teenage_mothers FROM t_85722NED WHERE measure = 'Jonger dan 20 jaar' AND Perioden_label = '2022';
```

| number_of_teenage_mothers |
|---|
|  |

<sub>question attested in https://www.rivm.nl/publicaties/monitor-onbedoelde-zwangerschappen-cijferoverzicht-2023</sub>

### How many choice support trajectories were carried out in 2022 in the Netherlands?

*Dataset:* [`85096NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85096NED/table) · *type:* `single_number` · *id:* `cbs-a0004`

*What the query pins down:* How many choice support trajectories were carried out in 2022 in the Netherlands, specifically for all referrers (total referrer)?

```sql
SELECT value AS number_of_choice_support_trajectories FROM t_85096NED WHERE measure = 'Toaal begonnen trajecten' AND Perioden = '2022JJ00' AND VerwijzerJeugdzorg_label = 'Totaal verwijzer';
```

| number_of_choice_support_trajectories |
|---|
| 15715.0 |

<sub>question attested in https://www.rivm.nl/publicaties/monitor-onbedoelde-zwangerschappen-cijferoverzicht-2023</sub>

### Have there been any recent updates on the persistence of changes on the labor market in the Netherlands from 2003Q1 to 2023Q4?

*Dataset:* [`82848NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82848NED/table) · *type:* `trend` · *id:* `cbs-a0005`

*What the query pins down:* Have there been any recent updates on the persistence of changes on the labor market in the Netherlands from 2020Q1 to 2023Q4, focusing on the 'Werkzame beroepsbevolking' (working labor force) and me…

```sql
SELECT Perioden_label AS period, value AS persistence_value FROM t_82848NED WHERE measure = 'Seizoengecorrigeerd' AND Arbeidspositie = '3000810' AND ArbeidspositieDrieMaandenEerder = '3000810' AND Perioden BETWEEN '2020KW01' AND '2023KW04';
```

| period | persistence_value |
|---|---|
| 2020 januari | 3119.0 |
| 2020 februari | 3130.0 |
| 2020 maart | 3144.0 |
| 2020 1e kwartaal | 3131.0 |
| 2020 april | 3220.0 |
| 2020 mei | 3228.0 |
| 2020 juni | 3186.0 |
| 2020 2e kwartaal | 3211.0 |
| … | 44 more rows |

<sub>question attested in https://papers.tinbergen.nl/24047.pdf</sub>

### How are recent labor market dynamics at the job occupation level compared to earlier time periods?

*Dataset:* [`85388NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85388NED/table) · *type:* `trend` · *id:* `cbs-a0006`

*What the query pins down:* How are recent labor market dynamics at the job occupation level (e.g., commercial, managerial, healthcare) compared to earlier time periods (2016, 2020, and 2024), focusing on the active working pop…

```sql
SELECT Perioden_label AS period, BeroepsklasseVoorgaandePeriode_label AS occupation_class, value AS labor_force, WisselingBeroepsklasse_label AS job_change_type FROM t_85388NED WHERE measure = 'Werkzame beroepsbevolking' AND Perioden IN ('2024KW01', '2020KW01', '2016KW01') AND BeroepsklasseVoorgaandePeriode_label IN ('Totaal', '03 Commerciële beroepen', '05 Managers', '10 Zorg en welzijn beroepen') AND WisselingBeroepsklasse_label IN ('Totaal', 'Geen wisseling beroep', 'Wisseling beroep: andere beroepsklasse') ORDER BY Perioden_label DESC, BeroepsklasseVoorgaandePeriode_label, WisselingBeroepsklasse_label;
```

| period | occupation_class | labor_force | job_change_type |
|---|---|---|---|
| 2024 1e kwartaal | 03 Commerciële beroepen | 945.0 | Geen wisseling beroep |
| 2024 1e kwartaal | 03 Commerciële beroepen | 1021.0 | Totaal |
| 2024 1e kwartaal | 03 Commerciële beroepen | 62.0 | Wisseling beroep: andere beroepsklasse |
| 2024 1e kwartaal | 05 Managers | 499.0 | Geen wisseling beroep |
| 2024 1e kwartaal | 05 Managers | 524.0 | Totaal |
| 2024 1e kwartaal | 05 Managers | 18.0 | Wisseling beroep: andere beroepsklasse |
| 2024 1e kwartaal | 10 Zorg en welzijn beroepen | 1379.0 | Geen wisseling beroep |
| 2024 1e kwartaal | 10 Zorg en welzijn beroepen | 1426.0 | Totaal |
| … | 28 more rows | | |

<sub>question attested in https://papers.tinbergen.nl/24047.pdf</sub>

### What is the housing market development in the municipality of Losser?

*Dataset:* [`85773NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85773NED/table) · *type:* `trend` · *id:* `cbs-a0007`

*What the query pins down:* What is the housing market development in the municipality of Losser for the first quarter of 2000, measured as the development compared to the same period one year earlier?

```sql
SELECT Perioden_label AS period, measure, value FROM t_85773NED WHERE measure = 'Ontwikkeling  t.o.v. een jaar eerder' AND Perioden_label = '2000 1e kwartaal' AND value IS NOT NULL LIMIT 1;
```

| period | measure | value |
|---|---|---|
| 2000 1e kwartaal | Ontwikkeling  t.o.v. een jaar eerder | 19.5 |

<sub>question attested in https://www.companen.nl/wp-content/uploads/2025/02/168.107-Woningbehoefteonderzoek-Gemeente-Losser-2024-definitief.pdf</sub>

### What role does migration play in the population growth of Leusden?

*Dataset:* [`83474NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83474NED/table) · *type:* `rate_or_share` · *id:* `cbs-a0008`

*What the query pins down:* What is the contribution of immigration to population growth in Leusden for the period ending in February 2004?

```sql
SELECT Perioden_label AS period, value AS migration_contribution_to_growth FROM t_83474NED WHERE measure = 'Immigratie' AND Perioden_label = '2004 februari';
```

| period | migration_contribution_to_growth |
|---|---|
| 2004 februari | 6656.0 |

<sub>question attested in https://www.companen.nl/wp-content/uploads/2024/09/Woningmarktonderzoek-Leusden-020224.pdf</sub>

### What is the prevalence of chronic kidney disease in the Netherlands among adults aged 18 years and older in 2021?

*Dataset:* [`83005NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83005NED/table) · *type:* `rate_or_share` · *id:* `cbs-a0009`

*What the query pins down:* What is the prevalence of chronic kidney disease in the Netherlands among adults aged 18 years and older in 2021? The query filters for the measure 'Nieraandoening' (chronic kidney disease), the popu…

```sql
SELECT value AS prevalence_percentage FROM t_83005NED WHERE measure = 'Nieraandoening' AND unit = '%' AND Persoonskenmerken_label = 'Leeftijd: 18 jaar of ouder' AND Perioden_label = '2021';
```

| prevalence_percentage |
|---|
| 1.9 |
| 1.6 |
| 2.2 |

<sub>question attested in https://www.nivel.nl/sites/default/files/bestanden/1004641.pdf</sub>

### What is the incidence of chronic kidney disease in the Netherlands in 2021?

*Dataset:* [`83005NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83005NED/table) · *type:* `rate_or_share` · *id:* `cbs-a0010`

*What the query pins down:* What is the incidence of chronic kidney disease in the Netherlands in 2021, for the population of people aged 18 years and older known in primary care?

```sql
SELECT value AS incidence_rate, Perioden_label AS year FROM t_83005NED WHERE measure = 'Nieraandoening' AND Perioden = '2021JJ00' AND Persoonskenmerken_label = 'Leeftijd: 18 jaar of ouder' AND Marges_label = 'Waarde';
```

| incidence_rate | year |
|---|---|
| 1.9 | 2021 |

<sub>question attested in https://www.nivel.nl/sites/default/files/bestanden/1004641.pdf</sub>

### What is the impact of migration on population growth in Haaksbergen?

*Dataset:* [`83474NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83474NED/table) · *type:* `trend` · *id:* `cbs-a0011`

*What the query pins down:* What is the impact of migration on population growth in Haaksbergen for the period ending in September 2000?

```sql
SELECT Perioden_label AS period, value AS migration_impact_on_population_growth FROM t_83474NED WHERE measure = 'Totale bevolkingsgroei' AND Perioden_label = '2000 september' AND unit = 'aantal';
```

| period | migration_impact_on_population_growth |
|---|---|
| 2000 september | 13670.0 |

<sub>question attested in https://www.haaksbergen.nl/Docs/bouwen/College%201-10-2024/Bijlage%205%20actualisatie%20woonbehoefteonderzoek%20Stec.pdf</sub>

### Who are the victims of work-related accidents in the Netherlands?

*Dataset:* [`84433NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84433NED/table) · *type:* `distribution` · *id:* `cbs-a0012`

*What the query pins down:* What are the occupations of victims of work-related accidents in the Netherlands for the most recent available period (2025)?

```sql
SELECT Beroep_label AS occupation, value AS victims_of_work_related_accidents FROM t_84433NED WHERE measure = 'Totaal werknemers met een ongeval' AND Marges = 'MW00000' AND Perioden = '2025JJ00';
```

| occupation | victims_of_work_related_accidents |
|---|---|
| Totaal | 2.5 |
| Beroepsniveau 1 (ISCO 2008) | 4.0 |
| Beroepsniveau 2 (ISCO 2008) | 3.6 |
| Beroepsniveau 3 (ISCO 2008) | 2.3 |
| Beroepsniveau 4 (ISCO 2008) | 1.2 |
| Beroepsniveau onbekend (ISCO 2008) | 3.9 |
| 01 Pedagogische beroepen | 2.2 |
| 011 Docenten | 2.1 |
| … | 136 more rows |

<sub>question attested in https://www.vzinfo.nl/sites/default/files/2021-11/venema-2011-monitor.pdf</sub>

### Are there differences between small and medium-sized enterprises (businesses with fewer than 250 employees) and large enterprises…

*Dataset:* [`84985NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84985NED/table) · *type:* `comparison` · *id:* `cbs-a0013`

*What the query pins down:* Are there differences between small and medium-sized enterprises (50 to 250 employees) and large enterprises (250 or more employees) in their research and development (R&D) activities in the Netherla…

```sql
SELECT 
  Bedrijfsgrootte_label AS business_size,
  measure,
  unit,
  MAX(value) AS value
FROM t_84985NED
WHERE 
  measure IN ('Uitgaven voor eigen R&D activiteiten', 'Werkzame personen', 'Ondernemingen met eigen R&D activiteiten')
  AND Bedrijfsgrootte_label IN ('50 tot 250 werkzame personen', '250 of meer werkzame personen')
  AND Perioden_label = '2024'
GROUP BY 
  Bedrijfsgrootte_label,
  measure,
  unit;
```

| business_size | measure | unit | value |
|---|---|---|---|
| 50 tot 250 werkzame personen | Uitgaven voor eigen R&D activiteiten | mln euro | 3693.0 |
| 50 tot 250 werkzame personen | Werkzame personen | aantal | 47238.0 |
| 250 of meer werkzame personen | Werkzame personen | aantal | 100552.0 |
| 250 of meer werkzame personen | Uitgaven voor eigen R&D activiteiten | mln euro | 12567.0 |

<sub>question attested in https://www.mejudice.nl/artikelen/detail/ondernemingen-onder-nederlandse-zeggenschap-investeren-het-meest-in-onderzoek-en-ontwikkeling-in-nederland</sub>

### What trend in healthy life expectancy do we observe in the Netherlands according to a health measure commonly used in Europe?

*Dataset:* [`71950ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/71950ned/table) · *type:* `trend` · *id:* `cbs-a0014`

*What the query pins down:* What is the trend in healthy life expectancy in the Netherlands over time for the total population (men and women combined) at birth, using the standard measure of life expectancy?

```sql
SELECT Perioden AS period, value AS healthy_life_expectancy FROM t_71950ned WHERE measure = 'Levensverwachting (LV)' AND Geslacht = '3000' AND LeeftijdOp31December = '10010' AND Marges = 'MW00000' AND Perioden BETWEEN '2000' AND '2020' ORDER BY period;
```

| period | healthy_life_expectancy |
|---|---|
| 2000JJ00 | 75.54 |
| 2001JJ00 | 75.8 |
| 2002JJ00 | 75.99 |
| 2003JJ00 | 76.23 |
| 2004JJ00 | 76.87 |
| 2005JJ00 | 77.19 |
| 2006JJ00 | 77.63 |
| 2007JJ00 | 78.01 |
| … | 12 more rows |

<sub>question attested in https://nidi.nl/demos/is-langer-leven-ook-gezonder-leven/</sub>

### To what extent is the labor participation of female status holders in the Netherlands related to the presence of young children i…

*Dataset:* [`85767NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85767NED/table) · *type:* `rate_or_share` · *id:* `cbs-a0015`

*What the query pins down:* To what extent is the labor participation of female status holders in the Netherlands related to the presence of young children (0-6 years old) in the household between 2014 and 2018?

```sql
SELECT 
  Perioden_label AS year,
  AantalKinderen_label AS children_in_household,
  SUM(value) AS total_count
FROM t_85767NED
WHERE 
  Perioden IN ('2014JJ00', '2015JJ00', '2016JJ00', '2017JJ00', '2018JJ00') 
  AND KenmerkenHuishoudens_label IN ('Jongste kind 0 tot 6 jaar', 'Jongste kind 6 tot 12 jaar', 'Zonder thuiswonende kinderen') 
  AND AantalKinderen_label IN ('Paar: met 1 kind', 'Paar: met 2 kinderen', 'Paar: zonder kind') 
GROUP BY 
  Perioden_label,
  AantalKinderen_label
ORDER BY 
  Perioden_label,
  AantalKinderen_label;
```

| year | children_in_household | total_count |
|---|---|---|
| 2014 | Paar: met 1 kind | 1619.0 |
| 2014 | Paar: met 2 kinderen | 2988.0 |
| 2014 | Paar: zonder kind | 7886.0 |
| 2015 | Paar: met 1 kind | 1559.0 |
| 2015 | Paar: met 2 kinderen | 2991.0 |
| 2015 | Paar: zonder kind | 7912.0 |
| 2016 | Paar: met 1 kind | 1528.0 |
| 2016 | Paar: met 2 kinderen | 2990.0 |
| … | 7 more rows | |

<sub>question attested in https://nidi.nl/demos/vrouwelijke-statushouders-met-jonge-kinderen-komen-nauwelijks-aan-werk-toe/</sub>

### What is the current state of injury-related issues in the Netherlands in 2021, based on data from the Letsel Informatie Systeem (…

*Dataset:* [`85568NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85568NED/table) · *type:* `distribution` · *id:* `cbs-a0016`

*What the query pins down:* What is the current state of injury-related issues (physical injury, mental injury, and total workers with an accident) in the Netherlands for the year 2021, based on data from the Letsel Informatie …

```sql
SELECT measure, unit, value, Perioden_label AS year FROM t_85568NED WHERE Perioden = '2021JJ00' AND measure IN ('Lichamelijk letsel', 'Geestelijk letsel', 'Totaal werknemers met een ongeval') AND KenmerkenWerknemers = '2021240' AND Marges = 'MW00000';
```

| measure | unit | value | year |
|---|---|---|---|
| Totaal werknemers met een ongeval | In % van alle werknemers | 2.5 | 2021 |
| Lichamelijk letsel | In % van alle werknemers | 1.6 | 2021 |
| Geestelijk letsel | In % van alle werknemers | 0.4 | 2021 |

<sub>question attested in https://www.veiligheid.nl/kennisaanbod/cijferrapportage/kerncijfers-letsels-nederland</sub>

### Why is the Dutch infant mortality rate starting to rise again?

*Dataset:* [`37979ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/37979ned/table) · *type:* `trend` · *id:* `cbs-a0017`

*What the query pins down:* Why is the infant mortality rate (per 1,000 live births) for all infants in the Netherlands starting to rise again in recent years?

```sql
SELECT Perioden_label AS year, value AS infant_mortality_rate FROM t_37979ned WHERE measure = 'Zuigelingensterfte, relatief' AND unit = 'per 1 000 levend geborenen' AND Geslacht_label = 'Totaal mannen en vrouwen' AND Perioden_label > '2010' ORDER BY year DESC LIMIT 5;
```

| year | infant_mortality_rate |
|---|---|
| 2025 | 3.4 |
| 2024 | 3.3 |
| 2023 | 3.6 |
| 2022 | 3.2 |
| 2021 | 3.3 |

<sub>question attested in https://www.rivm.nl/publicaties/beter-weten-beter-begin-samen-sneller-naar-betere-zorg-rond-zwangerschap</sub>

### What is the pace of sustainability improvement of the housing stock in the Netherlands?

*Dataset:* [`86054NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/86054NED/table) · *type:* `trend` · *id:* `cbs-a0018`

*What the query pins down:* What is the pace of sustainability improvement of the housing stock in the Netherlands, measured as the 'Saldo voorraad' (net stock change) over time, at the national level and annual granularity?

```sql
SELECT Perioden_label AS period, value AS sustainability_improvement_rate FROM t_86054NED WHERE measure = 'Saldo voorraad' AND RegioS_label = 'Nederland' AND unit = 'aantal' ORDER BY Perioden;
```

| period | sustainability_improvement_rate |
|---|---|
| 2020 | 74545.0 |
| 2021 | 79250.0 |
| 2022 | 79650.0 |
| 2023 | 78820.0 |
| 2024 | 70420.0 |
| 2026 1e kwartaal | 13490.0 |

<sub>question attested in https://www.seo.nl/publicaties/op-weg-naar-een-duurzame-economie-jubileumboek-seo-75-jaar/</sub>

### How often was the emergency department (SEH) used in the Netherlands between 2017 and 2020?

*Dataset:* [`83005NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83005NED/table) · *type:* `trend` · *id:* `cbs-a0019`

*What the query pins down:* How often was the emergency department (SEH) used in the Netherlands between 2017 and 2020, measured as the number of people with at least one contact per year?

```sql
SELECT Perioden_label AS year, SUM(value) AS emergency_department_use_count FROM t_83005NED WHERE measure = 'Personen met minimaal 1 contact' AND Perioden IN ('2017JJ00', '2018JJ00', '2019JJ00', '2020JJ00') AND Marges = 'MW00000' GROUP BY Perioden_label;
```

| year | emergency_department_use_count |
|---|---|
| 2018 | 11975.999999999998 |
| 2019 | 11799.299999999994 |
| 2017 | 11702.1 |
| 2020 | 11322.400000000005 |

<sub>question attested in https://www.rivm.nl/publicaties/zorggebruik-op-spoedeisende-hulp-in-kaart-gebracht-notitie-kosten-van-ziekten-studie</sub>

### To what extent has the development of private R&D intensity in the Netherlands between 2015 and 2022 been driven by changes in R&…

*Dataset:* [`84985NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84985NED/table) · *type:* `trend` · *id:* `cbs-a0020`

*What the query pins down:* To what extent has the development of private R&D intensity in the Netherlands between 2015 and 2022 been driven by changes in R&D intensity within sectors or by changes in the Dutch sector structure…

```sql
SELECT
  Perioden_label AS year,
  BedrijfstakkenBranchesSBI2008_label AS sector,
  measure AS measure_type,
  unit,
  value AS value
FROM t_84985NED
WHERE
  measure IN ('Uitgaven voor eigen R&D activiteiten', 'Ondernemingen met eigen R&D activiteiten')
  AND Bedrijfsgrootte_label = 'Totaal'
  AND Perioden IN ('2015JJ00', '2022JJ00')
ORDER BY year, sector;
```

| year | sector | measure_type | unit | value |
|---|---|---|---|---|
| 2015 | 10 Voedingsmiddelenindustrie | Ondernemingen met eigen R&D activiteiten | aantal | 425.0 |
| 2015 | 10-12 Voedings-, genotmiddelenindustrie | Uitgaven voor eigen R&D activiteiten | mln euro | 351.0 |
| 2015 | 10-12 Voedings-, genotmiddelenindustrie | Ondernemingen met eigen R&D activiteiten | aantal | 450.0 |
| 2015 | 11 Drankenindustrie | Ondernemingen met eigen R&D activiteiten | aantal | 15.0 |
| 2015 | 12 Tabaksindustrie | Ondernemingen met eigen R&D activiteiten | aantal | 5.0 |
| 2015 | 12 Tabaksindustrie | Uitgaven voor eigen R&D activiteiten | mln euro | 2.0 |
| 2015 | 13 Textielindustrie | Uitgaven voor eigen R&D activiteiten | mln euro | 12.0 |
| 2015 | 13 Textielindustrie | Ondernemingen met eigen R&D activiteiten | aantal | 70.0 |
| … | 190 more rows | | | |

<sub>question attested in https://www.bedrijvenbeleidinbeeld.nl/themas/beleidsthemas-verder-onder-de-loep/van-realiteit-naar-ambitie-investeringspatronen-en-structurele-verschillen-in-rd-door-bedrijven-in-de-context-van-de-3-procentdoelstelling</sub>

### Where can the most profit be gained from R&D investments: in companies that already invest a lot in R&D or in smaller players?

*Dataset:* [`84985NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84985NED/table) · *type:* `comparison` · *id:* `cbs-a0021`

*What the query pins down:* Where can the most profit be gained from R&D investments: in companies that already invest a lot in R&D or in smaller players? The analysis is based on the period 2015-2022, for the Netherlands, and …

```sql
SELECT
  Bedrijfsgrootte_label AS company_size,
  SUM(CASE WHEN measure = 'Uitgaven voor eigen R&D activiteiten' THEN value END) AS total_RD_spending,
  SUM(CASE WHEN measure = 'Ondernemingen met eigen R&D activiteiten' THEN value END) AS companies_with_RD
FROM t_84985NED
WHERE
  Perioden IN (SELECT Perioden FROM t_84985NED WHERE Perioden_label IN ('2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'))
  AND measure IN ('Uitgaven voor eigen R&D activiteiten', 'Ondernemingen met eigen R&D activiteiten')
  AND unit = 'mln euro'
GROUP BY Bedrijfsgrootte_label
ORDER BY total_RD_spending DESC;
```

| company_size | total_RD_spending | companies_with_RD |
|---|---|---|
| Totaal | 435955.0 |  |
| 250 of meer werkzame personen | 251035.0 |  |
| 50 tot 250 werkzame personen | 72439.0 |  |
| 0 tot 50 werkzame personen | 55528.0 |  |

<sub>question attested in https://www.bedrijvenbeleidinbeeld.nl/themas/beleidsthemas-verder-onder-de-loep/van-realiteit-naar-ambitie-investeringspatronen-en-structurele-verschillen-in-rd-door-bedrijven-in-de-context-van-de-3-procentdoelstelling</sub>

### What is the evolution of job security and satisfaction with working conditions in the sports and physical activity labor market i…

*Dataset:* [`84434NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84434NED/table) · *type:* `trend` · *id:* `cbs-a0022`

*What the query pins down:* What is the evolution of job security (measured by willingness to continue working until retirement age) and satisfaction with working conditions in the sports and physical activity labor market in t…

```sql
SELECT Perioden_label AS year, measure, ROUND(AVG(CASE WHEN Beroep_label = 'Beroepsniveau 2 (ISCO 2008)' THEN value END), 2) AS job_security, ROUND(AVG(CASE WHEN measure = 'Tevreden met arbeidsomstandigheden' THEN value END), 2) AS satisfaction_with_work_conditions FROM t_84434NED WHERE measure IN ('Tevreden met arbeidsomstandigheden', 'Leeftijd willen doorwerken') AND Perioden_label IN ('2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024') AND Beroep_label = 'Beroepsniveau 2 (ISCO 2008)' GROUP BY Perioden_label, measure ORDER BY Perioden_label;
```

| year | measure | job_security | satisfaction_with_work_conditions |
|---|---|---|---|
| 2014 | Leeftijd willen doorwerken | 62.47 |  |
| 2014 | Tevreden met arbeidsomstandigheden | 71.1 | 71.1 |
| 2015 | Leeftijd willen doorwerken | 62.5 |  |
| 2015 | Tevreden met arbeidsomstandigheden | 70.3 | 70.3 |
| 2016 | Leeftijd willen doorwerken | 62.47 |  |
| 2016 | Tevreden met arbeidsomstandigheden | 70.5 | 70.5 |
| 2017 | Tevreden met arbeidsomstandigheden | 69.77 | 69.77 |
| 2017 | Leeftijd willen doorwerken | 62.2 |  |
| … | 14 more rows | | |

<sub>question attested in https://www.mulierinstituut.nl/publicaties/27946/trendrapport-arbeidsmarkt-sport-en-bewegen-2023/</sub>

### Who is less physically active and/or exercises less than average in the Netherlands?

*Dataset:* [`85563NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85563NED/table) · *type:* `comparison` · *id:* `cbs-a0023`

*What the query pins down:* Which group in the Netherlands, aged 18 to 65, has a lower percentage of people meeting physical activity guidelines than the national average?

```sql
SELECT measure, Leeftijd_label, value AS percentage_less_active FROM t_85563NED WHERE measure = 'Voldoet aan beweegrichtlijn' AND RegioS_label = 'Nederland' AND Leeftijd_label = '18 tot 65 jaar' AND value < (SELECT AVG(value) FROM t_85563NED WHERE measure = 'Voldoet aan beweegrichtlijn' AND RegioS_label = 'Nederland' AND Leeftijd_label = '18 tot 65 jaar') ORDER BY value ASC LIMIT 1;
```

| measure | Leeftijd_label | percentage_less_active |
|---|---|---|
| Voldoet aan beweegrichtlijn | 18 tot 65 jaar | 50.0 |

<sub>question attested in https://www.mulierinstituut.nl/publicaties/27431/beweegdeelname-op-buurtniveau/</sub>

### How much income does government spending redistribute in the Netherlands?

*Dataset:* [`85590NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85590NED/table) · *type:* `rate_or_share` · *id:* `cbs-a0024`

*What the query pins down:* What is the total income redistribution through government spending in the Netherlands for the measure 'Sociale uitkeringen'?

```sql
SELECT value AS income_redistribution_mln_euros FROM t_85590NED WHERE measure = 'Sociale uitkeringen' AND Uitkeringen_label = 'Totaal sociale uitkeringen';
```

| income_redistribution_mln_euros |
|---|
| 67444.0 |
| 66956.0 |
| 68368.0 |
| 70721.0 |
| 72764.0 |
| 76485.0 |
| 81725.0 |
| 88171.0 |
| … | 23 more rows |

<sub>question attested in https://www.cpb.nl/en/inequality-and-redistribution-in-the-netherlands</sub>

### What are the differences between companies in the SME sector in the Netherlands?

*Dataset:* [`86117NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/86117NED/table) · *type:* `distribution` · *id:* `cbs-a0025`

*What the query pins down:* What are the differences between SME companies in the Netherlands in terms of telework, having a website, having ICT security documents, and offering digital goods or services, grouped by company siz…

```sql
SELECT measure, Bedrijfsgrootte_label, ROUND(AVG(value), 2) AS average_value FROM t_86117NED WHERE Bedrijfsgrootte_label IN ('2 tot 250 werkzame personen', '250 tot 500 werkzame personen', '500 of meer werkzame personen') AND measure IN ('Telewerk komt voor', 'Bedrijven met website', 'ICT-veiligheidsdocumenten aanwezig', 'Digitale goederen of diensten') GROUP BY measure, Bedrijfsgrootte_label ORDER BY measure, Bedrijfsgrootte_label;
```

| measure | Bedrijfsgrootte_label | average_value |
|---|---|---|
| Bedrijven met website | 2 tot 250 werkzame personen | 73.0 |
| Bedrijven met website | 250 tot 500 werkzame personen | 97.0 |
| Bedrijven met website | 500 of meer werkzame personen | 98.0 |
| Digitale goederen of diensten | 250 tot 500 werkzame personen | 9.5 |
| Digitale goederen of diensten | 500 of meer werkzame personen | 9.5 |
| ICT-veiligheidsdocumenten aanwezig | 2 tot 250 werkzame personen | 19.0 |
| ICT-veiligheidsdocumenten aanwezig | 250 tot 500 werkzame personen | 86.0 |
| ICT-veiligheidsdocumenten aanwezig | 500 of meer werkzame personen | 92.0 |
| … | 3 more rows | |

<sub>question attested in https://www.cbs.nl/nl-nl/maatwerk/2018/26/het-mkb-bestaat-niet</sub>

### Will the number of international migrants continue to increase in the coming years in Amsterdam?

*Dataset:* [`85752NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85752NED/table) · *type:* `trend` · *id:* `cbs-a0026`

*What the query pins down:* Will the number of international migrants (net migration) continue to increase in the coming years (2025-2065) in Amsterdam?

```sql
SELECT Perioden_label AS period, MAX(value) AS number_of_international_migrants FROM t_85752NED WHERE measure = 'Migratiesaldo (inclusief correcties)' AND Herkomstland_label = 'Totaal geboren buiten Nederland' AND Perioden_label IN ('2025', '2026', '2034', '2035', '2041', '2050', '2056', '2059', '2064', '2065') GROUP BY Perioden_label ORDER BY period;
```

| period | number_of_international_migrants |
|---|---|
| 2025 | 121723.0 |
| 2026 | 106071.0 |
| 2034 | 82652.0 |
| 2035 | 82092.0 |
| 2041 | 78572.0 |
| 2050 | 76122.0 |
| 2056 | 75706.0 |
| 2059 | 75552.0 |
| … | 2 more rows |

<sub>question attested in https://onderzoek.amsterdam.nl/video/lunchsessie-nieuwe-amsterdammers-toen-en-straks</sub>

### What do Dutch people spend on a vacation in 2023?

*Dataset:* [`85302NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85302NED/table) · *type:* `distribution` · *id:* `cbs-a0001`

*What the query pins down:* What was the total spending on vacations by Dutch people in the Netherlands in 2023, for all vacation destinations and all vacation characteristics combined?

```sql
SELECT measure, unit, value, Perioden_label AS period, BestemmingEnSeizoen_label AS destination_season, Vakantiekenmerken_label AS vacation_characteristic FROM t_85302NED WHERE measure = 'Totaal vakantie-uitgaven' AND Perioden_label = '2023' AND BestemmingEnSeizoen_label = 'Totaal vakanties' AND Vakantiekenmerken_label = 'Totaal vakanties' AND Marges_label = 'Waarde';
```

| measure | unit | value | period | destination_season | vacation_characteristic |
|---|---|---|---|---|---|
| Totaal vakantie-uitgaven | mln euro | 22835.0 | 2023 | Totaal vakanties | Totaal vakanties |

<sub>question attested in https://www.landelijkedataalliantie.nl/nl/home/download/cvo-basisrapport-2024?disposition=inline</sub>

### How did the number of holidays and overnight stays change in 2023 compared to 2022 in the Netherlands?

*Dataset:* [`85302NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85302NED/table) · *type:* `comparison` · *id:* `cbs-a0002`

*What the query pins down:* What were the total number of holidays and overnight stays in the Netherlands for the years 2022 and 2023?

```sql
SELECT 
  Perioden_label AS period,
  measure AS measure_type,
  SUM(value) AS count
FROM t_85302NED
WHERE 
  measure IN ('Totaal vakanties', 'Totaal overnachtingen')
  AND Perioden_label IN ('2022', '2023')
  AND BestemmingEnSeizoen_label = 'Totaal vakanties'
  AND Marges_label = 'Waarde'
GROUP BY Perioden_label, measure;
```

| period | measure_type | count |
|---|---|---|
| 2022 | Totaal overnachtingen | 2170.8 |
| 2023 | Totaal vakanties | 294085.0 |
| 2023 | Totaal overnachtingen | 2290.4999999999995 |
| 2022 | Totaal vakanties | 281311.0 |

<sub>question attested in https://www.landelijkedataalliantie.nl/nl/home/download/cvo-basisrapport-2024?disposition=inline</sub>

### What is the prevalence of chronic kidney disease in the Netherlands among adults aged 18 years and older in 2021?

*Dataset:* [`83005NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83005NED/table) · *type:* `rate_or_share` · *id:* `cbs-a0003`

*What the query pins down:* What is the prevalence of chronic kidney disease in the Netherlands among adults aged 18 years and older in 2021, as a percentage?

```sql
SELECT MAX(value) AS prevalence_percentage, Perioden_label AS year, Persoonskenmerken_label AS population FROM t_83005NED WHERE measure = 'Nieraandoening' AND unit = '%' AND Perioden_label = '2021' AND Persoonskenmerken_label = 'Leeftijd: 18 jaar of ouder' GROUP BY Perioden_label, Persoonskenmerken_label;
```

| prevalence_percentage | year | population |
|---|---|---|
| 2.2 | 2021 | Leeftijd: 18 jaar of ouder |

<sub>question attested in https://www.nivel.nl/sites/default/files/bestanden/1004641.pdf</sub>

### What is the current situation regarding the assets and liabilities of Dutch households?

*Dataset:* [`85889NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85889NED/table) · *type:* `distribution` · *id:* `cbs-a0004`

*What the query pins down:* What is the total value of assets and liabilities of Dutch households in the year 2023?

```sql
SELECT Perioden_label AS year, measure, value, unit FROM t_85889NED WHERE measure = 'Totaal' AND Huishoudenskenmerken = 'T001139' AND Perioden = '2023JJ00' LIMIT 1;
```

| year | measure | value | unit |
|---|---|---|---|
| 2023 | Totaal | 3119777.0 | mln euro |

<sub>question attested in https://www.cpb.nl/system/files/cpbmedia/omnidownload/CPB-Achtergronddocument-verscheidenheid-vermogens-Nederlandse-huishoudens-update.pdf</sub>

### Where do Dutch residents go on vacation in 2024?

*Dataset:* [`84367NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84367NED/table) · *type:* `distribution` · *id:* `cbs-a0005`

*What the query pins down:* What are the top vacation destinations for Dutch residents in 2024, as measured by the percentage of Dutch residents who traveled there?

```sql
SELECT Bestemming_label AS destination, value AS percentage_of_dutch_residents FROM t_84367NED WHERE measure = 'Percentage Nederlanders' AND Perioden = '2024JJ00' AND Marges = 'MW00000' ORDER BY value DESC LIMIT 10;
```

| destination | percentage_of_dutch_residents |
|---|---|
| Buitenland | 63.8 |
| Europa totaal | 59.1 |
| West-Europa | 40.2 |
| Zuid-Europa | 27.8 |
| Duitsland (DE) | 16.7 |
| Frankrijk (FR) | 15.5 |
| Spanje (ES) | 13.4 |
| België (BE) | 11.6 |
| … | 2 more rows |

<sub>question attested in https://www.landelijkedataalliantie.nl/nl/home/download/download-rapportage-vakantiegedrag-inwoners-nederland-2024-nl?disposition=inline</sub>

### What do Dutch residents spend on a vacation in 2024?

*Dataset:* [`84367NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84367NED/table) · *type:* `distribution` · *id:* `cbs-a0006`

*What the query pins down:* What is the total vacation expenditure by Dutch residents in the Netherlands for the year 2024, measured in million euros?

```sql
SELECT Perioden_label AS year, measure AS measure_name, unit AS unit_of_measure, value AS total_vacation_expenditures FROM t_84367NED WHERE measure = 'Totaal vakantie-uitgaven' AND Perioden = '2024JJ00' AND Marges = 'MW00000' AND unit = 'mln euro' LIMIT 1;
```

| year | measure_name | unit_of_measure | total_vacation_expenditures |
|---|---|---|---|
| 2024 | Totaal vakantie-uitgaven | mln euro | 19378.0 |

<sub>question attested in https://www.landelijkedataalliantie.nl/nl/home/download/download-rapportage-vakantiegedrag-inwoners-nederland-2024-nl?disposition=inline</sub>

### Which groups are risk groups for work accidents in the Netherlands?

*Dataset:* [`84433NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84433NED/table) · *type:* `ranking_or_list` · *id:* `cbs-a0007`

*What the query pins down:* Which occupational groups in the Netherlands had the highest rate of work accidents resulting in 4 or more days of absence in 2023?

```sql
SELECT Beroep_label AS risk_group, ANY_VALUE(value) AS risk_rate FROM t_84433NED WHERE measure = 'Werknemers ongeval 4 dgn of meer verzuim' AND Perioden = '2023JJ00' AND Beroep_label NOT LIKE '%Totaal%' GROUP BY Beroep_label ORDER BY ANY_VALUE(value) DESC LIMIT 10;
```

| risk_group | risk_rate |
|---|---|
| 1111 Reisbegeleiders | 5.7 |
| 0735 Schilders en metaalspuiters | 5.4 |
| 1214 Vrachtwagenchauffeurs | 4.4 |
| 0634 Militaire beroepen | 4.0 |
| 1222 Vuilnisophalers en dagbladenbezo. | 3.8 |
| 1116 Verleners van overige persoonlij. | 3.5 |
| 0771 Productiemachinebedieners | 3.1 |
| 077 Productiemachinebedieners en asse... | 2.9 |
| … | 2 more rows |

<sub>question attested in https://monitorarbeid.tno.nl/wp-content/uploads/sites/16/2023/10/Arbeidsongevallen-in-Nederland-2011.pdf</sub>

### Are there differences between small and medium-sized enterprises (businesses with fewer than 250 employees) and large enterprises…

*Dataset:* [`84985NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84985NED/table) · *type:* `comparison` · *id:* `cbs-a0008`

*What the query pins down:* Are there differences between small and medium-sized enterprises (50 to 250 employees) and large enterprises (250 or more employees) in their research and development (R&D) activities in the Netherla…

```sql
SELECT 
  Bedrijfsgrootte_label AS enterprise_size,
  measure,
  unit,
  SUM(value) AS total_value
FROM t_84985NED
WHERE 
  Bedrijfsgrootte_label IN ('250 of meer werkzame personen', '50 tot 250 werkzame personen')
  AND measure IN ('Ondernemingen met eigen R&D activiteiten', 'Uitgaven voor eigen R&D activiteiten', 'Werkzame personen')
  AND Perioden_label = '2024'
GROUP BY 
  Bedrijfsgrootte_label,
  measure,
  unit;
```

| enterprise_size | measure | unit | total_value |
|---|---|---|---|
| 50 tot 250 werkzame personen | Uitgaven voor eigen R&D activiteiten | mln euro | 14920.0 |
| 250 of meer werkzame personen | Uitgaven voor eigen R&D activiteiten | mln euro | 49786.0 |
| 50 tot 250 werkzame personen | Werkzame personen | aantal | 190383.0 |
| 250 of meer werkzame personen | Werkzame personen | aantal | 358213.0 |

<sub>question attested in https://www.mejudice.nl/artikelen/detail/ondernemingen-onder-nederlandse-zeggenschap-investeren-het-meest-in-onderzoek-en-ontwikkeling-in-nederland</sub>

### What is the development of municipal expenditures on sports facilities between 2018 and 2020 in the Netherlands?

*Dataset:* [`84138NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84138NED/table) · *type:* `trend` · *id:* `cbs-a0009`

*What the query pins down:* What is the development of municipal expenditures on sports facilities (specifically housing costs) between 2018 and 2020 in the Netherlands, measured in millions of euros?

```sql
SELECT Perioden_label AS year, value AS municipal_expenditures_millions_euro FROM t_84138NED WHERE measure = 'Huisvestingskosten' AND unit = 'mln euro' AND Perioden IN ('2018JJ00', '2019JJ00', '2020JJ00') AND Sportaccommodaties = 'T001417';
```

| year | municipal_expenditures_millions_euro |
|---|---|
| 2018 | 260.0 |
| 2020 | 246.0 |

<sub>question attested in https://www.mulierinstituut.nl/publicaties/26416/jaarrapport-duurzame-sportinfrastructuur-2021/</sub>

### Who is less physically active and/or exercises less than average in the Netherlands?

*Dataset:* [`85563NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85563NED/table) · *type:* `comparison` · *id:* `cbs-a0010`

*What the query pins down:* What percentage of Nederlanders in the Netherlands, overall (Totaal), do not meet the physical activity guidelines (Voldoet aan beweegrichtlijn)?

```sql
SELECT measure, Marges_label, value AS percentage_less_active FROM t_85563NED WHERE measure = 'Voldoet aan beweegrichtlijn' AND RegioS_label = 'Nederland' AND Leeftijd_label = 'Totaal' AND Marges_label = 'Waarde' AND unit = '%';
```

| measure | Marges_label | percentage_less_active |
|---|---|---|
| Voldoet aan beweegrichtlijn | Waarde | 47.5 |

<sub>question attested in https://www.mulierinstituut.nl/publicaties/27431/beweegdeelname-op-buurtniveau/</sub>

### Did the post-WWII baby boom in the Netherlands compensate for the births that did not occur during WWII?

*Dataset:* [`85524NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85524NED/table) · *type:* `yes_no` · *id:* `cbs-0001`

*What the query pins down:* Compare average annual live births (x1000) for 1935-1939, 1940-1944 and 1946-1950, and report the 1946-1950 surplus relative to five pre-war years.

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

| pre_war_1935_39 | war_1940_44 | boom_1946_50 | surplus_vs_prewar |
|---|---|---|---|
| 174.2 | 197.2 | 253.0 | 1659.0 |

<sub>question attested in https://papers.tinbergen.nl/</sub>

