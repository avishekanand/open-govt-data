# Research questions extracted from publications

Real questions that published studies set out to answer, recovered automatically
from the full text of the publications themselves — not generated from metadata.

**How these were produced.** Every publication in the CBS publications workbook
was crawled (`cbs.pub_download`), its complete text extracted, and each chunk
carrying a CBS/Eurostat signal — plus the opening chunks, where a paper states
its aim — passed to Qwen3-32B under a constrained JSON schema
(`enrich.pub_evidence`). For every question the model must quote a **witness
sentence verbatim from the document**; that quote is then checked to actually
occur in the source text and flagged if it does not. Dataset mentions are
resolved against the 12,308-dataset catalogue by `enrich.pub_link`.

Questions are shown in English (machine-translated by the same model), with the
original underneath. **Witness sentences are never translated** — they are
evidence, verified by exact match against the source text, so a translated quote
could not be checked.

| | |
|---|---|
| documents processed | 1,176 |
| research questions extracted | 3,276 |
| documents yielding ≥1 question | 918 |
| questions with a verified witness | 91.2% |
| dataset mentions | 3,329 |
| model | Qwen3-32B (vLLM, schema-constrained) |

---

## Part 1 — Questions with the datasets the study used

The benchmark-shaped cases: a real research question, plus the CBS tables the
publication actually cited, resolved to catalogue codes. A retrieval system
should be able to get from the question to those tables.

### 1. Kamerstuk 36600-XV, nr. 8 | Overheid.nl > Officiële bekendmakingen

*Source:* <https://zoek.officielebekendmakingen.nl/kst-36600-XV-8.html>

- **Q:** What percentage of migrant workers work at 100 percent, 105 percent, 110 percent, 115 percent, and 120 percent of the Wml? What is the total?
  - *original:* Hoeveel procent van de arbeidsmigranten werkt op 100 procent, 105 procent, 110 procent, 115 procent en 120 procent Wml? Hoe groot is het totaal?
  - *witness:* “Vraag 1 Hoeveel procent van de arbeidsmigranten werkt op 100 procent, 105 procent, 110 procent, 115 procent en 120 procent Wml? Hoe groot is het totaal?”
- **Q:** How large are the shortcomings arising from implementation information of Social Affairs and Employment (SZW)? In what way have these been addressed? What specific measures have b…
  - *original:* Hoe groot zijn de tegenvallers naar aanleiding van uitvoeringsinformatie van Sociale Zaken en Werkgelegenheid (SZW)? Op welke manier zijn deze gedekt? Welke maatregelen zijn hierv…
  - *witness:* “Vraag 2 Hoe groot zijn -=de tegenvallers naar aanleiding van uitvoeringsinformatie van Sociale Zaken en Werkgelegenheid (SZW)? Op welke manier zijn deze gedekt? Welke maatregelen zijn hiervoor precie…”

  **Datasets used** (resolved to the catalogue):

  - [`85264NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85264NED/table) — Arbeidsdeelname; kerncijfers
  - [`81588NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/81588NED/table) — Bedrijven; bedrijfsgrootte en rechtsvorm
  - [`85278NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85278NED/table) — Werkzame beroepsbevolking; positie in de werkkring
  - [`80598ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/80598ned/table) — Leden van vakverenigingen; geslacht en leeftijd
  - [`70061ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/70061ned/table) — Historie leden vakverenigingen
  - *microdata registers (no public table):* POLIS

### 2. open.overheid.nl

*Source:* <https://open.overheid.nl/documenten/c82b508c-6e93-4791-a895-93bb1957c49e/file>

- **Q:** How can the stacking of social issues be better measured in the distribution of the municipal fund for the social domain?
  - *original:* Hoe kan de stapeling van sociale problematiek beter worden gemeten in de verdeling van het gemeentefonds voor het sociaal domein?
  - *witness:* “Het onderzoek heeft als doel te bezien of in de verdeling van het gemeentefonds voor het sociaal domein tot een betere maatstaf / maatstaven kan worden gekomen voor de gemeentelijke kosten die voortk…”
- **Q:** To what extent does the current standard for regional center function lead to an overestimation of the center function in the social domain?
  - *original:* In hoeverre leidt de huidige maatstaf voor regionale centrumfunctie tot een overwaardering van de centrumfunctie in het sociaal domein?
  - *witness:* “Daarnaast heeft de ROB toen de vraag gesteld: “In hoeverre leidt dit tot een overwaardering van de centrumfunctie?””
- **Q:** How can a non-linear measure at the sub-municipal level be developed to describe the stacking of social issues?
  - *original:* Hoe kan een niet-lineaire maatstaf op subgemeentelijk niveau worden ontwikkeld om de stapeling van sociale problematiek te beschrijven?
  - *witness:* “Stapeling van sociale problematiek vraagt om een niet-lineaire maatstaf op subgemeentelijk niveau”

  **Datasets used** (resolved to the catalogue):

  - [`83841NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83841NED/table) — Laag en langdurig laag inkomen van huishoudens; huishoudenskenmerken
  - [`85383NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85383NED/table) — Maatstaven Financiële-verhoudingswet (Fvw)
  - [`85644NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85644NED/table) — Bevolking; geslacht, leeftijd, nationaliteit en regio, 1 januari
  - *microdata registers (no public table):* BRP, GBA, HOOGSTEOPLTAB, INHATAB

### 3. ifn.se

*Source:* <https://www.ifn.se/media/nxdngkyd/wp1503.pdf>

- **Q:** What is the magnitude of the response to a kink in the Dutch tax schedule?
  - *witness:* “We exploit a kink in the Dutch tax schedule where the marginal corporate income tax (CIT) rate increases by 5 percentage points (pp) for taxable income above e200,000.”
- **Q:** What are the main predictors of tax responsiveness?
  - *witness:* “We employ bunching (Chetty, Friedman, Olsen, and Pistaferri, 2011; Saez, 2010) and probit analysis to uncover: (i) the magnitude of the response, as measured by the CETI; (ii) the channels of the res…”
- **Q:** Is there persistence in behavioral responses to tax incentives?
  - *witness:* “We employ bunching (Chetty, Friedman, Olsen, and Pistaferri, 2011; Saez, 2010) and probit analysis to uncover: ... (iv) persistence in behavioral responses;”

  **Datasets used** (resolved to the catalogue):

  - [`85067NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85067NED/table) — Regions in the Netherlands 2022
  - [`85755NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85755NED/table) — Regions in the Netherlands 2024
  - [`81588NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/81588NED/table) — Bedrijven; bedrijfsgrootte en rechtsvorm

### 4. Het inkoopvolume van de Nederlandse overheid

*Source:* <https://open.overheid.nl/documenten/ronl-aa3fbd0d-5849-4589-b966-1e2156416862/pdf>

- **Q:** What is the development of the total estimated procurement volume of the Dutch governments in the years 2017, 2018, and 2019?
  - *original:* Wat is de ontwikkeling van het totale geschatte inkoopvolume van de Nederlandse overheden in de jaren 2017, 2018 en 2019?
  - *witness:* “3.1 Hoe ontwikkelt het totale geschatte inkoopvolume zich in de jaren 2017, 2018 en 2019?”
- **Q:** What is the development of the procurement volume of awarded contracts below and above the threshold in the years 2017, 2018, and 2019?
  - *original:* Wat is de ontwikkeling van het inkoopvolume van gegunde opdrachten onder en boven de drempel in de jaren 2017, 2018 en 2019?
  - *witness:* “3.2 Hoe ontwikkelt het totale inkoopvolume van gegunde opdrachten onder en boven de drempel zich in de jaren 2017, 2018 en 2019?”
- **Q:** What is the participation of SMEs in government contracts in the years 2017, 2018, and 2019?
  - *original:* Wat is de deelname van het mkb aan overheidsopdrachten in de jaren 2017, 2018 en 2019?
  - *witness:* “5 Wat is de deelname van het mkb aan overheidsopdrachten?”

  **Datasets used** (resolved to the catalogue):

  - [`85951NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85951NED/table) — Groeirekeningen; nationale rekeningen
  - [`84122NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84122NED/table) — Overheidsuitgaven en bestedingen; functies, transacties, overheidssec…
  - [`82563NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82563NED/table) — Overheid; inkomsten en uitgaven 1995-2017

### 5. magontslag.nl

*Source:* <https://www.magontslag.nl/content/toelichtingseo.pdf>

- **Q:** What factors determine the expected remaining employment duration upon dismissal and how do these factors influence the outcome of the calculation model?
  - *original:* Welke factoren bepalen de verwachte resterende baanduur bij ontslag en hoe beïnvloeden deze factoren de uitkomst van het rekenmodel?
  - *witness:* “Hieronder volgt een overzicht van factoren waarvan de invloed op de verwachte potentiële resterende baanduur bij ontslag in het rekenmodel zijn opgenomen.”

  **Datasets used** (resolved to the catalogue):

  - [`83859NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83859NED/table) — Gebieden in Nederland 2018
  - [`80472ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/80472ned/table) — Vacatures; SBI 2008; naar economische activiteit en bedrijfsgrootte
  - [`82309NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82309NED/table) — Arbeidsdeelname; kerncijfers, 2003-2022
  - *microdata registers (no public table):* POLIS

### 6. Evaluatie coronasteun cultuursector

*Source:* <https://www.seo.nl/wp-content/uploads/2023/06/2023-48-Evaluatie-coronasteun-cultuursector-definitief.pdf>

- **Q:** What is the effectiveness and efficiency of the specific measures taken for the cultural sector in light of the stated objectives?
  - *original:* Wat is de doeltreffendheid en doelmatigheid van de getroffen specifieke maatregelen voor de culturele sector in het licht van de gestelde doelstellingen?
  - *witness:* “Hoofddoelstelling van voorliggende evaluatie is om zicht te geven op: ● de doeltreffendheid en doelmatigheid van de getroffen specifieke maatregelen voor de culturele sector in het licht van de geste…”
- **Q:** What intended and unintended side effects (positive and negative) have the implemented measures had?
  - *original:* Welke bedoelde en onbedoelde neveneffecten (positief en negatief) hebben de getroffen maatregelen?
  - *witness:* “● bedoelde en onbedoelde neveneffecten (positief en negatief) van de getroffen maatregelen”
- **Q:** How were the measures implemented, what were the substantive considerations involved, and what was the collaboration between the Ministry of Education, Culture and Science (OCW) w…
  - *original:* Hoe is de totstandkoming van de maatregelen, de inhoudelijke overwegingen daarbij en de samenwerking van het ministerie van Onderwijs, Cultuur en Wetenschap (OCW) met de culturele…
  - *witness:* “● totstandkoming van de maatregelen, de inhoudelijke overwegingen daarbij en de samenwerking van het ministerie van Onderwijs, Cultuur en Wetenschap (OCW) met de culturele sector en andere betrokkene…”

  **Datasets used** (resolved to the catalogue):

  - [`70810NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/70810NED/table) — Professionele podia; werkgelegenheid, baten en lasten
  - [`82469NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82469NED/table) — Openbare bibliotheken; leden, collectie, uitleningen vanaf 1900
  - *microdata registers (no public table):* INPATAB

### 7. untitled

*Source:* <https://www.instituutgak.nl/wp-content/uploads/2021/01/proefschrift_Kristiansen_email.pdf>

- **Q:** How do social networks affect benefit receipt dynamics in the Netherlands?
  - *witness:* “Contacts with Benefits: How Social Networks Affect Benefit Receipt Dynamics in the Netherlands”
- **Q:** What is the relationship between social networks and the receipt of social assistance, unemployment, and disability benefits in the Netherlands?
  - *witness:* “We focus on the receipt of beneﬁts among the working-age population in the Netherlands, which includes social assistance, unemployment, and disability and sickness beneﬁts.”
- **Q:** What factors contribute to the recurrence of benefit receipt over the life course?
  - *witness:* “There is also quite some research indicating that beneﬁt receipt is a recurrent phenomenon over the life course for some people.”

  **Datasets used** (resolved to the catalogue):

  - [`84721NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84721NED/table) — Regions in the Netherlands 2020
  - [`83553NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83553NED/table) — Regions in the Netherlands 2017
  - *microdata registers (no public table):* SECMBUS

### 8. open.overheid.nl

*Source:* <https://open.overheid.nl/documenten/40aeb730-c37c-4635-93d6-14b3f82a8c42/file>

- **Q:** What is needed to set up a new version of the venture capital scheme in which individuals can provide funds to SMEs with tax benefits?
  - *original:* Wat is nodig om een nieuwe versie van de durfkapitaalregeling op te zetten waarbij particulieren met belastingvoordelen geld kunnen verstrekken aan mkb-bedrijven?
  - *witness:* “De Tweede Kamer heeft twee moties ingediend die vragen om (1) te onderzoeken wat nodig is om een nieuwe versie van de durfkapitaalregeling op te zetten waarbij particulieren met belastingvoordelen ge…”

  **Datasets used** (resolved to the catalogue):

  - [`84105NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84105NED/table) — Bbp, productie en bestedingen; kwartalen, waarden, na, 1995 - 2024-I
  - [`83131NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83131NED/table) — Consumentenprijzen; prijsindex 2015=100, 1996-2025

---

## Part 2 — A wider sample of extracted questions

Every question below carries a witness sentence verified against the document
text. Grouping is a rough heuristic on keywords in the *source publication's
title*, not on the question itself, so an occasional question sits under a
neighbouring theme.

### Labour, income and social security

- What is the predictive power of the model based on current income compared to more comprehensive models?
  <br/><sub>*Wat is de voorspelkracht van het model op basis van het huidig inkomen vergeleken met meer uitgebreide modellen?*</sub>
  <br/><sub>— [Modellen voor bestendig inkomen - SEO Economisch Onderzoek](https://www.seo.nl/publicaties/modellen-voor-bestendig-inkomen/)</sub>
- What are the quantitative and qualitative shortages in the Dutch cybersecurity labor market?
  <br/><sub>*Wat zijn de kwantitatieve en kwalitatieve tekorten op de Nederlandse cybersecurity-arbeidsmarkt?*</sub>
  <br/><sub>— [Onderzoek: Onderwijs en Arbeidsmarkt Cybersecurity - Pl...](https://www.ptvt.nl/publicatie/onderzoek-onderwijs-en-arbeidsmarkt-cybersecurity)</sub>
- What are the career and labor market prospects of vocational education graduates in ICT?
  <br/><sub>*Wat zijn de loopbaan- en arbeidsmarktperspectieven van mbo-gediplomeerden in de ICT?*</sub>
  <br/><sub>— [Rapportages | Loopbaan en arbeidsmarktperspectief van mbo-gediplomeerde…](https://onderzoek010.nl/document/Loopbaan-en-arbeidsmarktperspectief-van-mbo-gediplomeerden-in-de-ICT,-Risbo)</sub>
- Which policy options are promising for improving labor market outcomes for people with a migration background?
  <br/><sub>*Welke beleidsopties zijn kansrijk om de arbeidsmarktuitkomsten van personen met een migratieachtergrond te verbeteren?*</sub>
  <br/><sub>— [Kansrijk integratiebeleid op de arbeidsmarkt](https://www.cpb.nl/system/files/cpbmedia/omnidownload/Kansrijk_integratiebeleid_op_de_arbeidsmarkt2.pdf)</sub>
- How do changes in the tax system affect the tax burden of different income groups?
  <br/><sub>*Hoe beïnvloeden wijzigingen in het belastingstelsel de belastingdruk van verschillende inkomensgroepen?*</sub>
  <br/><sub>— [Effecten van belastinghervormingen op de belastingdruk van verschillend…](https://www.cpb.nl/effecten-van-belastinghervormingen-op-de-belastingdruk-van-verschillende-inkomensgroepen)</sub>

### Health and care

- How visible is aging in health care expenditures?
  <br/><sub>*Hoe zichtbaar is de vergrijzing in de uitgaven aan gezondheidszorg?*</sub>
  <br/><sub>— [Effecten van vergrijzing steeds beter zichtbaar in uitgaven gezondheids…](https://www.rivm.nl/nieuws/effecten-van-vergrijzing-steeds-beter-zichtbaar-in-uitgaven-gezondheidszorg)</sub>
- What are the regional differences in pelvic floor care, and are these desirable or undesirable?
  <br/><sub>*Wat zijn de regionale verschillen in bekkenbodemzorg en zijn deze wenselijk of ongewenst?*</sub>
  <br/><sub>— [SKMS NVOG project ‘inventarisatie zorgpaden rondom bekkenbodemzorg’ – N…](https://www.nvog.nl/skms-nvog-project-inventarisatie-zorgpaden-rondom-bekkenbodemzorg/)</sub>
- What bottlenecks have candidates, students, and graduates encountered at different points in time?
  <br/><sub>*Welke knelpunten kandidaten, studenten en afgestudeerden op verschillende momenten tegengekomen?*</sub>
  <br/><sub>— [Verkenning problematiek zorgopleidingen hbo/wo Caribische studenten | K…](https://kohnstamminstituut.nl/rapport/verkenning-problematiek-zorgopleidingen-hbo-wo-caribische-studenten/)</sub>
- How do limited financial resources influence the use of healthcare?
  <br/><sub>*Hoe beïnvloedt beperkte financiële middelen het gebruik van zorg?*</sub>
  <br/><sub>— [Weinig te besteden, minder zorg? Zorguitgaven van mensen met beperkte f…](https://www.cpb.nl/publicatie/weinig-te-besteden-minder-zorg-zorguitgaven-van-mensen-met-beperkte-financiele-middelen)</sub>
- How did the use of Wmo parental care develop over the years?
  <br/><sub>*Hoe ontwikkelde het gebruik van Wmo-ouderenzorg zich in de loop der jaren?*</sub>
  <br/><sub>— [Gemeentelijke trends in het gebruik van Wmo-ouderenzorg | CPB Website](https://www.cpb.nl/gemeentelijke-trends-in-het-gebruik-van-wmo-ouderenzorg)</sub>

### Education and youth

- What is the expected growing demand for cybersecurity personnel?
  <br/><sub>*Wat zijn de verwachte groeiende vraag naar cybersecuritypersoneel?*</sub>
  <br/><sub>— [Onderzoek: Onderwijs en Arbeidsmarkt Cybersecurity - Pl...](https://www.ptvt.nl/publicatie/onderzoek-onderwijs-en-arbeidsmarkt-cybersecurity)</sub>
- To what extent is the national picture of increased use of youth support also applicable in Enschede?
  <br/><sub>*In welke mate is het landelijke beeld van toename van jeugdhulpgebruik ook van toepassing in Enschede?*</sub>
  <br/><sub>— [Verdiepende analyses in- en uitstroom naar Jeugdhulp - Kennispunt Twente](https://kennispunttwente.nl/publicaties/sociaal-domein/verdiepende-analyses-in-en-uitstroom-naar-jeugdhulp)</sub>
- To what extent are the data on pre-school education that schools provide to DUO representative?
  <br/><sub>*In hoeverre zijn de gegevens over voorschoolse educatie die scholen aan DUO leveren representatief?*</sub>
  <br/><sub>— [Kwaliteit van data over voorschoolse educatie in BRON | Kohnstamm Insti…](https://kohnstamminstituut.nl/rapport/kwaliteit-van-data-over-voorschoolse-educatie-in-bron/)</sub>
- What are the obstacles Caribbean students encounter in admission and during their education?
  <br/><sub>*Wat zijn de obstakels die Caribische studenten ondervinden bij toelating en tijdens de opleiding?*</sub>
  <br/><sub>— [Verkenning problematiek zorgopleidingen hbo/wo Caribische studenten | K…](https://kohnstamminstituut.nl/rapport/verkenning-problematiek-zorgopleidingen-hbo-wo-caribische-studenten/)</sub>
- What is the wage difference between teachers in secondary education and comparable employees in the market sector?
  <br/><sub>*Wat is het loonverschil tussen leraren in het voortgezet onderwijs en vergelijkbare werknemers in de marktsector?*</sub>
  <br/><sub>— [Wat een leraar in het voortgezet onderwijs verdient - SEO Economisch On…](https://www.seo.nl/publicaties/wat-een-leraar-in-het-voortgezet-onderwijs-verdient/)</sub>

### Housing, regions and liveability

- Which interventions are carried out within the framework of the program 'Een nieuw bestaan, een nieuwe baan'?
  <br/><sub>*Welke interventies worden in het kader van het programma ‘Een nieuw bestaan, een nieuwe baan’ uitgevoerd?*</sub>
  <br/><sub>— [Eindrapport - Programma Een nieuw bestaan, een nieuwe baan](https://www.regioplan.nl/wp-content/uploads/2022/07/18128-Eindrapport-EenNieuwBestaanEenNieuweBaan-Regioplan-7juli22.pdf)</sub>
- What is the impact of the JOGG approach on overweight and physical activity behavior in children and youth aged 2 to 19 years?
  <br/><sub>*Wat is de invloed van de JOGG-aanpak op het overgewicht en beweeggedrag bij kinderen en jongeren tussen de 2 en 19 jaar?*</sub>
  <br/><sub>— [Daling overgewicht in JOGG-buurten | RIVM](https://www.rivm.nl/nieuws/daling-overgewicht-in-jogg-buurten)</sub>
- Which areas in the Netherlands have a structural pressure on their livability, and are there more or fewer of them compared to the previous measurement?
  <br/><sub>*Welke gebieden in Nederland hebben een structurele druk op hun leefbaarheid en zijn dit er meer of minder dan bij de vorige meting?*</sub>
  <br/><sub>— [leefbaarometer.nl](https://www.leefbaarometer.nl/resources/Analyserapport-Leefbaarheid-onder-de-loep-2022.pdf)</sub>
- How should subsidies be distributed more evenly across the country?
  <br/><sub>*Hoe moet de subsidies evenrediger over het land worden verdeeld?*</sub>
  <br/><sub>— [Boekman #139: Cultuur in de regio - Boekmanstichting](https://www.boekman.nl/tijdschrift/boekman-139-cultuur-in-de-regio/)</sub>
- Why are more and more Amsterdam residents moving to the region?
  <br/><sub>*Waarom verhuizen steeds meer Amsterdammers naar de regio?*</sub>
  <br/><sub>— [In de stad blijven of verhuizen naar de regio | Website Onderzoek en St…](https://onderzoek.amsterdam.nl/publicatie/in-de-stad-blijven-of-verhuizen-naar-de-regio-proefschrift)</sub>

### Migration, population and diversity

- Which conditions lead to the highest healthcare expenditures?
  <br/><sub>*Welke aandoeningen leiden tot de hoogste zorguitgaven?*</sub>
  <br/><sub>— [Effecten van vergrijzing steeds beter zichtbaar in uitgaven gezondheids…](https://www.rivm.nl/nieuws/effecten-van-vergrijzing-steeds-beter-zichtbaar-in-uitgaven-gezondheidszorg)</sub>
- What are the causes of the large and persistent differences in labor participation between persons with and without a migration background?
  <br/><sub>*Wat zijn de oorzaken van de grote en hardnekkige verschillen in de arbeidsparticipatie tussen personen met en zonder migratieachtergrond?*</sub>
  <br/><sub>— [Kansrijk integratiebeleid op de arbeidsmarkt](https://www.cpb.nl/system/files/cpbmedia/omnidownload/Kansrijk_integratiebeleid_op_de_arbeidsmarkt2.pdf)</sub>
- What are the demographic developments in the world, Europe, and the Netherlands?
  <br/><sub>*Wat zijn de demografische ontwikkelingen in de wereld, Europa en Nederland?*</sub>
  <br/><sub>— [Bevolkingsvraagstukken in Nederland anno 2012](https://publ.nidi.nl/output/books/nidi-book-86.pdf)</sub>
- What are the main overarching key conclusions and insights from this third measurement?
  <br/><sub>*Wat zijn de belangrijkste overkoepelende hoofdconclusies en inzichten uit deze derde meting?*</sub>
  <br/><sub>— [Monitor van gelijkwaardige kansen en evenredige posities op de arbeidsm…](https://open.overheid.nl/documenten/8dda65b3-1cda-43d3-bf4f-211d3dc9336b/file)</sub>
- What is the development of the number of working migrant workers in the province of Limburg?
  <br/><sub>*Wat is de ontwikkeling van het aantal werkende arbeidsmigranten in de provincie Limburg?*</sub>
  <br/><sub>— [Onderzoek internationale arbeidsmigranten provincie Limburg](https://www.limburg.nl/publish/pages/9299/eindrapportage_decisio_onderzoek_arbeidsmigranten_limburg_2024_03.pdf)</sub>

### Business, economy and innovation

- What is the current and future labor market position of graduates from bachelor's programs in Economics and Business Administration?
  <br/><sub>*Wat is de huidige en toekomstige arbeidsmarktpositie van afgestudeerden van bacheloropleidingen in de Economie en de Bedrijfskunde?*</sub>
  <br/><sub>— [Kabinet onderschat arbeidsmarktpositie studenten Economie en Bedrijfsku…](https://esb.nu/kabinet-onderschat-arbeidsmarktpositie-studenten-economie-en-bedrijfskunde/)</sub>
- What is the position of graduates after economic follow-up education?
  <br/><sub>*Hoe zit het met de positie van afgestudeerden na economische vervolgopleidingen?*</sub>
  <br/><sub>— [Arbeidsmarktperspectieven economieopleidingen op het mbo, hbo en wo - E…](https://esb.nu/arbeidsmarktperspectieven-van-economieopleidingen-op-het-mbo-hbo-en-wo/)</sub>
- Do companies under Dutch ownership conduct more research and development in the Netherlands than companies under foreign ownership?
  <br/><sub>*Voeren ondernemingen onder Nederlandse zeggenschap meer onderzoek en ontwikkeling uit in Nederland dan ondernemingen onder buitenlandse zeggenschap?*</sub>
  <br/><sub>— [Ondernemingen onder Nederlandse zeggenschap investeren het meest in ond…](https://www.mejudice.nl/artikelen/detail/ondernemingen-onder-nederlandse-zeggenschap-investeren-het-meest-in-onderzoek-en-ontwikkeling-in-nederland)</sub>
- Where exactly did the provided support end up?
  <br/><sub>*Waar is de geboden steun precies terechtgekomen?*</sub>
  <br/><sub>— [Ook ondernemingen met omzetgroei ontvingen steun - ESB](https://esb.nu/ook-ondernemingen-met-omzetgroei-ontvingen-steun/)</sub>
- How much time does an itinerant trader need to recoup his investments with a reasonable return?
  <br/><sub>*Hoeveel tijd heeft een ambulante handelaar nodig om zijn investeringen terug te verdienen met een redelijk rendement?*</sub>
  <br/><sub>— [Bedrijfslevenbeleid | Tweede Kamer der Staten-Generaal](https://www.tweedekamer.nl/kamerstukken/brieven_regering/detail?id=2025Z11154&did=2025D25552)</sub>

---

## Caveats

- **Witness verification is a floor, not a guarantee.** It proves the sentence
  exists in the document; it does not prove the model read it correctly. About
  9% of questions carry a witness that could not be located; those are excluded
  from Part 2 and flagged in Part 1.
- **Not every 'question' is a research question.** The workbook includes
  parliamentary documents (*Kamerstukken*), whose numbered `Vraag 1…` items are
  parliamentary questions to a minister rather than a study's research aim.
  They are genuine questions answered with CBS data, but a different genre.
- **Dataset linking resolves ~10% of mentions.** Most of the rest are microdata
  registers, surveys and CBS statistical programmes with no StatLine table id.
  The addressable gap is tables named in prose that lexical matching misses on
  vocabulary; embedding matching over the enriched English titles would close
  much of it.
- **Questions are chunk-scoped** — taken from the opening of each document,
  where the aim is normally stated. A question buried later may be missed.
- **Not every document is CBS-related.** 505 of 1,176 were classified as using
  no CBS data.

*Regenerate:* `python -m enrich.pub_evidence --resume && python scripts/make_question_examples.py`
