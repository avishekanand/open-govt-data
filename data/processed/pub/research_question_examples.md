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

Questions appear in their original language (mostly Dutch).

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

- **Q:** Hoeveel procent van de arbeidsmigranten werkt op 100 procent, 105 procent, 110 procent, 115 procent en 120 procent Wml? Hoe groot is het totaal?
  - *witness:* “Vraag 1 Hoeveel procent van de arbeidsmigranten werkt op 100 procent, 105 procent, 110 procent, 115 procent en 120 procent Wml? Hoe groot is het totaal?”
- **Q:** Hoe groot zijn de tegenvallers naar aanleiding van uitvoeringsinformatie van Sociale Zaken en Werkgelegenheid (SZW)? Op welke manier zijn deze gedekt? Welke maatregelen zijn hierv…
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

- **Q:** Hoe kan de stapeling van sociale problematiek beter worden gemeten in de verdeling van het gemeentefonds voor het sociaal domein?
  - *witness:* “Het onderzoek heeft als doel te bezien of in de verdeling van het gemeentefonds voor het sociaal domein tot een betere maatstaf / maatstaven kan worden gekomen voor de gemeentelijke kosten die voortk…”
- **Q:** In hoeverre leidt de huidige maatstaf voor regionale centrumfunctie tot een overwaardering van de centrumfunctie in het sociaal domein?
  - *witness:* “Daarnaast heeft de ROB toen de vraag gesteld: “In hoeverre leidt dit tot een overwaardering van de centrumfunctie?””
- **Q:** Hoe kan een niet-lineaire maatstaf op subgemeentelijk niveau worden ontwikkeld om de stapeling van sociale problematiek te beschrijven?
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

- **Q:** Wat is de ontwikkeling van het totale geschatte inkoopvolume van de Nederlandse overheden in de jaren 2017, 2018 en 2019?
  - *witness:* “3.1 Hoe ontwikkelt het totale geschatte inkoopvolume zich in de jaren 2017, 2018 en 2019?”
- **Q:** Wat is de ontwikkeling van het inkoopvolume van gegunde opdrachten onder en boven de drempel in de jaren 2017, 2018 en 2019?
  - *witness:* “3.2 Hoe ontwikkelt het totale inkoopvolume van gegunde opdrachten onder en boven de drempel zich in de jaren 2017, 2018 en 2019?”
- **Q:** Wat is de deelname van het mkb aan overheidsopdrachten in de jaren 2017, 2018 en 2019?
  - *witness:* “5 Wat is de deelname van het mkb aan overheidsopdrachten?”

  **Datasets used** (resolved to the catalogue):

  - [`85951NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85951NED/table) — Groeirekeningen; nationale rekeningen
  - [`84122NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84122NED/table) — Overheidsuitgaven en bestedingen; functies, transacties, overheidssec…
  - [`82563NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82563NED/table) — Overheid; inkomsten en uitgaven 1995-2017

### 5. magontslag.nl

*Source:* <https://www.magontslag.nl/content/toelichtingseo.pdf>

- **Q:** Welke factoren bepalen de verwachte resterende baanduur bij ontslag en hoe beïnvloeden deze factoren de uitkomst van het rekenmodel?
  - *witness:* “Hieronder volgt een overzicht van factoren waarvan de invloed op de verwachte potentiële resterende baanduur bij ontslag in het rekenmodel zijn opgenomen.”

  **Datasets used** (resolved to the catalogue):

  - [`83859NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83859NED/table) — Gebieden in Nederland 2018
  - [`80472ned`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/80472ned/table) — Vacatures; SBI 2008; naar economische activiteit en bedrijfsgrootte
  - [`82309NED`](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82309NED/table) — Arbeidsdeelname; kerncijfers, 2003-2022
  - *microdata registers (no public table):* POLIS

### 6. Evaluatie coronasteun cultuursector

*Source:* <https://www.seo.nl/wp-content/uploads/2023/06/2023-48-Evaluatie-coronasteun-cultuursector-definitief.pdf>

- **Q:** Wat is de doeltreffendheid en doelmatigheid van de getroffen specifieke maatregelen voor de culturele sector in het licht van de gestelde doelstellingen?
  - *witness:* “Hoofddoelstelling van voorliggende evaluatie is om zicht te geven op: ● de doeltreffendheid en doelmatigheid van de getroffen specifieke maatregelen voor de culturele sector in het licht van de geste…”
- **Q:** Welke bedoelde en onbedoelde neveneffecten (positief en negatief) hebben de getroffen maatregelen?
  - *witness:* “● bedoelde en onbedoelde neveneffecten (positief en negatief) van de getroffen maatregelen”
- **Q:** Hoe is de totstandkoming van de maatregelen, de inhoudelijke overwegingen daarbij en de samenwerking van het ministerie van Onderwijs, Cultuur en Wetenschap (OCW) met de culturele…
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

- **Q:** Wat is nodig om een nieuwe versie van de durfkapitaalregeling op te zetten waarbij particulieren met belastingvoordelen geld kunnen verstrekken aan mkb-bedrijven?
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

- Wat is de voorspelkracht van het model op basis van het huidig inkomen vergeleken met meer uitgebreide modellen?
  <br/><sub>— [Modellen voor bestendig inkomen - SEO Economisch Onderzoek](https://www.seo.nl/publicaties/modellen-voor-bestendig-inkomen/)</sub>
- Wat zijn de kwantitatieve en kwalitatieve tekorten op de Nederlandse cybersecurity-arbeidsmarkt?
  <br/><sub>— [Onderzoek: Onderwijs en Arbeidsmarkt Cybersecurity - Pl...](https://www.ptvt.nl/publicatie/onderzoek-onderwijs-en-arbeidsmarkt-cybersecurity)</sub>
- Wat zijn de loopbaan- en arbeidsmarktperspectieven van mbo-gediplomeerden in de ICT?
  <br/><sub>— [Rapportages | Loopbaan en arbeidsmarktperspectief van mbo-gediplomeerde…](https://onderzoek010.nl/document/Loopbaan-en-arbeidsmarktperspectief-van-mbo-gediplomeerden-in-de-ICT,-Risbo)</sub>
- Welke beleidsopties zijn kansrijk om de arbeidsmarktuitkomsten van personen met een migratieachtergrond te verbeteren?
  <br/><sub>— [Kansrijk integratiebeleid op de arbeidsmarkt](https://www.cpb.nl/system/files/cpbmedia/omnidownload/Kansrijk_integratiebeleid_op_de_arbeidsmarkt2.pdf)</sub>
- Hoe beïnvloeden wijzigingen in het belastingstelsel de belastingdruk van verschillende inkomensgroepen?
  <br/><sub>— [Effecten van belastinghervormingen op de belastingdruk van verschillend…](https://www.cpb.nl/effecten-van-belastinghervormingen-op-de-belastingdruk-van-verschillende-inkomensgroepen)</sub>

### Health and care

- Hoe zichtbaar is de vergrijzing in de uitgaven aan gezondheidszorg?
  <br/><sub>— [Effecten van vergrijzing steeds beter zichtbaar in uitgaven gezondheids…](https://www.rivm.nl/nieuws/effecten-van-vergrijzing-steeds-beter-zichtbaar-in-uitgaven-gezondheidszorg)</sub>
- Wat zijn de regionale verschillen in bekkenbodemzorg en zijn deze wenselijk of ongewenst?
  <br/><sub>— [SKMS NVOG project ‘inventarisatie zorgpaden rondom bekkenbodemzorg’ – N…](https://www.nvog.nl/skms-nvog-project-inventarisatie-zorgpaden-rondom-bekkenbodemzorg/)</sub>
- Welke knelpunten kandidaten, studenten en afgestudeerden op verschillende momenten tegengekomen?
  <br/><sub>— [Verkenning problematiek zorgopleidingen hbo/wo Caribische studenten | K…](https://kohnstamminstituut.nl/rapport/verkenning-problematiek-zorgopleidingen-hbo-wo-caribische-studenten/)</sub>
- Hoe beïnvloedt beperkte financiële middelen het gebruik van zorg?
  <br/><sub>— [Weinig te besteden, minder zorg? Zorguitgaven van mensen met beperkte f…](https://www.cpb.nl/publicatie/weinig-te-besteden-minder-zorg-zorguitgaven-van-mensen-met-beperkte-financiele-middelen)</sub>
- Hoe ontwikkelde het gebruik van Wmo-ouderenzorg zich in de loop der jaren?
  <br/><sub>— [Gemeentelijke trends in het gebruik van Wmo-ouderenzorg | CPB Website](https://www.cpb.nl/gemeentelijke-trends-in-het-gebruik-van-wmo-ouderenzorg)</sub>

### Education and youth

- Wat zijn de verwachte groeiende vraag naar cybersecuritypersoneel?
  <br/><sub>— [Onderzoek: Onderwijs en Arbeidsmarkt Cybersecurity - Pl...](https://www.ptvt.nl/publicatie/onderzoek-onderwijs-en-arbeidsmarkt-cybersecurity)</sub>
- In welke mate is het landelijke beeld van toename van jeugdhulpgebruik ook van toepassing in Enschede?
  <br/><sub>— [Verdiepende analyses in- en uitstroom naar Jeugdhulp - Kennispunt Twente](https://kennispunttwente.nl/publicaties/sociaal-domein/verdiepende-analyses-in-en-uitstroom-naar-jeugdhulp)</sub>
- In hoeverre zijn de gegevens over voorschoolse educatie die scholen aan DUO leveren representatief?
  <br/><sub>— [Kwaliteit van data over voorschoolse educatie in BRON | Kohnstamm Insti…](https://kohnstamminstituut.nl/rapport/kwaliteit-van-data-over-voorschoolse-educatie-in-bron/)</sub>
- Wat zijn de obstakels die Caribische studenten ondervinden bij toelating en tijdens de opleiding?
  <br/><sub>— [Verkenning problematiek zorgopleidingen hbo/wo Caribische studenten | K…](https://kohnstamminstituut.nl/rapport/verkenning-problematiek-zorgopleidingen-hbo-wo-caribische-studenten/)</sub>
- Wat is het loonverschil tussen leraren in het voortgezet onderwijs en vergelijkbare werknemers in de marktsector?
  <br/><sub>— [Wat een leraar in het voortgezet onderwijs verdient - SEO Economisch On…](https://www.seo.nl/publicaties/wat-een-leraar-in-het-voortgezet-onderwijs-verdient/)</sub>

### Housing, regions and liveability

- Welke interventies worden in het kader van het programma ‘Een nieuw bestaan, een nieuwe baan’ uitgevoerd?
  <br/><sub>— [Eindrapport - Programma Een nieuw bestaan, een nieuwe baan](https://www.regioplan.nl/wp-content/uploads/2022/07/18128-Eindrapport-EenNieuwBestaanEenNieuweBaan-Regioplan-7juli22.pdf)</sub>
- Wat is de invloed van de JOGG-aanpak op het overgewicht en beweeggedrag bij kinderen en jongeren tussen de 2 en 19 jaar?
  <br/><sub>— [Daling overgewicht in JOGG-buurten | RIVM](https://www.rivm.nl/nieuws/daling-overgewicht-in-jogg-buurten)</sub>
- Welke gebieden in Nederland hebben een structurele druk op hun leefbaarheid en zijn dit er meer of minder dan bij de vorige meting?
  <br/><sub>— [leefbaarometer.nl](https://www.leefbaarometer.nl/resources/Analyserapport-Leefbaarheid-onder-de-loep-2022.pdf)</sub>
- Hoe moet de subsidies evenrediger over het land worden verdeeld?
  <br/><sub>— [Boekman #139: Cultuur in de regio - Boekmanstichting](https://www.boekman.nl/tijdschrift/boekman-139-cultuur-in-de-regio/)</sub>
- Waarom verhuizen steeds meer Amsterdammers naar de regio?
  <br/><sub>— [In de stad blijven of verhuizen naar de regio | Website Onderzoek en St…](https://onderzoek.amsterdam.nl/publicatie/in-de-stad-blijven-of-verhuizen-naar-de-regio-proefschrift)</sub>

### Migration, population and diversity

- Welke aandoeningen leiden tot de hoogste zorguitgaven?
  <br/><sub>— [Effecten van vergrijzing steeds beter zichtbaar in uitgaven gezondheids…](https://www.rivm.nl/nieuws/effecten-van-vergrijzing-steeds-beter-zichtbaar-in-uitgaven-gezondheidszorg)</sub>
- Wat zijn de oorzaken van de grote en hardnekkige verschillen in de arbeidsparticipatie tussen personen met en zonder migratieachtergrond?
  <br/><sub>— [Kansrijk integratiebeleid op de arbeidsmarkt](https://www.cpb.nl/system/files/cpbmedia/omnidownload/Kansrijk_integratiebeleid_op_de_arbeidsmarkt2.pdf)</sub>
- Wat zijn de demografische ontwikkelingen in de wereld, Europa en Nederland?
  <br/><sub>— [Bevolkingsvraagstukken in Nederland anno 2012](https://publ.nidi.nl/output/books/nidi-book-86.pdf)</sub>
- Wat zijn de belangrijkste overkoepelende hoofdconclusies en inzichten uit deze derde meting?
  <br/><sub>— [Monitor van gelijkwaardige kansen en evenredige posities op de arbeidsm…](https://open.overheid.nl/documenten/8dda65b3-1cda-43d3-bf4f-211d3dc9336b/file)</sub>
- Wat is de ontwikkeling van het aantal werkende arbeidsmigranten in de provincie Limburg?
  <br/><sub>— [Onderzoek internationale arbeidsmigranten provincie Limburg](https://www.limburg.nl/publish/pages/9299/eindrapportage_decisio_onderzoek_arbeidsmigranten_limburg_2024_03.pdf)</sub>

### Business, economy and innovation

- Wat is de huidige en toekomstige arbeidsmarktpositie van afgestudeerden van bacheloropleidingen in de Economie en de Bedrijfskunde?
  <br/><sub>— [Kabinet onderschat arbeidsmarktpositie studenten Economie en Bedrijfsku…](https://esb.nu/kabinet-onderschat-arbeidsmarktpositie-studenten-economie-en-bedrijfskunde/)</sub>
- Hoe zit het met de positie van afgestudeerden na economische vervolgopleidingen?
  <br/><sub>— [Arbeidsmarktperspectieven economieopleidingen op het mbo, hbo en wo - E…](https://esb.nu/arbeidsmarktperspectieven-van-economieopleidingen-op-het-mbo-hbo-en-wo/)</sub>
- Voeren ondernemingen onder Nederlandse zeggenschap meer onderzoek en ontwikkeling uit in Nederland dan ondernemingen onder buitenlandse zeggenschap?
  <br/><sub>— [Ondernemingen onder Nederlandse zeggenschap investeren het meest in ond…](https://www.mejudice.nl/artikelen/detail/ondernemingen-onder-nederlandse-zeggenschap-investeren-het-meest-in-onderzoek-en-ontwikkeling-in-nederland)</sub>
- Waar is de geboden steun precies terechtgekomen?
  <br/><sub>— [Ook ondernemingen met omzetgroei ontvingen steun - ESB](https://esb.nu/ook-ondernemingen-met-omzetgroei-ontvingen-steun/)</sub>
- Hoeveel tijd heeft een ambulante handelaar nodig om zijn investeringen terug te verdienen met een redelijk rendement?
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
