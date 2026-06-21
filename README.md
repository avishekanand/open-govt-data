# OGD Data Analysis 🏛️📊

**Open Governmental Data Analysis** - A comprehensive toolkit for fetching, processing, and analyzing open government datasets from various statistical agencies.

## 🎯 Overview

This project provides tools to automatically fetch, enrich, and analyze open governmental datasets, with a focus on Eurostat data. It combines data collection, AI-powered metadata enrichment, and batch processing capabilities.

## 🚀 Features

- **📥 Automated Data Fetching**: Batch download datasets from Eurostat and other statistical agencies
- **📊 Structured Processing**: Convert raw statistical data into analysis-ready formats
- **🔍 Metadata Generation**: Create comprehensive dataset catalogs with searchable metadata

## 📁 Project Structure

```
open-govt-data/
├── eurostat_fetch_one.py           # main downloader + flattener
├── batch_fetch_eurostat.py         # batch processing tool
├── csv_to_ollama_jsonl_complete_only.py # AI enrichment pipeline
├── cbs_tiny_agent.py               # CBS Netherlands data agent
├── utils/
│   ├── __init__.py                 # Python package init
│   └── jsonl_to_csv.py             # JSONL to CSV converter
├── data/
│   ├── eurostat_base.csv           # original dataset catalog
│   ├── eurostat_gemma3.jsonl       # AI-enriched metadata
│   └── eurostat_gemma3_gpt5.jsonl  # refined metadata catalog
├── downloads/                      # generated CSV outputs (gitignored)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT license
└── README.md                       # this file
```

## ⚙️ Installation

### Requirements

- Python ≥ 3.9
- Packages:

```bash
pip install pandas requests tabulate numpy
```

### Clone and run

```bash
git clone https://github.com/avishekanand/open-govt-data.git
cd open-govt-data
pip install -r requirements.txt
```

### Optional: Set up Ollama (for AI enrichment)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma3:latest
```

## 🎮 Usage

### Fetch Individual Datasets

```bash
# Download a single Eurostat dataset
python eurostat_fetch_one.py --code TPS00001 --out population.csv

# With filters
python eurostat_fetch_one.py --code TRNG_LFS_22 --filter geo=NL time=2023 --out training.csv
```

### Batch Download Multiple Datasets

```bash
# Download first 10 datasets for testing
python batch_fetch_eurostat.py --input data/eurostat_gemma3_gpt5.jsonl --output-dir downloads --max-datasets 10

# Full batch download with progress tracking
python batch_fetch_eurostat.py --input data/eurostat_gemma3_gpt5.jsonl --output-dir downloads --delay 2.0

# Resume from a specific point
python batch_fetch_eurostat.py --input data/eurostat_gemma3_gpt5.jsonl --output-dir downloads --start-from 100 --skip-existing
```

### AI-Powered Metadata Enrichment

```bash
# Enrich dataset metadata with AI-generated descriptions and queries
python csv_to_ollama_jsonl_complete_only.py \
    --input data/eurostat_enriched.csv \
    --output eurostat_enriched_ai.jsonl \
    --model gemma3:latest \
    --prompt-mode json \
    --temperature 0.2
```

### Convert JSONL to CSV for Excel

```bash
# Convert JSONL metadata to Excel-friendly CSV
python utils/jsonl_to_csv.py data/eurostat_gemma3_gpt5.jsonl -o eurostat_data.csv

# Convert any JSONL file
python utils/jsonl_to_csv.py input.jsonl --output output.csv
```

## 🇳🇱 CBS StatLine — metadata search engine

A "Dutch Public Data Intelligence Engine" over public CBS aggregate data (no
confidential microdata). Lives in the `cbs/` package and uses the CBS OData v4
API (`https://datasets.cbs.nl/odata/v1/CBS/{TABLE_ID}`).

```bash
# 1. Fetch the full table catalogue (4,857 tables -> Parquet)
python -m cbs.catalog

# 2. Ingest the semantic metadata for the active tables (Properties/Dimensions/Measures)
python -m cbs.batch_ingest_statline --limit 700 --sample-data 5

# 3. (optional) doc2query-enrich tables into English with a local LLM (gemma4 via Ollama)
python -m cbs.enrich_cbs --limit 8 --model gemma4:latest

# 4. Build the SQLite FTS5 term-matching index over all metadata text fields
python -m cbs.build_search_index --selftest

# 5. Launch the search web app  ->  http://localhost:8501
streamlit run cbs/search_app.py
```

Single-table deep ingest (with full code lists + sample observations):

```bash
python -m cbs.ingest_statline --table 83765NED --regions GM0363 GM0503
```

The search indexes Dutch titles/descriptions, gemma4 English enrichment,
dimensions and measures — so both `inkomen` and `income` find the same tables.

### Running on a server

The whole pipeline is wrapped in idempotent, resumable scripts under `scripts/`:

```bash
# End-to-end (catalogue -> metadata -> enrich -> index), configurable via env vars
LIMIT=700 MODEL=gemma4:latest ./scripts/run_pipeline.sh

# Point enrichment at a remote/cluster Ollama
OLLAMA_HOST=http://gpu-node:11434 ./scripts/run_pipeline.sh

# Skip the slow LLM step (metadata + index only)
SKIP_ENRICH=1 ./scripts/run_pipeline.sh

# Serve the UI headless on a server, then SSH port-forward to view it
PORT=8080 ./scripts/serve_app.sh
```

On **TU Delft DAIC** (SLURM + GPU), submit the enrichment as a batch job — it starts
Ollama on the allocated GPU node and caches the model on the umbrella share:

```bash
sbatch scripts/enrich_daic.slurm           # edit partition / OLLAMA_MODELS path first
```

## 📊 Dataset Sources

- **🇪🇺 Eurostat**: European Union statistical data
- **🇳🇱 CBS Netherlands**: Dutch national statistics
- **🌍 More sources**: Extensible framework for additional agencies

## 🔧 Key Components

### `eurostat_fetch_one.py`
- Fetches individual Eurostat datasets via SDMX API
- Handles complex dimension structures and missing data
- Provides detailed summaries and data previews
- Robust error handling and retry logic

### `batch_fetch_eurostat.py`
- Processes hundreds of datasets automatically
- Progress tracking with CSV logs
- Configurable delays and timeouts
- Resume capability for interrupted runs

### `csv_to_ollama_jsonl_complete_only.py`
- AI-powered metadata enrichment
- Generates dataset descriptions and example queries
- Multiple prompt modes (JSON, loose text)
- Caching and batch processing support

## 📈 Example Output

### Dataset Summary
```
Dataset: TPS00001 - Population on 1 January
→ Time coverage: 2014 … 2025 (total 12 years)
→ Dimensions: freq, indic_de, geo, time
→ Total observations: 580
→ File size: 13.0 KB
```

### AI-Generated Metadata
```json
{
  "code": "TPS00001",
  "title": "Population on 1 January",
  "enriched_description": "Annual population counts for EU countries...",
  "example_queries": [
    "How has population changed across EU countries from 2014-2025?",
    "Which countries show the fastest population growth?",
    "What are the population trends in Nordic countries?"
  ],
  "potential_applications": [
    "Demographic planning and forecasting",
    "Resource allocation for public services",
    "Economic analysis and policy development"
  ]
}
```

5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Eurostat** for providing comprehensive open data APIs
- **CBS Netherlands** for accessible statistical data
- **Ollama** for local LLM capabilities
- **OpenAI** for AI-powered enrichment

---

*Making governmental data accessible, analyzable, and actionable* 🚀
