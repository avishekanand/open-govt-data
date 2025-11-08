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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Eurostat** for providing comprehensive open data APIs
- **CBS Netherlands** for accessible statistical data
- **Ollama** for local LLM capabilities
- **OpenAI** for AI-powered enrichment

## 📞 Contact

- **Project**: [OGD Data Analysis](https://github.com/avishekanand/ogd-data-analysis)
- **Issues**: [GitHub Issues](https://github.com/avishekanand/ogd-data-analysis/issues)

---

*Making governmental data accessible, analyzable, and actionable* 🚀
