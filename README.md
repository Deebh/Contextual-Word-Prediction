# Installation Instructions

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install poetry
```

### Mac specific
```bash
poetry remove tensorflow[and-cuda]
poetry add tensorflow-metal
```

```bash
poetry install --no-root
```