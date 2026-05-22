FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from tests.benchmark_data import generate_example_pair; from pathlib import Path; generate_example_pair(Path('data/example_pdb'))"

EXPOSE 8501

CMD ["python", "main.py", "--web", "--port", "8501"]
