# Weld Quality Prediction


## Install requierements.txt
Use `uv` or `pip` to install the dependencies.
```bash
uv pip install -r requierements.txt
```

## Import dataset from huggingface
### The initial dataset (not clean, just formatted with column names)

```python
import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/MoSBAIHI/weld-quality-dataset/data/train-00000-of-00001.parquet")
```

### A cleaner dataset, handled categoric values and null values are replace with None
```python
import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/wstoloah/weld-quality-v1/data/train-00000-of-00001.parquet")
```

