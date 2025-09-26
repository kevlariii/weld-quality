# Weld Quality Prediction

## Import dataset from huggingface

```python
import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/MoSBAIHI/weld-quality-dataset/data/train-00000-of-00001.parquet")
```

