# How to use the loader module

The loader module offers functions to load and split datasets for semi-supervised learning tasks.

First, make sure you are logged into the Hugging Face Hub.
if you haven't done so, go to your Hugging Face account => Settings => Access Tokens and create a new token and copy it to your clipboard.

### Method 1: Using .env file (recommended)
Then, create a file named `.env` in the root directory of the project and add the following line:
```
HF_API_KEY=your_huggingface_token_here  
```

In your notebook or script, you can load your token using the `python-dotenv` package:
```python
from dotenv import load_dotenv
import os

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
```

### Method 2: Using `notebook_login()`
Run the following code in a notebook cell, a login widget will apear where you can paste your token:
```python
from huggingface_hub import notebook_login
notebook_login()
```

### Method 3: Using Hugging Face CLI
Alternatively, you can log in using the Hugging Face CLI by running the following command in your terminal and following the instructions:
```bash
huggingface-cli login
```

You can now use the loader module as follows:

If you have access to your Hugging Face API key in your .env file, use `load_data_with_token()` function to load the dataset splits. Otherwise, you can use the `load_data()`function as your key is already stored in your Hugging Face CLI.

First, make sure you are in the root directory of the project.


Then, import the necessary functions and constants:
```python
from src.semi_supervised.loader import load_data, load_data_with_token, split_train_val_test, TARGET_COLS

# To remember the target column names, print TARGET_COLS
print(TARGET_COLS)
```

Then load the labeled and unlabeled splits of the dataset:
```python
labeled_data = load_data(
    target='yield_strength_MPa', # target column name
    split='labeled', # 'labeled' or 'unlabeled'
    HF_API_KEY=HF_API_KEY # your Hugging Face API key
)

unlabeled_data = load_data(
    target='yield_strength_MPa', # target column name
    split='unlabeled', # 'labeled' or 'unlabeled'
    HF_API_KEY=HF_API_KEY # your Hugging Face API key
)
```

Note that you can replace `'yield_strength_MPa'` with any other target column name from `TARGET_COLS`.

Also note that the loaded datasets will have only the specified target column, and all other target columns will be removed.

The loaded datasets are returned as pandas DataFrames and are imputed for missing values using appropriate strategies inside the `load_data` function.

Finally, you can split the labeled and unlabeled data into training, validation, and test sets:
```python
(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(
    labeled_data,
    unlabeled_data,
    target='yield_strength_MPa', # target column name
    val_size=0.2, # proportion of validation set
    test_size=0.1, # proportion of test set
    random_state=42 # do not modify
)
```

This will return the training, validation, and test sets as tuples of features (X) and target (y).

The training set will include both labeled and unlabeled data, while the validation and test sets will only include labeled data.

