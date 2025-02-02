
Run `pip install -r requirements.txt` to install the required dependencies.

## Data Preprocessing 

### Location: data_preprocessing/src/
- `initial_EDA` and `quantize_and_inverse_EDA.ipynb` are used for initial data analysis.
- `countries.csv` contains all ISO countries.
- `collect_cities.py` collects all cities worldwide for each country.
- `get_coordinates_data.py` and `get_adj_matrices_data.py` retrieve coordinate data and adjacency data, transforming them to fit the project requirements.
- `split_datasets.py` splits the entire dataset into train, test, and validation sets with a 70-15-15 split.
- Note: The global cities dataset is too large for repository upload and must be downloaded by running the respective scripts. This data will be used to train regression models and Variational Graph Auto Encoder (VGAE).

## Implementation

### Regression Models (regression/src/)
- `regression_single_coord_prediction.py` tests four regression models individually for predicting the next single coordinate, including visualizations.
- `regression_multiple_coords_prediction.py` tests all four regression models simultaneously for predicting multiple coordinates, including visualizations.

### Transformer Models

#### LLaMA (transformer/llama)
- The dataset folder contains prepared data in JSON format for training the LLaMA model.

##### transformer/llama/src
- `llama_prompt.txt` contains the prompt used for dataset creation.
- `fine_tune_data_prep_llama.py` creates the JSON format dataset.
- `fine_tune_llama.py` trains the model and generates output for sample test cities.
- `generate_coordinates_plot.py` displays comparison plots of original and predicted coordinates.

#### GPT4-Mini (transformer/gpt4o-mini)
- The dataset folder contains prepared data in JSON format and model predictions.

##### transformer/gpt4o-mini/src
- Note: Training code is not included as fine-tuning was performed on OpenAI playground.
- `finetune_data_prep.py` prepares the dataset in JSON format.
- `get_openai_model_predictions.py` generates and saves model predictions.
- `get_predicted_coordinates.py` merges input and output coordinates from the JSON output.

### VGAE
- The model_checkpoints folder contains saved best models for each version, enabling direct inference without retraining.

#### vgae/src
- `VGAE.py` is the main training and testing code for specified node ranges (10-50 nodes and 100-500 nodes).
- `vgae_experiment_*.py` files contain modifications addressing the overconnectivity issue in the main model.
- `plot_metrics_progression.py` visualizes metric progression during training.
- `plot_results.py` visualizes results from the test set.