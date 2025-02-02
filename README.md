# Synthetic_City_Generator

run pip install -r requirements.txt

Data preprocessing. 

data_preprocessing/src/
- initial_EDA and quantize_and_inverse_EDA.ipynb are used for initial data analysis.
- countries.csv contains all ISO countries
- collect_cities.py to collect all cities around the world for each country
- get_coordinates_data.py and get_adj_matrices_data.py gets the coordinate data and the adjacency data and transforms it to fit for the project needs.
- split_datasets.py splits the whole datset into test train and validation sets with 70-15-15 split.
- the dataset for cities all over the world is too large to be uploaded to the repository so it has to be downloaded by running the respective slides. this data will be used to train the regression models and Variational Graph Auto Encoder (VGAE).

Implementation.

regression/src/
- regression_single_coord_prediction.py contains code to test each of the 4 regression models individually for the task of predicting the next single coordinate and show the results along with visualisations.
- regression_multiple_coords_prediction.py contains code to test all 4 regression models simultaneously for the tasking of predicting multiple coordinates at once and show the results along with visualisations..

transformer/llama
- dataset folder contains already contains the dataset to train the llama model in json format.

transformer/llama/src
- llama_prompt.txt contains the prompt used to create the dataset to train the llama model
- fine_tune_data_prep_llama.py creates the datset in json format to tain the llama model
- fine_tune_llama.py trains the llama model and generates output for a sample test set city
- generate_coordinates_plot.py shows the comparision plot of original and the predicted coordinates

transformer/gpt4o-mini
- dataset folder contains already contains the dataset to train the gpt4o-mini model in json format. it also contains the predictions obtained

transformer/gpt4o-mini/src
- this folder does not contain the code to train the model because the finetuning was done on open ai playground.
- finetune_data_prep.py prepares the dataset in json format to train the model
- get_openai_model_predictions.py gets the predictions of the model and saves it in a json
- get_predicted_coordinates.py merges together the input and output coordinates that are formatted seperately in the output json to form the full sequence of cooridnates of the city to be plotted.


