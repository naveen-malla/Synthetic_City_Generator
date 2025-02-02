# Synthetic_City_Generator

run pip install -r requirements.txt

Data preprocessing. 

src/
- initial_EDA and quantize_and_inverse_EDA.ipynb are used for initial data analysis.
- countries.csv contains all ISO countries
- collect_cities.py to collect all cities around the world for each country
- get_coordinates_data.py and get_adj_matrices_data.py gets the coordinate data and the adjacency data and transforms it to fit for the project needs.
- split_datasets.py splits the whole datset into test train and validation sets with 70-15-15 split.

Implementation:

Experiements

openai gpt4o mini

lstm 

(change the testing file)

vgae with cooridnates only