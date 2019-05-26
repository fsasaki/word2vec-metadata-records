# Word2vec and Metadata Records

Repository for code related to word2vec and metadata records processing.

# Usage

## Step 1: get metadata records

The output of this step is stored in the file linked-data-query-results/result-with-delimiter.csv .

As of writing, the SPARL endpoint from the Econstor project has been used. The data from that endpoint can be loaded via the script sparql_query.py

## Step 2: train models

Run training_model.py . This script calls helper functions from another script: preprocess_sparql_output.py. 

training_model.py calls the parameters for models defined in training-settings.csv.

## Step 3: analyse all models

Run analyse_models.py . This will generate a file model-analysis.csv , that contains some statistics about all models in the (sub)directory.

# Step 4: visualize a term

Run visualize.py. The result will be stored in close-words.png