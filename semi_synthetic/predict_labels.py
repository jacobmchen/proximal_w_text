import pandas as pd
import numpy as np
import pickle
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

# function for applying the prompt constraint
def apply_prompt(condition, text):
    return 'Context: ' + text + '\nIs it likely the patient has '+ condition +'?\nConstraint: Even if you are uncertain, you must pick either “Yes” or “No” without using any other words.'

def check_token(token):
    if token != 'Yes' and token != 'No':
        print('generated weird token:', token)

def document_level_classifier(condition, text, tokenizer, model):

    input_ids = tokenizer(apply_prompt(condition, text), return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids, max_new_tokens=99999)

    prediction = tokenizer.decode(outputs[0])[6:-4]

    check_token(prediction)

    if prediction == 'Yes':
        return 1
    else:
        return 0

def sentence_level_classifier(condition, sentences, tokenizer, model):
    # still need to implement a function for splitting sentences

    for sentence in sentences:
        input_ids = tokenizer(apply_prompt(condition, sentence), return_tensors="pt").input_ids.to("cuda")

        outputs = model.generate(input_ids, max_new_tokens=99999)

        prediction = tokenizer.decode(outputs[0])[6:-4]

        check_token(prediction)
        if prediction == 'Yes':
            return 1

    return 0

def predict_labels(condition, method='document', text_partition=1, testing=False):
    # condition is what you would like to insert into the prompt
    # type is whether you want to be using a document level classifier or a sentence level classifier
    # if testing is set to True, then the program will terminate after processing the first 5 rows of data

    data = pd.read_csv('csv_files/master_data.csv')

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")

    # apply the constraint and also ask the model to evaluate yes or no

    # this list stores the predictions
    predictions = []

    all_sentences = pickle.load(open('list_of_sentences.p', 'rb'))

    for index, row in data.iterrows():
        if testing:
            if index == 5:
                break

        #############################
        # the following code is for document level classification
        #############################

        if method == 'document':
            predictions.append(document_level_classifier(condition, row['notes_half'+str(text_partition)], tokenizer, model))

        #############################
        # the following code is for sentence level classification
        #############################

        if method == 'sentence':
            predictions.append(sentence_level_classifier(condition, all_sentences[index], tokenizer, model))

    prediction_data = pd.DataFrame({'prediction': predictions})
    prediction_data.to_csv('csv_files/predictions-xxl-'+ condition.replace(' ', '') + '-' + method + '-texthalf' + str(text_partition) + '.csv', index=False)
