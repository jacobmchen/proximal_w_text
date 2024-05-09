import numpy as np 
import pandas as pd 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch 
import argparse
import json 

#MODEL_NAME = "google/flan-t5-small"
MODEL_NAME = "google/flan-t5-xxl"

def apply_prompt(condition, text):
    return 'Context: ' + text + '\nIs it likely the patient has '+ condition +'?\nConstraint: Even if you are uncertain, you must pick either “Yes” or “No” without using any other words.'

def map_predictions(output): 
    if 'yes' in output.lower(): return 1
    else: return 0

def go(text_split, gpu_no):
    df = pd.read_csv(f"text_csv_files/text_data_{text_split}.csv")

    if gpu_no == "-1":
        print("Using CPU.")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print("GPU is available!")
        # device = torch.device(f"cuda:{gpu_no}")
        device = torch.device("cuda")
    else:
        print("GPU not available, using CPU instead.")
        device = torch.device("cpu")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    fname = f'flan_jsonl_files/flan--{text_split}.jsonl'
    w =  open(fname, 'w')

    for i, row in df.iterrows(): 
        text = row[text_split]
        patient_preds = {'HADM_ID': row['HADM_ID']}

        try: 
            for j, U_input in enumerate(['acute respiratory failure', 'atrial fibrillation', 'congestive heart failure',
                                            'coronary atherosclerosis of native coronary artery', 'hypertension']):  
                inputs = tokenizer(apply_prompt(U_input, text), return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=5)
                result = tokenizer.batch_decode(outputs)[0]
                pred = map_predictions(result)
                patient_preds[U_input] = pred
            json.dump(patient_preds, w)
            w.write('\n')
        except: 
            print("Bad output, HADM_ID=", row['HADM_ID'])

        if i % 1000 == 0 : print(i)
    
    print('saved to:', fname)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("text_split", help="key for the data file", type=str)#choose from: ECG, Echo, Nursing, Radiology, Toy
    parser.add_argument("gpu_no", help="key for the data file", type=str)#choose from: ECG, Echo, Nursing, Radiology, Toy
    args = parser.parse_args()
    print(args.text_split)
    go(args.text_split, args.gpu_no)

    #Usage 
    # python flan_infer_proxy.py ECG 0 
    # python flan_infer_proxy.py Echo 1
    # python flan_infer_proxy.py Nursing 2



