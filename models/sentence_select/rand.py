#!/usr/bin/env python3
import torch
import logging
import random
import pandas as pd
import argparse
import models.sentence_select.model_zoo as model_zoo

logger = logging.getLogger(__name__)

nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

def get_segments(segments, max_tokens=50):
    filtered_segments = []
    for segment in segments:
        doc = nlp(segment)
        if len(doc) <= max_tokens:
            filtered_segments.append(segment)
    return filtered_segments
    
def random_baseline_selection(segments, num_sentences):
    return random.sample(segments, min(num_sentences, len(segments)))

def save_results_to_csv(businesses, preds, y_labels, segments, output_path):
    # saves the results to a CSV file with the specified columns.
    df = pd.DataFrame({
        'Business ID': businesses,
        'Pred Score': preds,
        'True Y Label': y_labels,
        'Selected Sentences': segments
    })
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_used', type=str, help='feature used',
                        default="notes", choices=["all", "notes", "all_but_notes"])
    parser.add_argument("--debug", action="store_true", default=False,
                        help="whether to enter debug mode")
    parser.add_argument('--segment', type=str, help='segmentation type')
    parser.add_argument('--data', type=str, help='dataset', default="yelp")
    parser.add_argument('--data_split', type=str, help='data split naming', default="test")
    parser.add_argument('--target_type', type=str, help='target type. regression or classification', default="reg", choices=["reg", "cls"])
    parser.add_argument('--score_function', type=str, help='score function for choosing return segments', default="best")
    parser.add_argument('--return_candidates', type=int, help='', default=1)
    parser.add_argument('--model', type=str, help='task',
                        choices=["Transformer"])
    parser.add_argument('--device', type=str, help='task',
                        default="0")
    parser.add_argument('--num_review', type=int, help='task',
                        default=50)
    parser.add_argument('--num_sentences', type=str, 
                        help='number of sentences selected in the summary',
                        default="6")
    parser.add_argument('--trunc_length', type=int,
                        help="hyperparameter for language fluency",
                        default=None)
    parser.add_argument('--data_dir', type=str, help='task',
                        default="/data/joe/Information-Solicitation-Yelp/")
    parser.add_argument('--result_dir', type=str, help='result directory',
                        default="/data/joe/Information-Solicitation-Yelp/models/sentence_select/")
    parser.add_argument('--trained_model_path', type=str, help='task', required=True)
    args = parser.parse_args()

    num_sent = args.num_sentences
    
    if isinstance(num_sent, str) and "trunc" in num_sent:
        # Extract and convert the integer part if it's a truncation string
        args.trunc_length = int(num_sent[:-5])
        args.num_sentences = args.trunc_length
    else:
        # Ensure it's converted to an integer if it's not already
        args.num_sentences = int(num_sent)

    logger.info("Running Random Sentence Selection Baseline...")

    if args.model == "Transformer":
        model = model_zoo.Transformer(args)

    output_path = f"{args.data_dir}/result.csv"
        
    segments, businesses, avg_scores = model._get_data()

    all_business_ids = []
    all_pred_scores = []
    all_y_labels = []
    all_selected_sentences = []
    unique_businesses = list(set(businesses))

    for business in unique_businesses:
    # Filter segments for the current business
        business_segments = [seg for seg, biz in zip(segments, businesses) if biz == business]
        business_y_labels = [label for label, biz in zip(avg_scores, businesses) if biz == business]
    
        # Ensure there are enough segments to sample
        if len(business_segments) < args.num_sentences:
            selected_segments = business_segments  # Use all if fewer than required
        else:
            selected_segments = random.sample(business_segments, args.num_sentences)
    
        # Run model prediction on the selected segments
        segment_preds = model.run_model(selected_segments)
    
        # Collect results for the current business
        all_business_ids.extend([business] * len(selected_segments))
        all_pred_scores.extend(segment_preds)
        all_y_labels.extend(business_y_labels[:len(selected_segments)])
        all_selected_sentences.extend(selected_segments)

    df = pd.DataFrame({
    'Business ID': all_business_ids,
    'Pred Score': all_pred_scores,
    'True Y Label': all_y_labels,
    'Selected Sentences': all_selected_sentences
    })
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

