#!/usr/bin/env python

import argparse
import pandas as pd

def compute_pos_weights(df, label_columns, treat_uncertain_as_positive=True):
    """
    Computes positive weights for each label based on the ratio of negative to positive samples.
    If treat_uncertain_as_positive is True, uncertain labels (-1) are replaced with 1.
    
    pos_weight = (# negatives) / (# positives)
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset.
        label_columns (list of str): List of label column names.
        treat_uncertain_as_positive (bool): If True, replace -1 with 1 before counting.
    
    Returns:
        dict: Mapping from label names to their computed pos_weight.
    """
    pos_weights = {}
    
    for label in label_columns:
        # Copy the dataframe to avoid modifying the original
        valid = df.copy()
        if treat_uncertain_as_positive:
            # Replace uncertain (-1) values with 1
            valid[label] = valid[label].replace(-1, 1)
        
        pos_count = (valid[label] == 1).sum()
        neg_count = (valid[label] == 0).sum()
        
        if pos_count > 0:
            pos_weights[label] = neg_count / pos_count
        else:
            pos_weights[label] = 1.0  # Default if no positives
        
    return pos_weights

def main():
    parser = argparse.ArgumentParser(
        description="Calculate pos_weight for CheXpert labels, treating uncertain (-1) as positive."
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to your CheXpert CSV file.")
    args = parser.parse_args()
    
    # Define the label columns (first 5 columns are metadata and excluded)
    label_columns = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Pneumothorax",
        "Support Devices"
    ]
    
    # Load the CSV
    df = pd.read_csv(args.csv)
    
    # Compute pos_weights, treating uncertain as positive
    pos_weights = compute_pos_weights(df, label_columns, treat_uncertain_as_positive=True)
    
    # Print the dictionary of pos_weights
    print("pos_weights dict:")
    for label, weight in pos_weights.items():
        print(f"  {label}: {weight:.3f}")
    
    # Print the weights as an array (list) in the order of label_columns
    weights_array = [round(pos_weights[label], 3) for label in label_columns]
    print("\npos_weights array:")
    print(weights_array)

if __name__ == "__main__":
    main()
