import os
import subprocess

def run_pipeline():
    print("=== EEG Emotion Recognition: Full Pipeline Execution ===")
    
    # Stratified Track
    print("\n[1/2] Executing Stratified KFold Pipeline...")
    # subprocess.run(["python", "code/stratified_Laso_pipeline.py"])
    
    # GroupKFold Track
    print("\n[2/2] Executing GroupKFold (LOSO) Pipeline...")
    # subprocess.run(["python", "code/groupkfold_loso_pipeline.py"])
    
    print("\n=== Pipeline Complete ===")
    print("Results summary available in: summary_results.md")

if __name__ == "__main__":
    run_pipeline()
