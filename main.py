import sys
from model import NgramLanguageModel
from utils import load_data

def main():
    """
    Main function to run the homework tasks.
    """
    try:
        # Load all datasets
        raw_train_data = load_data('ptbdataset\\ptb.train.txt')
        raw_valid_data = load_data('ptbdataset\\ptb.valid.txt')
        raw_test_data = load_data('ptbdataset\\ptb.test.txt')
        
        print(f"Loaded {len(raw_train_data)} training sentences.")
        print(f"Loaded {len(raw_valid_data)} validation sentences.")
        print(f"Loaded {len(raw_test_data)} test sentences.\n")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure ptb.train.txt, ptb.valid.txt, and ptb.test.txt are in the same directory.")
        return

    # --- Task 3.1: MLE Models ---
    
    # Build vocabulary from training data once.
    # We set n=4 here just to ensure padding is sufficient for all models,
    # but the vocab itself is independent of n.
    base_model = NgramLanguageModel(4, raw_train_data, smoothing='mle')
    main_vocab = base_model.vocab
    
    # Preprocess test data once for each n-gram size
    test_data_n1 = [base_model.preprocess_sentence(s) for s in raw_test_data]
    test_data_n2 = [base_model.preprocess_sentence(s) for s in raw_test_data]
    test_data_n3 = [base_model.preprocess_sentence(s) for s in raw_test_data]
    test_data_n4 = [base_model.preprocess_sentence(s) for s in raw_test_data]
    
    # Note: We re-use the *raw* training data for each model.
    # The model class will handle its own preprocessing (padding)
    # based on its 'n'.
    
    # Unigram (N=1)
    mle_unigram = NgramLanguageModel(1, raw_train_data, smoothing='mle', vocab=main_vocab)
    mle_unigram.train()
    pp_mle_uni = mle_unigram.calculate_perplexity(test_data_n1)
    
    # Bigram (N=2)
    mle_bigram = NgramLanguageModel(2, raw_train_data, smoothing='mle', vocab=main_vocab)
    mle_bigram.train()
    pp_mle_bi = mle_bigram.calculate_perplexity(test_data_n2)

    # Trigram (N=3)
    mle_trigram = NgramLanguageModel(3, raw_train_data, smoothing='mle', vocab=main_vocab)
    mle_trigram.train()
    pp_mle_tri = mle_trigram.calculate_perplexity(test_data_n3)

    # 4-gram (N=4)
    mle_4gram = NgramLanguageModel(4, raw_train_data, smoothing='mle', vocab=main_vocab)
    mle_4gram.train()
    pp_mle_4 = mle_4gram.calculate_perplexity(test_data_n4)

    # --- Task 3.2: Smoothing (for Trigram) ---
    
    # Add-1 Smoothing (Trigram)
    add1_trigram = NgramLanguageModel(3, raw_train_data, smoothing='add1', vocab=main_vocab)
    add1_trigram.train()
    pp_add1_tri = add1_trigram.calculate_perplexity(test_data_n3)
    
    # Linear Interpolation (Trigram)
    interp_trigram = NgramLanguageModel(3, raw_train_data, smoothing='interp', vocab=main_vocab)
    interp_trigram.train()
    # Tune lambdas on validation data
    interp_trigram.tune_interpolation(raw_valid_data)
    # Evaluate on test data
    pp_interp_tri = interp_trigram.calculate_perplexity(test_data_n3)

    # --- Task 4: Report Results ---
    
    print("\n--- Perplexity Results ---")
    print("Model\t\t\tPerplexity on Test Set")
    print("-------------------------------------------------")
    print(f"MLE Unigram (N=1)\t\t{pp_mle_uni:.2f}")
    print(f"MLE Bigram (N=2)\t\t{pp_mle_bi}")
    print(f"MLE Trigram (N=3)\t\t{pp_mle_tri}")
    print(f"MLE 4-gram (N=4)\t\t{pp_mle_4}")
    print(f"Trigram + Add-1\t\t{pp_add1_tri:.2f}")
    print(f"Trigram + Interpolation\t{pp_interp_tri:.2f}")
    print("-------------------------------------------------")

    # --- Task 4.4: Generate Text ---
    print("\n--- Generated Sentences (from Trigram + Interpolation Model) ---")
    best_model = interp_trigram
    for i in range(5):
        print(f"{i+1}: {best_model.generate_sentence()}")

if __name__ == "__main__":
    main()