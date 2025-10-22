import math
from collections import defaultdict, Counter
import numpy as np
from utils import START_TOKEN, END_TOKEN, UNK_TOKEN

class NgramLanguageModel:
    """
    Class for N-gram Language Models.
    
    Args:
        n (int): The order of the N-gram model (e.g., 1 for unigram, 2 for bigram).
        train_data (list of list of str): The preprocessed training sentences.
        smoothing (str): The smoothing technique to use ('mle', 'add1', 'interp').
        vocab (set): The vocabulary built from the training data.
    """
    def __init__(self, n, train_data, smoothing='mle', vocab=None):
        self.n = n
        self.smoothing = smoothing
        
        # Build vocabulary if not provided
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = self.build_vocab(train_data)
        
        # Preprocess data by mapping words to vocab or <unk>
        self.processed_data = [self.preprocess_sentence(sent) for sent in train_data]
        
        # N-gram counts
        self.ngram_counts = defaultdict(int)
        # (n-1)-gram (prefix) counts
        self.prefix_counts = defaultdict(int)
        # Total token count (for unigram and perplexity)
        self.token_count = 0
        
        # For smoothed models
        self.vocab_size = len(self.vocab)
        
        # For interpolation
        self.lambda1 = 0
        self.lambda2 = 0
        self.lambda3 = 0
        self.unigram_model = None
        self.bigram_model = None

    def build_vocab(self, train_data, freq_threshold=1):
        """
        Builds a vocabulary from the training data, mapping infrequent words to <unk>.
        
        Args:
            train_data (list of list of str): Raw training sentences.
            freq_threshold (int): Words with a frequency <= this will be mapped to <unk>.
            
        Returns:
            set: A set of words in the vocabulary.
        """
        word_counts = Counter(word for sent in train_data for word in sent)
        vocab = {word for word, count in word_counts.items() if count > freq_threshold}
        vocab.add(UNK_TOKEN)
        vocab.add(START_TOKEN) # <s> is not in PTB data, but </s> is
        vocab.add(END_TOKEN)
        print(f"Vocabulary size created: {len(vocab)} words.")
        return vocab

    def preprocess_sentence(self, sentence):
        """
        Processes a single sentence, adding start/end tokens and mapping OOVs to <unk>.
        
        Args:
            sentence (list of str): A single sentence as a list of words.
            
        Returns:
            list of str: The processed sentence.
        """
        # Add (n-1) start tokens for n-gram models
        padding = [START_TOKEN] * (self.n - 1)
        # Map words to <unk> if not in vocab, then add end token
        processed = padding + [word if word in self.vocab else UNK_TOKEN for word in sentence] + [END_TOKEN]
        return processed

    def train(self):
        """
        Trains the N-gram model by counting n-grams and (n-1)-gram prefixes.
        """
        print(f"Training {self.n}-gram model (Smoothing: {self.smoothing})...")
        
        if self.smoothing == 'interp':
            # Interpolation requires training 1, 2, and 3-gram MLE models
            # We assume n=3 for interpolation as per homework spec
            self.unigram_model = NgramLanguageModel(1, [[]], smoothing='mle', vocab=self.vocab)
            self.unigram_model.processed_data = self.processed_data # Use same processed data
            self.unigram_model.train()
            
            self.bigram_model = NgramLanguageModel(2, [[]], smoothing='mle', vocab=self.vocab)
            self.bigram_model.processed_data = self.processed_data
            self.bigram_model.train()
            
            # Continue to train the trigram part for this model
            pass # Fall through to train trigram counts

        for sentence in self.processed_data:
            # We count </s> as a token, but not <s>
            # Start from index (n-1) to skip padding
            for i in range(self.n - 1, len(sentence)):
                self.token_count += 1
                
                # Get the n-gram (e.g., trigram: (w_i-2, w_i-1, w_i))
                ngram = tuple(sentence[i - self.n + 1 : i + 1])
                self.ngram_counts[ngram] += 1
                
                # Get the prefix (e.g., (w_i-2, w_i-1))
                prefix = tuple(sentence[i - self.n + 1 : i])
                self.prefix_counts[prefix] += 1
                
                # Special case for Unigram (n=1): prefix is empty, count is total tokens
                if self.n == 1:
                    self.prefix_counts[()] = self.token_count

    def tune_interpolation(self, validation_data):
        """
        Finds the optimal lambda weights for interpolation using the validation set.
        This function performs a simple grid search.
        
        Args:
            validation_data (list of list of str): The preprocessed validation sentences.
        """
        if self.smoothing != 'interp' or self.n != 3:
            return

        print("Tuning interpolation lambdas on validation data...")
        best_lambdas = (0, 0, 0)
        best_perplexity = float('inf')
        
        # Simple grid search with a step of 0.1
        steps = np.arange(0.0, 1.1, 0.1)
        for l1 in steps:
            for l2 in steps:
                l3 = 1.0 - l1 - l2
                if l3 < 0:
                    continue
                
                self.lambda1 = l1
                self.lambda2 = l2
                self.lambda3 = l3
                
                # Preprocess validation data for a 3-gram model
                processed_valid = [self.preprocess_sentence(sent) for sent in validation_data]
                
                perplexity = self.calculate_perplexity(processed_valid)
                
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_lambdas = (l1, l2, l3)

        self.lambda1, self.lambda2, self.lambda3 = best_lambdas
        print(f"Best lambdas found: λ1={self.lambda1:.1f}, λ2={self.lambda2:.1f}, λ3={self.lambda3:.1f} (PP={best_perplexity:.2f})")

    def get_ngram_probability(self, ngram):
        """
        Calculates the probability of a single n-gram based on the model's
        smoothing strategy.
        
        Args:
            ngram (tuple of str): The n-gram to calculate the probability for.
            
        Returns:
            float: The probability of the n-gram.
        """
        prefix = ngram[:-1]
        ngram_count = self.ngram_counts.get(ngram, 0)
        prefix_count = self.prefix_counts.get(prefix, 0)

        if self.smoothing == 'mle':
            if prefix_count == 0:
                # This happens for unseen prefixes, prob is 0
                return 0.0
            return ngram_count / prefix_count
        
        elif self.smoothing == 'add1':
            return (ngram_count + 1) / (prefix_count + self.vocab_size)
        
        elif self.smoothing == 'interp':
            # Assumes n=3
            p1 = self.unigram_model.get_ngram_probability((ngram[2],))
            p2 = self.bigram_model.get_ngram_probability(ngram[1:])
            
            # Get trigram MLE probability
            if self.prefix_counts.get(prefix, 0) == 0:
                p3 = 0.0
            else:
                p3 = self.ngram_counts.get(ngram, 0) / self.prefix_counts.get(prefix, 0)
                
            return (self.lambda3 * p3) + (self.lambda2 * p2) + (self.lambda1 * p1)
        
        else:
            raise ValueError(f"Unknown smoothing type: {self.smoothing}")

    def calculate_perplexity(self, test_data):
        """
        Calculates the perplexity of the model on a given test set.
        
        Args:
            test_data (list of list of str): The preprocessed test sentences.
            
        Returns:
            float: The perplexity score. Returns float('inf') if a zero probability
                   is encountered in an MLE model.
        """
        log_prob_sum = 0.0
        test_token_count = 0

        for sentence in test_data:
            # Start from index (n-1) to skip padding
            for i in range(self.n - 1, len(sentence)):
                test_token_count += 1
                ngram = tuple(sentence[i - self.n + 1 : i + 1])
                
                prob = self.get_ngram_probability(ngram)
                
                if prob == 0.0:
                    # As per instructions, if any prob is 0, perplexity is infinite
                    if self.smoothing == 'mle':
                        return float('inf')
                    else:
                        # Smoothing should prevent this, but handle defensively
                        return float('inf')
                
                log_prob_sum += math.log2(prob)
        
        avg_log_prob = log_prob_sum / test_token_count
        perplexity = 2 ** (-avg_log_prob)
        return perplexity

    def generate_sentence(self, max_length=20):
        """
        Generates a random sentence based on the trained model's probabilities.
        
        Args:
            max_length (int): The maximum number of words for the sentence.
            
        Returns:
            str: The generated sentence.
        """
        if self.smoothing == 'mle':
            print("Warning: Generating text with MLE model is difficult and may get stuck.")
        
        # Start with the required padding
        context = [START_TOKEN] * (self.n - 1)
        sentence = []
        
        # Create a list of the vocabulary to sample from
        vocab_list = list(self.vocab - {START_TOKEN})
        
        for _ in range(max_length):
            # 1. Get the probability distribution for the next word
            probabilities = []
            prefix = tuple(context[-(self.n - 1):])
            
            for word in vocab_list:
                ngram = prefix + (word,)
                prob = self.get_ngram_probability(ngram)
                probabilities.append(prob)
            
            # 2. Normalize probabilities (they might not sum to 1 perfectly)
            prob_sum = sum(probabilities)
            if prob_sum == 0.0:
                # This can happen if the context is unseen and smoothing is poor
                # We'll just stop the sentence here.
                break
            
            probabilities = [p / prob_sum for p in probabilities]

            # 3. Sample a word from the distribution
            next_word = np.random.choice(vocab_list, p=probabilities)
            
            # 4. Check for end of sentence
            if next_word == END_TOKEN:
                break
            
            # 5. Append and update context
            sentence.append(next_word)
            context.append(next_word)
            
        return " ".join(sentence)