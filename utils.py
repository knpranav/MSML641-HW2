# --- Constants ---
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

def load_data(filepath):
    """
    Loads a PTB file, splitting sentences into lists of words.
    
    Args:
        filepath (str): Path to the .txt file.
        
    Returns:
        list of list of str: A list of sentences, where each sentence
                            is a list of its words.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        # .strip() removes leading/trailing whitespace (like newlines)
        # .split() tokenizes on spaces
        return [line.strip().split() for line in f.readlines()]