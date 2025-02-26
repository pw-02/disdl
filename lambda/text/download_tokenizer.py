# import nltk
# nltk.download('stopwords', download_dir='./nltk_data')
# nltk.download('wordnet', download_dir='./nltk_data')

# from transformers import GPT2Tokenizer, GPT2Model

# # Download and cache the tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Save the tokenizer
# tokenizer.save_pretrained('./gpt2-tokenizer')


from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer

# # Specify the tokenizer name
# tokenizer_name = "EleutherAI/pythia-70m"

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Save the tokenizer
tokenizer.save_pretrained('bert-tokenizer')