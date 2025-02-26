corpus="""Hello Welcome,to Krish Naik's NLP Tutorials.
Please do watch the entire course! to become expert in NLP.
"""
print(corpus, "---> corpus")

##  Tokenization
## Sentence-->paragraphs
## Tokenization 
## Paragraph-->words
## sentence--->words

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
# nltk.download('punkt')
# nltk.download('punkt_tab')
documents = sent_tokenize(corpus)    
treeBank = TreebankWordTokenizer()
for sentence in documents:
    words = word_tokenize(sentence)
    tokenise = treeBank.tokenize(sentence)
    print(words, "---> words")
    print(tokenise, '---> tokenise')

