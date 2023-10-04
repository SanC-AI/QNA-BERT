# QNA-BERT
BERT based Question and Answering Python code

I have used "bert-large-uncased-whole-word-masking-finetuned-squad" model, which is fine-tuned for the Stanford Question Answering Dataset (SQuAD).
It has a maximum sequence length of 384 tokens. This means that the total number of tokens in both the input sequence (question + context) and the output sequence (answer) combined should not exceed 384 tokens.

#Learnings
1. the tokenizer splits the word like "sticky" into tokens ['s', '(', 'cid', ':', '415', ')', 'ck', '##y']. If any word contains "ti" in it this tokenizer converts it into ['(', 'cid', ':', '415', ')'] tokens. Its our responsibility to convert it back to "ti"
2. this models tokenizer splits word like "decay" into ['dea', '##ca', '##y']and "germs" into ['ge', '##rm', '##s'] token. So again its our responsibility to remove '##' and combine the tokens to make 1 word.
3. model "bert-base-uncased" is un-trained, before using this model, we need to train it first, other wise, it is giving whole bunch of tokens from where it has match the first token from question.
4. this is the memory heavy model, for my 289KB of PDF, it has taken ~1.588GB of RAM.
5. Also it is CPU intensive when loading the PDF as well as finding the answers. It has taken 25% of CPU from my 12th Gen i5-1235U 1.30GHz.
