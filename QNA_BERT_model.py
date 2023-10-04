from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import os
from pdfminer.high_level import extract_text

# Load the pre-trained BERT model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # You can choose a different BERT model
# Load a different pre-trained BERT model and tokenizer
#model_name = "bert-base-uncased"  # You can choose a different BERT model

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def replace_tokens(tokens):
    result = []
    i = 0
    while i < len(tokens):
        if i + 4 < len(tokens) and tokens[i] == '(' and tokens[i + 1] == 'cid' and tokens[i + 2] == ':' and tokens[i + 3] == '415' and tokens[i + 4] == ')':
            # Replace ('cid', ':', '415') with '##ti'
            result.append('##ti')
            i += 5
        else:
            # Keep other tokens as they are
            result.append(tokens[i])
            i += 1
    return result

def merge_tokens(tokens):
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i + 1].startswith('##'):
            # Merge tokens like "slim ##y" into "slimy"
            merged_tokens.append(tokens[i] + tokens[i + 1][2:])
            i += 2
        else:
            # Keep other tokens as they are
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens

def remove_hash(tokens):
    return [token.replace('##', '') for token in tokens]

# Function to answer a question given a context and a question
def answer_question(context, question):
    # Tokenize the question and context separately
    question_tokens = tokenizer.tokenize(question)
    context_tokens = tokenizer.tokenize(context)
       
    # Ensure the combined length does not exceed the model's maximum sequence length
    max_seq_length = tokenizer.model_max_length
    total_length = len(question_tokens) + len(context_tokens) + 2  # Add 2 for [CLS] and [SEP]

    print("model_max_length : ", str(max_seq_length), "total_lenght : " , str(total_length))
    
    if total_length > max_seq_length:
        # If the combined length exceeds the maximum, truncate the context
        max_context_length = max_seq_length - len(question_tokens) - 3
        context_tokens = context_tokens[:max_context_length]
    
    # Combine the question and context tokens with [SEP] token in between
    input_tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + context_tokens + ["[SEP]"]

    print("Total tokens generated are : " , str(len(input_tokens)))
    print("input_tokens: ", str(input_tokens))
    
    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    print("converted input_ids: ", str(input_ids))
    
    # Convert input IDs to tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    
    # Get the start and end logits for the answer
    outputs = model(input_ids)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # Find the tokens with the highest probability as the start and end positions
    start_index = start_logits.argmax().item()
    end_index = end_logits.argmax().item()

    # Check if the indices are within the valid range
    if start_index >= len(input_tokens) or end_index >= len(input_tokens):
        return "Answer not found"  # Handle cases where indices are out of range
    
           
    # Get the answer span from the tokens
    answer_tokens = input_tokens[start_index+1 : end_index]
    #print("tokens ", str(answer_tokens))

    # Filter out special tokens from the answer_tokens
    filtered_answer_tokens = [token for token in answer_tokens if not tokenizer.special_tokens_map.get(token)]
    print("filtered_answer_tokens 1", str(filtered_answer_tokens))
    
    #filtered_answer_tokens = remove_hash(filtered_answer_tokens)
    
    #filtered_answer_tokens = [token.replace('##', '') for token in filtered_answer_tokens]
    #filtered_answer_tokens = [token.replace('cid', 't') for token in filtered_answer_tokens]
    #filtered_answer_tokens = [token.replace('415', 'i') for token in filtered_answer_tokens]
    #filtered_answer_tokens = [token.replace('(', '') for token in filtered_answer_tokens]
    #filtered_answer_tokens = [token.replace(':', '') for token in filtered_answer_tokens]
    #filtered_answer_tokens = [token.replace(')', '') for token in filtered_answer_tokens]

    filtered_answer_tokens = replace_tokens(filtered_answer_tokens)

    filtered_answer_tokens = merge_tokens(filtered_answer_tokens)

    print("filtered_answer_tokens 2", str(filtered_answer_tokens))

    # Combine the tokens in the answer and print it out.
    answer = ' '.join(filtered_answer_tokens)
    #answer = tokenizer.decode(filtered_answer_tokens)
    print("in fun: ", answer)
    
    return answer

if __name__ == "__main__":
    pdf_file_path = "./teeth.pdf"  # Replace with the path to your PDF file
    if os.path.exists(pdf_file_path):
        pdf_text = extract_text(pdf_file_path)
        
        while True:
            user_question = input("Enter your question (or 'q' to quit): ")
            
            if user_question.lower() == 'q':
                break
            
            answer = answer_question(pdf_text, user_question)
            print("Answer:", answer)
    else:
        print("PDF file not found.")
