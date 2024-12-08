# training params
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-05

# bert model params
BERT_MODEL_PATH = '/home/user/project/NLP-Series-text-cls/models/bert-large-uncased'
BERT_HIDDEN_LAYER_SIZE = 1024
# finetuned llm params
LLM_MODEL_PATH = '/home/user/project/LLaMA-Factory/models/llama3_lora_sft_spaces'
# model params
LSTM_HIDDEN_SIZE = 256
GCN_HIDDEN_SIZE = 2048

# dataset params
DATASET_PATH = '../datasets/spaces/'
MAX_LABEL_SIZE = 7