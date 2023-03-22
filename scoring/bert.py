from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class Bert(nn.Module):
    def __init__(self, dropout=0.5):
        super(self.__class__, self).__init__()
        
        self.bert = BertModel(BertConfig())
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        
    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        #final_layer = self.relu(linear_output)

        #return final_layer
        return linear_output

def remove_punctuation(text):
    '''Функция получает на вход предложение, удаляет пунктуацию и возвращает обратно'''
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

def remove_stopwords(text):
    '''Функция получает на вход предложение, удаляет стоп слова и возвращает его обратно'''
    stopword = stopwords.words('english')
    words = text.split()
    ans = ''
    for word in words:
        if word not in stopword:
            ans += word + ' '
    return ans

def predict(model, tokenizer, text):
    text = remove_punctuation(text)
    text = text.lower()
    text = remove_stopwords(text)
    print(text)
    data = tokenizer(str(text), padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = data['attention_mask']
    input_ids = data['input_ids'].squeeze(1)

    output = model(input_ids, mask)
    print(output.item())

    return(round(output.item()))