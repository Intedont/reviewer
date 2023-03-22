from django.shortcuts import render
from scoring.forms import ReviewForm
from django.http import HttpResponse
from transformers import BertTokenizer
from scoring.bert import Bert, predict
import torch

model = Bert()
model.load_state_dict(torch.load('scoring/static/best.pt', map_location='cpu'))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def index(request):
    answer = None
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data.get('review')
            print(review_text)
            answer = predict(model, tokenizer, review_text)
    else:
        form = ReviewForm()

    context = {'form': form, 'answer': answer}
    return render(request, 'index.html', context)