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
    score = 0
    status = ''
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data.get('review')
            print(review_text)
            score = predict(model, tokenizer, review_text)
            if(score >= 7):
                status = 'положительный'
            else:
                status = 'отрицательный'
    else:
        form = ReviewForm()

    context = {'form': form, 'score': score, 'status': status}
    return render(request, 'index.html', context)