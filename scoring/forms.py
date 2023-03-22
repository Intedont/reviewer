from django import forms

class ReviewForm(forms.Form):
    review = forms.CharField(max_length=512, label='Рецензия', widget=forms.Textarea(attrs={"id":"review_input"}))
