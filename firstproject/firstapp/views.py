from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.urls import reverse
posts= [
        {'id': 1, 'title': 'First Post', 'content': 'This is the content of the first post.'},
        {'id': 2, 'title': 'Second Post', 'content': 'This is the content of the second post.'},
        {'id': 3, 'title': 'Third Post', 'content': 'This is the content of the third post.'},
    ]
def index(request):
    Blog_Title = "Chandrakanth's Blog"
   
    return render(request, 'index.html', {'Blog_Title': Blog_Title, 'posts': posts})
# Create your views here.
def Detail (request,post_id):
       next(post_id)
       return render(request, 'detail.html')

def old_url_view(request):
    return redirect(reverse('new_url'))
def new_url_view(request):
    return HttpResponse("This is the new URL view.")