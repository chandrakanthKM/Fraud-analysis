from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from .models import post
from django.urls import reverse_lazy
 
class BlogListView(ListView):
    model = post
    template_name = 'blog/home.html'   # your template path
    context_object_name = 'posts'
class BlogDetailView(DetailView):
    model = post
    template_name = 'blog/post_detail.html'  # your template path
class BlogCreateView(CreateView):
    model = post
    template_name = 'blog/post_new.html'  # your template path
    fields ='__all__'
class BlogUpdateView(UpdateView):
    model = post
    template_name = 'blog/post_edit.html'  # your template path
    fields = ['title', 'body']
class BlogDeleteView(DeleteView):
    model = post
    template_name = 'blog/post_delete.html'  # your template path
    success_url = reverse_lazy('home')
# Create your views here.
