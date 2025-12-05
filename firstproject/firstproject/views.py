from django.shortcuts import render

def custom_404_view(request, exception):
    return render(request, 'firstproject/Template/404.html', status=404)