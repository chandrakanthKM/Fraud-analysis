from django.urls import path
from . import views

urlpatterns = [
	path("", views.index, name="index"),
    path("post/<int:post_id>", views.Detail, name="Detailpage {post_id}"),
	path('old-url/', views.old_url_view, name='old_url'),
    path('new_something_url/', views.new_url_view, name='new_url'),]