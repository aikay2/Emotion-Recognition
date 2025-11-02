from django.db import models

# Create your models here.
class Submission(models.Model):
    name = models.CharField(max_length=200)
    email = models.EmailField()
    image = models.ImageField(upload_to='uploads/')
    predicted_emotion = models.CharField(max_length=50, blank=True)
    message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.predicted_emotion}"