from django.shortcuts import render
from django.http import JsonResponse
from .forms import SubmissionForm
from .models import Submission
from .prediction import predict_emotion, map_to_message  # updated for emoji

def index(request):
    """
    Handles normal form submissions and AJAX submissions.
    Returns JSON for AJAX requests or renders template with results.
    """
    if request.method == 'POST':
        form = SubmissionForm(request.POST, request.FILES)
        if form.is_valid():
            submission = form.save(commit=False)
            submission.save()  # now submission.image.path exists

            image_path = submission.image.path
            try:
                emotion = predict_emotion(image_path)
                message, emoji = map_to_message(emotion)
            except Exception as e:
                emotion = "error"
                message, emoji = (f"Error processing image: {str(e)}", "⚠️")

            submission.predicted_emotion = emotion
            submission.message = message
            submission.save()

            # AJAX response
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'emotion': emotion.capitalize(),
                    'emoji': emoji,
                    'message': message,
                    'image_url': submission.image.url,
                })

            # normal form submission response
            result = {
                'emotion': emotion.capitalize(),
                'emoji': emoji,
                'message': message,
                'image_url': submission.image.url,
            }
            return render(request, 'core/index.html', {'form': SubmissionForm(), 'result': result})

        else:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': 'Invalid form submission'})
            # normal fallback
            return render(request, 'core/index.html', {'form': form})

    else:
        form = SubmissionForm()

    return render(request, 'core/index.html', {'form': form})
