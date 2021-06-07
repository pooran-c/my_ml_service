"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

# ML registry
import inspect
from ..apps.ml.registry import  MLRegistry

from ..apps.ml.tweets_classifier.ten_flow import tens_flow_classifier

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    rf = tens_flow_classifier("../../research/")
    # add to ML registry
    registry.add_algorithm(endpoint_name="income_classifier",
                            algorithm_object=rf,
                            algorithm_name="random forest",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Piotr",
                            algorithm_description="rnn",
                            algorithm_code=inspect.getsource(tens_flow_classifier))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
