import inspect

from django.test import TestCase

from .registry import MLRegistry
from .tweets_classifier.ten_flow import  tens_flow_classifier

class MLTests(TestCase):
    '''
    def test_tf_algo(self):
        input_data = {'I am very happy'}
        my_algo = tens_flow_classifier("../../research/")

        response = my_algo.compute_prediction(input_data)
        print(response)
        self.assertEqual('OK', response['status'])
        self.assertEqual('joy', response['label'])
    '''
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "income_classifier"
        algorithm_object = tens_flow_classifier("../../research/")
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Pooran"
        algorithm_description = "rnn in tf"
        algorithm_code = inspect.getsource(tens_flow_classifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                               algorithm_status, algorithm_version, algorithm_owner,
                               algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)

