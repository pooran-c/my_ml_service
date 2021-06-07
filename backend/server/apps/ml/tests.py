from django.test import TestCase

from .tweets_classifier.ten_flow import  tens_flow_classifier

class MLTests(TestCase):
    def test_tf_algo(self):
        input_data = {'I am very happy'}
        my_algo = tens_flow_classifier("../../research/")

        response = my_algo.compute_prediction(input_data)
        print(response)
        self.assertEqual('OK', response['status'])
        self.assertEqual('joy', response['label'])

