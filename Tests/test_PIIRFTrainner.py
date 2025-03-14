from unittest import TestCase
from Classifiers.GBMclassifier import *

class TestPIIGBMTrainner(TestCase):
    def test_classify_piidatagram_proba(self):
        trainner:PIIGBMTrainner = PIIGBMTrainner.loadFromFile("../save.clf")

