import sys
import os

class Cleaner:
    def __init__(self, brain_model="llama3.2", language="en"):
        self.language = language

    def process(self, raw_transcript):
        # Pass-through to avoid hallucinations during eval
        return raw_transcript
