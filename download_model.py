'''
Created on Jan 9, 2025

@author: Anssi Jääskeläinen
'''

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = 'openai/whisper-large-v3-turbo'
AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
AutoProcessor.from_pretrained(model_id)