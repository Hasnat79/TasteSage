# Tiny stories slm 


An SLM is built using 2M tiny stories dataset. 
---
- [Models](/tiny_stories_slm/models/) contains the basic configuration and design classes of gpt2 model 

- [data_loader.py](/tiny_stories_slm/data_loader.py) file downloads and the tiny stories dataset from HF, maps the texts with gpt2 tokenizer and save the train, validation tokenized data as a binary file in local disk.

- [train_slm.py](/tiny_stories_slm/train_slm.py) trains the model with necessary training settings and hyperparameters following the reference tutorial video

- [slm_inference.py](/tiny_stories_slm/slm_inference.py) loads the gpt2 model and best model saved by the `train_slm.py` model and generates stories from a given input prompt upto max new tokens.

### Reference
Here is the reference [video](https://www.youtube.com/watch?v=pOFcwcwtv3k&t=9935s) that has been followed to develop this. 