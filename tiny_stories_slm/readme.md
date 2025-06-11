# Tiny stories slm 


An SLM is built using 2M tiny stories dataset. 
---
- [Models](/tiny_stories_slm/models/) contains the basic configuration and design classes of gpt2 model 

- [data_loader.py](/tiny_stories_slm/data_loader.py) file downloads and the tiny stories dataset from HF, maps the texts with gpt2 tokenizer and save the train, validation tokenized data as a binary file in local disk.

- [train_slm.py](/tiny_stories_slm/train_slm.py) trains the model with necessary training settings and hyperparameters following the reference tutorial video

- [slm_inference.py](/tiny_stories_slm/slm_inference.py) loads the gpt2 model and best model saved by the `train_slm.py` model and generates stories from a given input prompt upto max new tokens.

    - Download the trained model (60k Epoch) file
    ``` 
    wget -O best_model_params_60K_EP.pt "https://huggingface.co/hasnat79/tiny_stories_gpt2_60k_epoch/resolve/main/best_model_params_60K_EP.pt?download=true"
    ```

    - update the path 
    ``` 
    best_model_params_path = "best_model_params_60K_EP.pt"
    ``` 
    - update the prompt and run the `slm_inference.py` file. 
    ```
    sentence = "A little girl went to the woods"
    ```
    

### Reference
Here is the reference [video](https://www.youtube.com/watch?v=pOFcwcwtv3k&t=9935s) that has been followed to develop this. 