from models import GPTConfig, GPT
import tiktoken 
import torch

if __name__ == "__main__":
    config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=128,       # or whatever context size you're training with
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)
    model = GPT(config)  # re-create the model with same config
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    best_model_params_path = "/scratch/user/hasnat.md.abdullah/TasteSage/src/taste_sage_best_model_params_40k.pt"
    model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device))) # load best model states

    sentence = """This is just another over hyped under performing pizza joint. Although service was good the pizza came up short. Crust: had a strange texture and after taste I could not identify. The sauce: too sweet for my liking. I understand others may like it like this. Toppings: very generous amounts but mediocre quality. Price: on the high side."""
    enc = tiktoken.get_encoding("gpt2")
    context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0))
    y = model.generate(context, 200,temperature=1.0 )
    print("-----")
    print(enc.decode(y.squeeze().tolist()))

 #-----------
# infer result after 20k epochs:
# Expensive for a diner. Average food and good service. It was fine but I won't be going back. Stars:  decent laid back meal and friendly bar food. St. Louis? Stars: 3.0If you haven't been to this matter, downtown visit.  Good atmosphereden's has great music, walk down what you walk into and good food.   The food is fresh and they have the best tasting, but they have some very friendly green of meat and sauce.  The bar itself is cool and welcoming. Stars: 5.0Do not get the southwestern poursite or vegetarian version. They serve up house made Muthere. Their coffee is fresh and delicious. The coffee was amazing.  I also had hot braised tea hot cure option with a caramel and employee got the blueberry muffin. It also offers a steam on the side is pretty good. Stars: 4.0As Zone is like close to the mall, this is cafeteria staple hidden gem in South Philly. This place is really nice for anyone when in Indiana. We love the atmosphere, large groups food and a


#-----
#  The food is awful -> and the staff is very nice. Stars: 3.0My first time here and I had the crab cakes and my wife had the shrimp and grits. Both were very good. The service was great. Our waiter came over to our table to check on us, which was great. We had the crab cakes, crab cake and it. I had the crab cake and I had the chocolate cake. Both were good. The service was very good and our server was very friendly. Stars: 4.0This is one of our favorite places to eat.  I love their food, especially the beer and the beer is amazing!  The staff is always friendly, and the atmosphere is very nice.  The service is great! Stars: 4.0I have to give it another chance to go back to this location. We are a couple of times and the food is consistently great. I've been here twice.  The food is good and the drinks are good. Stars: 5


#----
# Went there for the first time today. Not a huge menu to choose from but the food was ok. A little bit pricey though. KFC 5 dollar boxes are a better deal and just a good. Probably will be back but not real soon. ->  Stars: 3.0Having opening every box in the letter by a gas station... Ted Drew is when I went to opening it at Toputs.   I definitely recommended their vegetarian markets twist but I sublimely found it very busy.  Love the variety.  Their special menu is curated with a inspired item of flavor and orders is well prepared to make it right back for us.  They didn't seem wonderful but, the sweet bartender had me the portion size separately which stands out the kamips can do while my husband's favorite board tried.  Also also recommend the Fall Pig Crab sandwich.
# We only had the salmon roll - and the Chicken Salad, which tasted really just sour and delicious.  My first dish came with chicken enchilada with avocado and sweet onions. The pot chicken was divine.  My husband ordered an.... YOU ARE THIS GOOD!! Crispy Italian culinary cuts. Perhaps they could make that crunch like we also had the sauces.  We are

# ---
