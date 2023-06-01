# Gesture-CLIP

----
# NOTE: This is work in progress use at your own risk xD

I have modified CLIP for our use case in GENEA challenge 2023. 


## Approach
From the orignal clip, the Image encoder is replaced by the input encoder and text encoder is replaced by motion encoder. Both are Transformer encoder models.
![CLIP](CLIP.png)



## Usage
Install latest version of pytorch and lightning
### Installation
```bash
pip install -r requirements.txt
# pip install this branch
pip install git+https://github.com/shivammehta25/CLIP.git@GestCLIP
```

### Getting embeddings

```python
import torch
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CHECKPOINT_PATH, device=device)

with torch.no_grad():
    features_from_inputs = model.embed({
            'text': torch.randn(500, 768),
            'audio': torch.randn(500, 768) # You can send mismatched size, it will interpolate and send output as the maxed value
    })
    features_from_target_motion = model.embed_motion({
        'motion': torch.randn(500, 60)
    })
    
    logits_per_image, logits_per_text = features_from_inputs.mean(1), features_from_target_motion.mean(1) 
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```