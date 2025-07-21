import json
import sys
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin


class QualityModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(QualityModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)


device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup configuration and model
config = AutoConfig.from_pretrained("nvidia/quality-classifier-deberta")
tokenizer = AutoTokenizer.from_pretrained("nvidia/quality-classifier-deberta")
model = QualityModel.from_pretrained("nvidia/quality-classifier-deberta").to(device)
model.eval()

# Prepare and process inputs
"""
text_samples = [".?@fdsa Low quality text.", "This sentence is ok."]
inputs = tokenizer(
    text_samples, return_tensors="pt", padding="longest", truncation=True
).to(device)
outputs = model(inputs["input_ids"], inputs["attention_mask"])

# Predict and display results
predicted_classes = torch.argmax(outputs, dim=1)
predicted_domains = [
    config.id2label[class_idx.item()] for class_idx in predicted_classes.cpu().numpy()
]
print(predicted_domains)
"""

for doc in sys.stdin:
    try:
        t = json.loads(doc)
    except:
        continue
    text, i, reg = t["text"], t["id"], t["register"]
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True).to(device)
    outputs = model(inputs["input_ids"], inputs["attention_mask"])
    predicted_classes = torch.argmax(outputs, dim=1)
    predicted_domains = [config.id2label[class_idx.item()] for class_idx in predicted_classes.cpu().numpy()]
    result = {
        "id": i,
        "text": text,
        "register": reg,
        "predicted_class": predicted_classes,
        "predicted_domain": predicted_domains
    }

    print(result)