import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.modeling_mplus import MPlus


class NeuralMemory:
    def __init__(self):
        # load the model mplus-8b (currently we only have the pretrained version)
        self.model = MPlus.from_pretrained(
            "YuWangX/mplus-8b",
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained("YuWangX/mplus-8b")
        device = torch.device("mps")
        self.model = self.model.to(device)
        self.model.put_ltm_to_numpy()
        # After this, the usage of MPlus is the same as MemoryLLM-8B, please check "How to use the model" below.

    def persist(self, ctx):
        # please make sure the context to inject into the memory is larger than 16 tokens,
        # this is the hard minimum when training the model.
        # The memory will be disturbed when less than 16 tokens are injected into the memory.
        self.model.inject_memory(
            self.tokenizer(ctx, return_tensors='pt', add_special_tokens=False).input_ids.to(self.model.device),
            update_memory=True
        )

    def query(self, text, max_length=50):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])

    def reflect(
            self,
            text: str,
            max_length: int = 50,
            threshold: float = 0.7,
            beta: float = 1.0
    ) -> str:
        # 1) encode once
        enc = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=False).to(self.model.device)
        ids = enc["input_ids"]

        # 2) single forward to get the CLM loss → surprise
        with torch.no_grad():
            out = self.model(**enc, labels=ids, return_dict=True)
            loss = out.loss
            surprise = beta * (1.0 - torch.sigmoid(-loss)).item()

            print(f"Surprise: {surprise:.2f} >= threshold: {threshold:.2f}")
            if surprise >= threshold:
                print("Persisting...")
                self.persist(text)

        # 3) generate reply
        gen = self.model.generate(**enc,
                                  max_length=max_length,
                                  pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True)
