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

    def reflect(self, text: str, max_length = 50, threshold: float = 0.7) -> float:
        """Measure how *surprising* ``user_input`` is and persist it when novelty is high.

        Evaluates the model's own likelihood of the observed input as an introspective signal.
        high loss indicates the model was uncertain about the user's message,
        so we store the input for later recall.

        Args:
            :param text: Prompt from the user.
            :param max_length: Maximum length of the response.
            :param threshold: Threshold for surprise.

        Returns:
            The response to the query.
        """

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        if inputs.shape[1] < 2:
            return 0.0

        with torch.no_grad():
            outputs = self.model(inputs[:, :-1], return_dict=True)
            logits = outputs.logits

        target = inputs[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            reduction="mean",
        )
        certainty = torch.sigmoid(-loss).item()
        surprise = 1.0 - certainty

        if surprise >= threshold:
            self.persist(text)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])
