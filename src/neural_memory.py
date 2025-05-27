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

    def query(self, text, max_length=50):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])

    def persist(self, ctx):
        # please make sure the context to inject into the memory is larger than 16 tokens,
        # this is the hard minimum when training the model.
        # The memory will be disturbed when less than 16 tokens are injected into the memory.
        self.model.inject_memory(
            self.tokenizer(ctx, return_tensors='pt', add_special_tokens=False).input_ids.to(self.model.device),
            update_memory=True
        )

    def evaluate_and_update(self, user_input: str, response: str, threshold: float = 0.7) -> float:
        """Assess the certainty of ``response`` given ``user_input`` and update long-term memory
        when the certainty is above ``threshold``.

        The certainty estimate roughly follows the idea of Intuitor by using the
        model's own likelihood over the generated tokens as an introspective
        signal.

        Args:
            user_input: Prompt from the user.
            response: Model generated response to ``user_input``.
            threshold: Value between 0 and 1. Memory is updated when the
                certainty is greater or equal to this value.

        Returns:
            The computed certainty score.
        """

        # Encode the full conversation and locate the response tokens.
        input_ids = self.tokenizer(user_input + response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        prompt_ids = self.tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)

        response_length = input_ids.shape[1] - prompt_ids.shape[1]
        if response_length <= 0:
            return 0.0

        with torch.no_grad():
            outputs = self.model(input_ids[:, :-1], return_dict=True)
            logits = outputs.logits[:, -response_length:, :]

        target = input_ids[:, -response_length:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="mean")
        certainty = torch.sigmoid(-loss).item()

        if certainty >= threshold:
            self.persist(user_input + " " + response)

        return certainty
