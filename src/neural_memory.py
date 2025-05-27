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
        """
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
        # 1) tokenize once and move to device
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)
        input_ids = encoding["input_ids"]

        # 2) single forward to get the loss via labels
        with torch.no_grad():
            output = self.model(
                **encoding,
                labels=input_ids,
                # huggingface will compute loss internally
                return_dict=True
            )
            loss = output.loss

        surprise = 1.0 - torch.sigmoid(-loss).item()
        if surprise >= threshold:
            self.persist(text)

        if surprise >= threshold:
            self.persist(text)

        # 3) single generate call for the reply
        gen_ids = self.model.generate(
            **encoding,
            attention_mask=encoding["attention_mask"],
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )
        reply = self.tokenizer.decode(
            gen_ids[0, input_ids.size(1):],
            skip_special_tokens=True
        )

        return reply