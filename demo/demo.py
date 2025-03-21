from src.neural_memory import NeuralMemory


def main():
    nm = NeuralMemory()
    ctx = "Last week, John had a wonderful picnic with David. During their conversation, David mentioned multiple times that he likes eating apples. Though he didn't mention any other fruits, John says he can infer that David also like bananas."
    nm.persist(ctx)
    prompt = "Question: What fruits does David like? Answer: David likes"
    response = nm.query(prompt, max_length=30)
    print("Prompt:")
    print(prompt)
    print("Output:")
    print(response)


if __name__ == "__main__":
    main()
