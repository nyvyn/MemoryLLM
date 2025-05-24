from src.neural_memory import NeuralMemory


def main():
    nm = NeuralMemory()

    print(f"### - Initial test, no context:")
    prompt = "Question: What foods does David like? Answer: David likes"
    response = nm.query(prompt, max_length=30)
    print("Prompt:")
    print(prompt)
    print("Output:")
    print(response)

    print(f"### - In-context learning:")
    ctx = ("Last week, John had a wonderful picnic with David. During their conversation, David mentioned multiple "
           "times that he likes eating nuts. Though he didn't mention any other foods, John says he can infer that "
           "David also likes other salty foods.")
    response = nm.query(ctx + ' ' + prompt, max_length=90)
    print("Prompt:")
    print(prompt)
    print("Output:")
    print(response)

    print(f"### - Memory injection")
    nm.persist(ctx)
    response = nm.query(prompt, max_length=30)
    print("Prompt:")
    print(prompt)
    print("Output:")
    print(response)


if __name__ == "__main__":
    main()