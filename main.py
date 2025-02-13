from google import genai

MODEL_NAME = 'gemini-2.0-flash' 
def __read_api_key(file_path: str):
    with open(file_path, "r+") as f:
        key = f.read()
    return key


def generate_simple_text(client: genai.Client):
    c = 1
    new_text = None
    prompts = [
        "Once there was a cat called Tony",
        "One day, while Tony was laying on a couch, he sees a mouse",
        "Continue this story."
    ]
    while True:
        print(f"##### {c} ######")
        c += 1

        if new_text is not None:
            prompts = prompts[:-1] + [new_text] + [prompts[-1]]

        print(prompts)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompts
        )
        print(response.usage_metadata)
        #print(response.text)
        new_text = response.text


def main(choice: int):
    API_KEY = __read_api_key("./API.txt")
    client = genai.Client(api_key=API_KEY)
    model_name = 'gemini-2.0-flash'

    funcs = {
        1: generate_simple_text
    }

    funcs[choice](client)

if __name__ == "__main__":
    main(1)