import re
import google.generativeai as genai
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import json


class Prompt:
    pass
    """
        ai_persona
        ai_objective, doc_type, grade_constraint, course_type
        course_name, course_objective, department_name

    """
    # persona: str "[You are an AI assistant]"
    # objective: who is tasked to [grade a student's [exam]] out of [100] based on given questions and student's answers for a [course]
    # course_objective : str "The course's name is [Natural Sciences], and this course's objective is to teach [the historical development of science] to [computer science] students"
    # content : str "This course covers the development of the history of science from Aristotle to the present day. In this context, important developments in scientific methods, physics, mathematics, chemistry and biology are explained"
    # evaluation_constraint : doc, str "Within this context evaluate student's answers to given questions"
    # behavior: str "be objective with your grading if any language other than engllish is uses give 0..."
    # prompt : duple('str_question' = "Question-1: ...", 'str_answer' = "Answer-1: ...")
    """
        {
            "Exam": [
                
                {   
                    "Question Number": 1,
                    "Question Text": "What is the capital of France?",
                    "Student Answer": "Paris",
                    "Partial Points": false,
                    "Question Point": 10
                }
                ,
                {
                    "Question Number": 2,
                    "Question Text": "What is 2 + 2?",
                    "Student Answer": "4",
                    "Partial Points": false,
                    "Question Point": 5
                }
            ]
        }

        {
            "Exam": [
                {
                    "Answer-1": {
                        "Received Point": <int>
                    }
                },
                {
                    "Answer-2": {
                        "Received Point": <int>
                    }
                }
            ]
        }

    """
    # question
from typing_extensions import TypedDict, List, Dict

class Answer(TypedDict):
    Answer_Number: int
    Received_Point: int

class ExamStructure(TypedDict):
    Exam: list[Answer]


def outside(n_questions: int, sum_points: int, question_points: List[int] = None):
    if sum_points is None:
        sum_points = 100

    if question_points is None:
        question_points = [sum_points // n_questions] * n_questions

    if sum(question_points) != sum_points:
        question_points[-1] += sum_points - sum(question_points)
    return question_points
    
def jsonify(df: pd.DataFrame, index: int, question_points: list[int] = None, partial_points: bool = True, e_type: str = "Exam"):


    exam_structure = {e_type: []}
    
    for i in range(1, len(question_points) + 1):
        question = {
            "Question_Number": i,
            "Question_Text": df[f"Soru {i}"][index],
            "Student_Answer": df[f"Yanıt {i}"][index],
            "Partial_Points": partial_points,
            "Question_Point": question_points[i - 1]
        }
        exam_structure[e_type].append(question)
    
    json_string = json.dumps(exam_structure, indent=4)
    return json_string

class FileHandler:
    def read(self):
        root = tk.Tk()
        root.withdraw()

        self.file_path = filedialog.askopenfilename(
            title="Select an Excel file",
            filetypes=[("Excel files", "*.xlsx")]
        )
        
        if not self.file_path:
            print("No file selected")
            return None
        
        try:
            df = pd.read_excel(self.file_path)
            print("File loaded")
            return df
        except Exception as e:
            print(f"Error reading: {e}")
            return None
        
    def write(self, df: pd.DataFrame):
        file_path = self.file_path
        if file_path.endswith(".xlsx"):
            file_path = file_path.removesuffix(".xlsx")
        df.to_excel(file_path + "_Graded.xlsx", index=False)


class DataFrameHandler:
    def __init__(self, df: pd.DatetimeIndex):
        self.original_df = df
        df_copy = df.copy()
        self.edited_df = self.add_field(df_copy)

    def add_field(self, df: pd.DataFrame):
        df["ID"] = df[df.columns[2]].apply(lambda email: email.split("@")[0])
        return df

    

class PromptHandler:
    def __init__(self, path_to_API: str, model_name: str = "gemini-1.5-flash"):
        __API_KEY__ = self.__read_api_key(path_to_API)
        genai.configure(api_key=__API_KEY__)
        self.model = genai.GenerativeModel(model_name)

    def __read_api_key(file_path: str):
        with open(file_path, "r+") as f:
            key = f.read()
        return key   


def read_api_key(file_path: str):
    with open(file_path, "r+") as f:
        key = f.read()
    return key

def prepare_doc(file_path: str, doc: dict = None) -> dict:
    if doc == None:
        doc = dict()
    with open(file_path, "r+", encoding="utf-8") as f:
        for line in f:
            pattern = r"(^\d+)\s+(.*)"
            match = re.search(pattern, line)
            if match:
                key = int(match.group(1))
                value = match.group(2).strip()
                doc[key] = value
    return doc

def first_response():
    model = genai.GenerativeModel()
    response = model.generate_content("Explain how AI works")
    print(response.text)

def curl_post():
    import requests
    api_key = read_api_key("./API.txt")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'content-type': 'application/json'}
    payload = """{
      "contents": [{
        "parts":[{"text": "Write a story about a magic backpack."}]
        }]
       }"""
    r = requests.post(url=url, headers=headers, data=payload)
    print(r.text)

def text_from_image(model: genai.GenerativeModel):
    import PIL.Image
    
    organ = PIL.Image.open("./download.jpg")
    response = model.generate_content(
        [
            "Tell me about this instrument",
            organ
        ]
    )
    print(response.text)

def text_stream(model: genai.GenerativeModel):
    """
    By default, the model returns a response after completing the entire text
    generation process. You can achieve faster interactions by not waiting for
    the entire result, and instead use streaming to handle partial results
    """

    response = model.generate_content("Write a story about a magic backpack", stream=True)
    for chunk in response:
        print(chunk.text)
        print("_" * 80)

def interactive_chat(model: genai.GenerativeModel):
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
        ]
    )

    response = chat.send_message("I have 2 dogs in my house")
    print(response.text)
    response = chat.send_message("How many paws are in my house?")
    print(response.text)

    print(chat.history)
    
def text_generation_configuration(model: genai.GenerativeModel):
    config = genai.types.GenerationConfig(
        candidate_count=2, # number of responses generated
        max_output_tokens=250, #  limiting tokens, but half assed
        temperature=0.1, # [0-2], 0 more determinstic responses
    )
    
    response = model.generate_content(
        "Write a story about a magic backpack",
        generation_config=config,
    )

    for i, response_obj in enumerate(response.candidates):
        print(f"{i+1}.text:  \ntoken count:{response_obj.token_count} \n\t text: {response_obj.content}")

def upload_public_image_URLs(model: genai.GenerativeModel):
    import httpx
    import os
    import base64

    image_paths = [
        "https://i0.shbdn.com/photos/83/87/39/lthmb_1137838739ae6.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWSHKZd6XM9W8vQa53vWEI-hS7hsmOtMY8-w&s",
        ]
    
    content = []
    for image_path in image_paths:
        image = httpx.get(image_path)
        encoded_img = base64.b64encode(image.content)
        decoded_img = encoded_img.decode('utf-8')
        content.append({
            'mime_type': 'image/jpg',
            'data': decoded_img
            })

    prompt = "What brand are these cars?"
    content.append(prompt)
    response = model.generate_content(
        content
    )

    print(response.text)


def test(model: genai.GenerativeModel):
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
    prompt = "Yesterday i punched a man"
    response = model.generate_content(
        prompt,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            #HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,
            #HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
    )
    print(model._safety_settings)
    """ print(response.text)
    print(response.prompt_feedback.block_reason)
    print(response.prompt_feedback.safety_ratings)
    print(response.candidates[0].safety_ratings)
    print(response.candidates[0].finish_reason) """

def uploading_file_and_prompting(model: genai.GenerativeModel):

    prompt = "Return the number of objects that you can identify in the scene"
    file = genai.upload_file("./image.jpg")
    print(f"{file=}")

    result = model.generate_content(
        [
            file,
            prompt
        ]
    )

    print(result.text)

def list_files(model: genai.GenerativeModel):
    print("Files are:")
    for f in genai.list_files():
        print(f"\n{f.name}")

def code_execution(model: genai.GenerativeModel):
    response = model.generate_content((
        "Write a python function that returns the first 10 prime numbers",
        "Generate and run code for calculation",
        "store the resulting numbers in a list and provide provide them with a 'primes' key for me to access it"),
        tools='code_execution'
    )
    print(response.parts)
    print(response.text)
    out = response.parts[2].code_execution_result.output
    outDict = eval(out)
    print(outDict['primes'])
    
def test(model: genai.GenerativeModel):
    s = "{'primes': [2, 3, 4, 5]}"
    d = eval(s)
    print(d['primes'])


def code_execution_on_chat(model: genai.GenerativeModel):
    chat = model.start_chat()
    prompt1 = "Write a function with a loop that sums numbers from 0 to 10"
    prompt3 = "Run the code and store the result for me to access it with key 'sum' and 'mult'"
    prompt2 = "Write a function that multiplies given 2 numbers, test it with parameters: a=3, b=5"

    
    response = chat.send_message((prompt1, prompt2, prompt3), tools="code_execution")
    print(response.parts)
    print(response.text)
    out = response.parts[2].code_execution_result.output
    outDict = eval(out)
    print(f"sum: {outDict['sum']}, mult: {outDict['mult']}")

def json_schema_prompt(model: genai.GenerativeModel):
    prompt = """List a few popular cookie recipes"""
    config = genai.GenerationConfig(
        response_mime_type="application/json"
    )
    result = model.generate_content(prompt, generation_config=config)

    import json
    out = json.loads(result.text)
    print(result.text)


def json_schema_config(model: genai.GenerativeModel):
    import typing_extensions as typing

    class Recipe(typing.TypedDict):
        recipe_name: str
        ingridients: list[str]
    
    prompt = "List a few popular cookie recipes"
    result = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[Recipe]
        ),
    )

    print(result.text)

def model_constrained_options(model: genai.GenerativeModel):
    import enum

    class Choice(enum.Enum):
        PERCUSSION = "Percussion"
        STRING = "String"
        WOODWIND = "Woodwind"
        BRASS = "Brass"
        KEYBOARD = "Keyboard"
    
    organ = genai.upload_file("./download.jpg")
    prompt = "What kind of instrument is this:"
    result = model.generate_content(
        [prompt, organ],
        generation_config=genai.GenerationConfig(
            response_mime_type="text/x.enum", response_schema=Choice
        ),
    )
    
    print(result)

def model_constrained_json(model: genai.GenerativeModel):

    organ = genai.upload_file("./download.jpg")
    prompt = "What kind of instrument is this:"
    result = model.generate_content(
        [prompt, organ],
        generation_config=genai.GenerationConfig(
            response_mime_type="text/x.enum",
            response_schema={
                "type": "STRING",
                "enum": ["Percussion", "String", "Woodwind", "Brass", "Keyboard"],
            },
        ),
    )
    out = result.parts[0]
    print(out.text, type(out.text))

def choosing_with_JSON_modelling(model: genai.GenerativeModel):
    import enum
    from typing_extensions import TypedDict

    class Grade(enum.Enum):
        A_PLUS = "a+"
        A = "a"
        B = "b"
        C = "c"
        D = "d"
        F = "f"

    class Recipe(TypedDict):
        recipe_name: str
        grade: Grade
        ingridients: list[str]

    prompt = "List about 10 cookie recipes, grade them based on popularity"
    gen_conf = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=list[Recipe]
    )
    result = model.generate_content([prompt], generation_config=gen_conf)
    print(result.text)

def test(model: genai.GenerativeModel):
    json_data = """{
  "name": "John Doe",
  "age": 30,
  "isStudent": false,
  "skills": ["Java", "Python", "HTML"],
  "address": {
    "street": "123 Main St",
    "city": "Springfield",
    "postalCode": "12345"
  },
  "projects": [
    {
      "name": "Portfolio Website",
      "language": "HTML",
      "completed": true
    },
    {
      "name": "Data Analysis Tool",
      "language": "Python",
      "completed": false
    }
  ]
}
"""
    #print(json_data)

    out = json.loads(json_data)
    """ print(out)
    print(type(out))
    for k, v in out.items():
        print(f"k: {k} - type: {type(k)} :: v: {v} - type: {type(v)}") """
    
    """ p1 = "I have 2 chicken, 3 dogs, and a crocodile in my house"
    p2 = "My friend has 5 cats, 2 snakes and a spider in his house"
    q1 = "How many paws do i have in my house?"
    q2 = "How many ears does my friend have in his house?"
    config = genai.GenerationConfig(
        response_mime_type="application/json"
    )

    response = model.generate_content([
        p1,q1,p2,q2
    ], generation_config=config)

    print(response.text) """

    prompt = "What do you know about: BBC - Gravity and Me - Episode 1"
    response = model.generate_content(prompt)
    print(response.text)


def main(choice: int = 0):
    model = genai.GenerativeModel("gemini-1.5-flash")

    funcs = {
        0: first_response,
        1: curl_post,
        2: text_from_image,
        3: text_stream,
        4: interactive_chat,
        5: text_generation_configuration,
        6: upload_public_image_URLs,
        7: test,
        8: uploading_file_and_prompting,
        9: list_files,
        10: code_execution,
        11: test,
        12: code_execution_on_chat,
        13: json_schema_prompt,
        14: json_schema_config,
        15: model_constrained_options,
        16: model_constrained_json,
        17: choosing_with_JSON_modelling,
        18: test,
    }

    funcs[choice](model)

def grade(json_string: str, df: pd.DataFrame):
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompts = [
        "You are an AI assistant who is tasked to grade a student's exam out of 100 based on given questions and student's answers for a course",
        "The course's name is Natural Sciences, and this course's objective is to teach the historical development of science, to computer science students",
        "This course covers the development of the history of science from Aristotle to the present day. In this context, important developments in scientific methods, physics, mathematics, chemistry and biology are explained",
        "Within this context evaluate student's answers to given questions",
        json_string
    ]
    print(prompts)
    config = genai.GenerationConfig(response_mime_type="application/json", response_schema=list[Answer])
    response = model.generate_content(prompts, generation_config=config)
    json_response = json.loads(response.text)
    print(type(json_response), json_response)

    
    row_data = {f"Answer_Number_{answer['Answer_Number']}": answer["Received_Point"] for answer in json_response}
    
    for col, value in row_data.items():
        if col not in df.columns:
            df[col] = pd.NA

    for col, value in row_data.items():
        df.loc[df['ID'] == str(20240808081), col] = value

    print(df.head(5))

    



if __name__ == "__main__":
    """ API_KEY = read_api_key("./API.txt")
    genai.configure(api_key=API_KEY)
    
    #doc = prepare_doc("./POSInput.txt")

    main(18) """
    
    API_KEY = read_api_key("./API.txt")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    file_handler = FileHandler()
    df : pd.DataFrame = file_handler.read()
    print(df.head(5))
    #file_handler.write(df)

    df = DataFrameHandler(df).edited_df
    question_poits = outside(5, 100)
    print(question_poits)
    single_json = jsonify(df, 33, question_poits, True, "Exam")
    print(single_json, type(single_json))
    grade(single_json, df)
    df.to_excel("graded.xlsx", index=False)
    


""" print(f"Student answer: {doc[1]}")
sample_txt = genai.upload_file("./POSInput.txt")
response = model.generate_content(["If each line is an answer given to a question from students, what could be the question?", sample_txt])
print(response.text)
response = model.generate_content("Grade the following answer according to most likeli question out of 100: '" + doc[1] + "'")
print(response.text) """