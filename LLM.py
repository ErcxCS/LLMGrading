import re
import google.generativeai as genai
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import json
from typing_extensions import TypedDict, List, Dict

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
    def __init__(self, df: pd.DataFrame):
        self.original_df = df
        df_copy = df.copy()
        self.edited_df = self.add_field(df_copy)

    def add_field(self, df: pd.DataFrame):
        df["ID"] = df[df.columns[2]].apply(lambda email: email.split("@")[0])
        return df
    
    def get_df(self) -> pd.DataFrame:
        return self.edited_df
    
class Course:
    def __init__(self, df: pd.DataFrame):
        # From excel
        pass
    def __init__(self, url: str):
        # From url
        pass
    def __init__(self, name: str, objective: str, department: str, content: str):
        self.name = name
        self.objective = objective
        self.department = department
        self.content = content

class Prompt:
    def __init__(
            self,
            course: Course,
            has_correct_answers: bool = False,
            ai_persona: str = "an AI assistant",
            doc_type: str = "exam",
            total_points: int = 100,
            course_type: str = "course",
    ):

        self.total_points = total_points
        self.doc_type = doc_type

        self.persona = " ".join(["You are", ai_persona])
        self.objective = " ".join(["who is tasked to", "grade a student's", doc_type,])
        self.objective = " ".join([self.objective, "out of", str(total_points), "total points"])
        self.objective = " ".join([self.objective, "based on given questions", "and", "student's answers", "for a", course_type])
        self.course_objective = " ".join(["The", course_type + "'s", "name is", course.name + ","])
        self.course_objective = " ".join([self.course_objective, "and this", course_type + "'s", "objective is to teach"])
        self.course_objective = " ".join([self.course_objective, course.objective, "to", course.department, "students"])

        self.course_content = course.content
        self.evaluation_constraint = "Within this context, evaluate student's answers to given questions"
        if has_correct_answers:
            self.evaluation_constraint = " ".join([self.evaluation_constraint, ", with how their answers contextually similar to provided correct answers"])
        self.behavior = "Be objective with your grading" #+ "and provide a short sentence explaining the reason of the point received for the question"

        self.prompt_list = [
            " ".join([self.persona, self.objective]),
            self.course_objective,
            self.course_content,
            self.evaluation_constraint,
            self.behavior
        ]

class Answer(TypedDict):
    Answer_Number: int
    Received_Point: int
    #Reason: str
# Key termlerde kaldin
class PromptHandler:
    def __init__(self, path_to_API: str, model_name: str = "gemini-1.5-flash"):
        __API_KEY__ = self.__read_api_key(file_path=path_to_API)
        genai.configure(api_key=__API_KEY__)
        self.model = genai.GenerativeModel(model_name)
        self.config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[Answer]
        )

    def __read_api_key(self, file_path: str):
        with open(file_path, "r+") as f:
            key = f.read()
        return key
    
    def jsonify(self, df: pd.DataFrame, index: int, question_points: list[int] = None, partial_points: bool = True, e_type: str = "Exam"):
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
    
    def get_question_points(self, n_questions: int, sum_points: int = 100, question_points = None):
        if sum_points is None:
            sum_points = 100

        if question_points is None:
            question_points = [sum_points // n_questions] * n_questions

        if sum(question_points) != sum_points:
            question_points[-1] += sum_points - sum(question_points)
        return question_points
    
    def build_prompts(self, df: pd.DataFrame):
        question_points = self.get_question_points(5) # find number of questions
        #####
        course_name = "Natural Sciences"
        course_objective = "the historical development of science"
        department_name = "Computer Science"
        course_content = "This course covers the development of the history\
 of science from Aristotle to the present day. In this context,\
 important developments in scientific methods, physics, mathematics,\
 chemistry and biology are explained"
        #####
        course = Course(course_name, course_objective, department_name, course_content)
        self.prompts = {}
        for student in range(len(df)):
            prompt = Prompt(course=course)
            json_string = self.jsonify(df, index=student, question_points=question_points, partial_points=True, e_type=prompt.doc_type)
            print(json_string)
            
            prompt.prompt_list.append(json_string)
            self.prompts[df['ID'][student]] = prompt.prompt_list

    def grade(self, prompt_list: list):
        response = self.model.generate_content(prompt_list, generation_config=self.config)
        print(response.usage_metadata)
        json_response = json.loads(response.text)
        return json_response

    def grade_exam(self, df: pd.DataFrame):
        if "Grade" not in df.columns:
            df["Grade"] = pd.NA 

        for student, prompt in self.prompts.items():
            print(f"grading: {student}")
            try:
                json_response = self.grade(prompt_list=prompt)
            except Exception as e:
                print(e)
                return df
            
            row_data = {}
            for answer in json_response:
                answer_number = answer["Answer_Number"]
                row_data[f"Soru {answer_number} Puan"] = answer["Received_Point"]
                #row_data[f"Reason {answer_number}"] = answer["Reason"]

            for col, value in row_data.items():
                if col not in df.columns:
                    df[col] = pd.NA

            total_grade = 0
            for col, value in row_data.items():
                df.loc[df['ID'] == str(student), col] = value
                if col.endswith("Puan") and isinstance(value, (int, float)):
                    total_grade += value
                
            df.loc[df['ID'] == str(student), "Grade"] = total_grade
        return df



if __name__ == "__main__":
    file_handler = FileHandler()
    df : pd.DataFrame = file_handler.read()
    df = DataFrameHandler(df).get_df()
    prompt_handler = PromptHandler(path_to_API="./API.txt")
    prompt_handler.build_prompts(df)
    prompt_handler.grade_exam(df)
    file_handler.write(df)