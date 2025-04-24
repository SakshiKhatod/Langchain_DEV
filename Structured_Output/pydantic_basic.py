from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "John Doe"  # Default value for name
    age: Optional[int] = None  # Age must be greater than 0
    # email: EmailStr  # Email must be a valid email address
    cgpa: float = Field(gt=0.0, le=10.0)  # CGPA must be between 0.0 and 4.0


# new_Student = {"name": "John Doe"}
new_Student = {"age": 30, "cgpa": 3.5}
# new_Student = {
#     "name": 32
# }  # This will raise a validation error because the name should be a string
student = Student(**new_Student)  # Unpack the dictionary into the Student model
print(student)  # name=John Doe   name='John Doe' age=30
print(type(student))  # <class '__main__.Student'>


student_dict = dict(student)

print(student_dict["age"])

student_json = student.model_dump_json()
print(student_json)  # {"name": "John Doe", "age": 30, "cgpa": 3.5}
