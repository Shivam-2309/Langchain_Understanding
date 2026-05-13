from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class Student(BaseModel):
    name: str
    
    age: Optional[int] = Field(
        default=None,
        description="The age of the student",
        json_schema_extra={"example": 25}
    )

    email: EmailStr

    cgpa: float = Field(
        gt=0,
        lt=10,
        description="The CGPA of the student",
        json_schema_extra={"example": 9.5}
    )

student = {
    "name": "Shivam",
    "age": 25,
    "email": "shivam@example.com",
    "cgpa": 9.4
}

new_student = Student(**student)

print(new_student)