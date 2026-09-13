from typing import Optional

from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    account_type: str
    name: str
    email: EmailStr
    password: str
    department_code: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    account_type: str
    department_id: Optional[int] = None

    model_config = {
        "from_attributes": True
    }