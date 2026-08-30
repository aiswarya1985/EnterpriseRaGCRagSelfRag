import datetime

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel



from app.config import settings

'''HTTPBearer() automatically 
extracts and validates the Authorization: 
Bearer <token> header from incoming requests, 
returning a 401 Unauthorized error if it is missing or invalid. 
It also enables the interactive Authorize lock 
button in FastAPI's /docs page to easily test protected endpoints.'''
security = HTTPBearer()

''' user name should contain value, else error. is_admin defaults to false
if not provided '''
class User(BaseModel):
    username: str
    is_admin: bool = False

'''bcrypt.gensalt(rounds=12) generates a unique random string (salt) to make passwords unguessable even if two users choose the same password. The rounds=12 setting tells the algorithm to run $2^{12}$ (4,096) computation loops, intentionally slowing down hackers trying to brute-force passwords.'''
def hash_password(password:str)->str:
     return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

'''It hashes the incoming plain_password using the same salt and settings extracted from hashed_password, then checks if the newly generated hash matches the stored hash.'''
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

'''jwt.encode encodes into 3 parts one long encoded string header:{"alg": "HS256", "typ": "JWT"}
payload:user, admin, token expiry
signature:Hash of Header + Payload signed using JWT_SECRET
enhanced security
'''
def create_access_token(
    username: str,
    expires_delta_seconds: int | None = None,
    is_admin: bool = False,
) -> str:
    if expires_delta_seconds is None:
        expires_delta_seconds = settings.jwt_expiration_minutes * 60
        expire = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
            seconds=expires_delta_seconds
        )

    payload = {
        "sub": username,
        "exp": expire,
        "iat": datetime.datetime.now(datetime.UTC),
        "is_admin": is_admin,
    }

    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")

'''Automatically inspects the credentials if header has valid token
credentials: HTTPAuthorizationCredentials = Depends(security),
jwt.decode : get the username and others from payload
finally returns username and is_admin flag as class User'''

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=["HS256"],
        )
        username = payload.get("sub")
        if not isinstance(username, str) or username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )
        is_admin: bool = payload.get("is_admin", False)
        return User(username=username, is_admin=is_admin)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        ) from None
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        ) from None

'''Depends(get_current_user) should be run successfully before require_admin'''
def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user    