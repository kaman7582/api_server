import sys
from cx_Freeze import setup, Executable
 
#'uvicorn','starlette','click','h11','fastapi','anyio'
build_exe_options = {'packages': ['uvicorn','starlette','click','h11','fastapi','anyio','pydantic','sklearn','keras','numpy','flask'], 
                     'excludes': []
                     }
base = None
if sys.platform == 'win32':
  base = 'Win32GUI'
 
setup(  name = 'runFastApi',
        version = '0.1',
        description = 'AI server',
        options = {'build_exe': build_exe_options},
        executables = [Executable('web_server.py')])
