import sys

def get_error_details(message) -> str:
    _,_,exc_tb = sys.exc_info()
    if exc_tb:
        fileName = exc_tb.tb_frame.f_code.co_filename
        lineNumber = exc_tb.tb_lineno
        error_message = f"The error has occured in file - {fileName} on line number - {lineNumber}. ERROR IS - {message}"
        return error_message
    else:
        return message


class CustomException(Exception):
    def __init__(self,message):
        super().__init__(message)
        self.error_message = get_error_details(message=message)

    def __str__(self) -> str:
         return self.error_message
    

