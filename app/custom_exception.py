import traceback # for tracking error tracback library
import sys


class CustomException(Exception): # inherit from Exception bcoz we need predefined exceptions too

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message) # super - inheritance - error exist in predefined exceptions show that else ...
        self.error_message = self.get_detailed_error_message(error_message,error_detail)

    @staticmethod
    def get_detailed_error_message(error_message , error_detail:sys):

        _, _, exc_tb = traceback.sys.exc_info() # last one exc_tb is traceback
        file_name = exc_tb.tb_frame.f_code.co_filename # will show the filename in which error occured
        line_number = exc_tb.tb_lineno

        return f"Error in {file_name} , line {line_number} : {error_message}"
    
    def __str__(self): # text representation of error message
        return self.error_message

