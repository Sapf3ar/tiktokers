class GetVideoCaptureError(Exception):
    """
    Exception raised when there is an error in getting video capture from the given path.
    """

    def __init__(self):
        super().__init__("Error occurred when getting video capture from given path")


class YieldVideoCaptureError(Exception):
    """
    Exception raised when there is an error in yielding video capture.
    """

    def __init__(self):
        super().__init__("Error occurred when yielding video capture")


class OCRException(Exception):
    """
    Base class for exceptions in this module. Catching 'OCRException' will catch all exceptions
    derived from it, that are defined in this module.
    """


class GetFramesError(OCRException):
    """
    Exception raised when there is an error in getting frames from the video capture.
    """

    def __init__(self):
        super().__init__("Can't get frames from given capture")


class ReadTextFromFramesError(OCRException):
    """
    Exception raised when there is an error in reading text from the frames.
    """

    def __init__(self):
        super().__init__("Error occurred while reading text from frames")


class FilterTextsFromResultError(OCRException):
    """
    Exception raised when there is an error in filtering the recognized texts.
    """

    def __init__(self):
        super().__init__("Error occurred when filtering recognized text")
