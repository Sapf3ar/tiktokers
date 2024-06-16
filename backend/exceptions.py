class GetVideoCaptureError(Exception):
    def __init__(self):
        super().__init__("Error occurred when getting video capture from given path")


class YieldVideoCaptureError(Exception):
    def __init__(self):
        super().__init__("Error occurred when yielding video capture")


class OCRException(Exception): ...


class GetFramesError(OCRException):
    def __init__(self):
        super().__init__("Can't get frames from given capture")


class ReadTextFromFramesError(OCRException):
    def __init__(self):
        super().__init__("Error occurred while reading text from frames")


class FilterTextsFromResultError(OCRException):
    def __init__(self):
        super().__init__("Error occurred when filtering recognized text")
