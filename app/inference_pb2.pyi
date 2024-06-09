from google.protobuf import message as _message
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceRequest(_message.Message):
    __slots__ = ["audio", "user_id"]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    def __init__(self, audio: _Optional[bytes] = ...) -> None: ...

class InferenceReply(_message.Message):
    __slots__ = ["pred"]
    PRED_FIELD_NUMBER: _ClassVar[int]
    pred: str
    def __init__(self, pred: _Optional[str] = ...) -> None: ...
