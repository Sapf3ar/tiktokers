# OCR

## Getting started

Install dependencies:

`pip install -r -requirements.txt`

To get recognized text from the video:

```python
from text_extractor import TextExtractor

text_extractor = TextExtractor()
video_link = <link_to_the_video>
texts = text_extractor.extract_text(video_link)

unique_texts = list(dict.fromkeys(texts))
video_text = " ".join(unique_texts)
```

Parameters of `extract_text` method:
- **video_link**: link to the video to get the text from
- **sample_rate**: sampling rate when receiving a frame from a video
- **confidence_threshold**: the threshold for filtering texts in which the model is not confident enough
- **max_image_size**: the maximum image size of the larger side to resize the image
- **max_frames**: the maximum number of frames captured from a video
- **batch_size**: batch size for OCR model