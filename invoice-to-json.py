import os
import pandas as pd
from PIL import Image
from datasets import DatasetInfo, Features, Value, Image as ImageFeature
from datasets.tasks import QuestionAnswering

class DocutorAIConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

def _info():
    return DatasetInfo(
        description="Document QA dataset for information extraction",
        features=Features({
            'id': Value('string'),
            'question': Value('string'),
            'answer': Value('string'),
            'image_path': Value('string'),
            'image_id': Value('string'),
            'image': ImageFeature()
        }),
        supervised_keys=None,
        homepage="",
        license="mit",
        citation="",
        task_templates=[
            QuestionAnswering(
                question_column="question",
                answers_column="answer",
                context_column="image"
            )
        ]
    )

def _split_generators(dl_manager):
    """Returns SplitGenerators."""
    return [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                'filepath': 'train.tsv',
                'images_dir': 'images'
            }
        )
    ]

def _generate_examples(filepath, images_dir):
    """Yields examples."""
    df = pd.read_csv(filepath, sep='\t')
    for idx, row in df.iterrows():
        # Construct full image path
        image_path = os.path.join(images_dir, row['image_path'])
        
        # Load image using PIL
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')  # Convert to RGB format
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
            
        yield idx, {
            'id': row['id'],
            'question': row['question'],
            'answer': row['answer'],
            'image_id': row['image_id'],
            'image_path': row['image_path'],
            'image': image  # This will be automatically converted to the correct format by datasets
        }