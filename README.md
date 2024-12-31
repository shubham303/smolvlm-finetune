---
language:
  - en
license: mit
task_categories:
  - document-question-answering
pretty_name: Invoice-to-Json
tags:
  - document-ai
  - document-understanding
  - visual-question-answering
size_categories:
  - 1K<n<10K
---

# Invoice-to-Json Dataset

## Dataset Description

### Dataset Summary

Invoice-to-Json is a dataset designed for document understanding and information extraction tasks. It consists of document images paired with questions and answers, specifically focused on extracting structured information (JSON format) from documents.

### Supported Tasks

- **Document Question Answering**: The dataset supports training models to answer questions about document content
- **Information Extraction**: Models can be trained to extract structured data in JSON format from documents
- **Document Understanding**: The dataset can be used to develop models that comprehend document layout and content

### Languages

The dataset contains English text only.

### Dataset Structure

The dataset contains:
- Document images
- Associated questions about the document content
- Ground truth answers in structured format
- Unique identifiers for both questions and images

#### Data Instances

Each instance in the dataset contains:
```python
{
    'id': 'string',           # Unique identifier for the QA pair
    'question': 'string',     # Question about the document
    'answer': 'string',       # Answer in structured format
    'image_path': 'string',   # Path to the associated image
    'image_id': 'string'      # Unique identifier for the image
}
```

#### Data Fields

- `id`: Unique identifier for each question-answer pair
- `question`: The question asking for specific information from the document
- `answer`: The ground truth answer, typically in JSON format
- `image_path`: Path to the associated document image
- `image_id`: Unique identifier for the document image

#### Data Splits

The dataset is provided with a training split.

### Data Collection and Annotation

[To be filled: Please provide information about how the data was collected and annotated]

### Considerations for Using the Data

#### Social Impact of Dataset

This dataset aims to improve document understanding and information extraction systems, which can:
- Enhance automation of document processing
- Reduce manual data entry errors
- Improve accessibility of document content
- Speed up document processing workflows

#### Discussion of Biases

[To be filled: Please provide information about any potential biases in the dataset]

### Citation Information

If you use this dataset, please cite:

```
@misc{Invoice-to-Json,
  title={Invoice-to-Json: A Document Understanding and Information Extraction Dataset},
  year={2024}
}
```

### Licensing Information

This dataset is released under the MIT License.

### Contributions

Thanks to all the contributors who participated in creating and annotating this dataset!

For more information or to contribute, please visit the dataset repository.