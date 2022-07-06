"""Embedding data loader."""
import os
import matchzoo as mz


def load_bert_embedding(input_dir: str):# -> mz.embedding.Embedding:
    """
    Return the pretrained bert embedding.

    :return: The :class:`mz.embedding.Embedding` object.
    """
    file_name = 'embedding_roberta_chinese.out'
    file_path = os.path.join(input_dir, file_name)

    return mz.embedding.load_from_file(file_path=file_path, mode='bert')
