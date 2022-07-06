"""Toy data loader."""

import os
import typing
import csv
import jieba
import pandas as pd

import matchzoo


def load_data(data_root, stage: str = 'train', task: str = 'classification'):# \
        # -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load Toy data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filter: Whether remove the questions without correct answers.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'valid', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    # file_path = os.path.join(data_root, f'finetune_train_{stage}.tsv')
    file_path = os.path.join(data_root, f'finetune_{stage}_100.tsv')
    data_pack = _read_data(file_path)
    # if filter and stage in ('dev', 'test'):
    #     ref_path = data_root.joinpath(f'WikiQA-{stage}.ref')
    #     filter_ref_path = data_root.joinpath(f'WikiQA-{stage}-filtered.ref')
    #     with open(filter_ref_path, mode='r') as f:
    #         filtered_ids = set([line.split()[0] for line in f])
    #     filtered_lines = []
    #     with open(ref_path, mode='r') as f:
    #         for idx, line in enumerate(f.readlines()):
    #             if line.split()[0] in filtered_ids:
    #                 filtered_lines.append(idx)
    #     data_pack = data_pack[filtered_lines]

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        data_pack.one_hot_encode_label(task.num_classes, inplace=True)
        return data_pack, [False, True]
    else:
        raise ValueError(f"{task} is not a valid task.")


def _read_data(path):
    table = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE,
                        names=['label', 'src_id', 'src_title', 'src_pvs', 'tgt_id', 'tgt_title', 'tgt_pvs'])

    table['src_text'] = table.apply(lambda x: x['src_title'] + " [SEP] " + " ".join(jieba.cut(x['src_pvs'])), axis=1)
    table['tgt_text'] = table.apply(lambda x: x['tgt_title'] + " [SEP] " + " ".join(jieba.cut(x['tgt_pvs'])), axis=1)

    df = pd.DataFrame({
        'text_left': table['src_text'],
        'text_right': table['tgt_text'],
        'id_left': table['src_id'],
        'id_right': table['tgt_id'],
        'label': table['label']
    })

    return matchzoo.pack(df)
