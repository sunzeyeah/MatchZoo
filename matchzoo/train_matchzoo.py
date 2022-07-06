"""MatchZoo Model training, evaluation & prediction
Supported models: DRMM, ARCI, MatchPyramid
"""

import os
import ast
import shutil
import logging
import argparse
import time
import numpy as np
import matchzoo as mz

from transformers import BertTokenizer
from keras.optimizers import Adam


logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()

    ## Required Parameters
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="match model to use: arci, drmm, match_pyramid")
    parser.add_argument("--input_dir", default=None, type=str, required=True,
                        help="input data dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="output model dir")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="pretrained model name or path (e.g. bert)")

    ## Optional Parameters
    parser.add_argument("--do_train", action="store_true",
                        help="whether do training or not")
    parser.add_argument("--do_eval", action="store_true",
                        help="whether do evaluation or not")
    parser.add_argument("--do_pred", action="store_true",
                        help="whether do prediction or not")
    parser.add_argument("--n_epoch", default=10, type=int,
                        help="number of training epochs")
    parser.add_argument("--num_workers", default=30, type=int,
                        help="number of workers in training")
    parser.add_argument("--use_multiprocessing", action="store_true",
                        help="whether use multi-processing in training")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="max sequence length")
    parser.add_argument("--num_dup", default=5, type=int,
                        help="number of duplicates for each positive sample")
    parser.add_argument("--num_neg", default=1, type=int,
                        help="number of negative samples for one positive sample")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="train data batch size")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="evaluation data batch size")
    parser.add_argument("--embedding_dim", default=768, type=int,
                        help="word/character embedding dimension")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="drop out rate")
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="do lower case in tokenization")
    parser.add_argument("--mask_whole_word", default=True, type=bool,
                        help="mask whole word in tokenization")
    # optimization
    parser.add_argument("--optimizer", default="adam", type=str,
                        help="optimizer name")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="learning rate")
    # arci & arcii
    parser.add_argument("--num_blocks", default=1, type=int,
                        help="number of blocks")
    parser.add_argument("--num_filters", default="[128]", type=str,
                        help="number of filters")
    parser.add_argument("--kernel_size", default="[3]", type=str,
                        help="kernel size; ARCI: [3], MatchPyramid: [[3,3], [3,3]]")
    parser.add_argument("--pool_size", default="[4]", type=str,
                        help="pool size")
    parser.add_argument("--conv_activation_fn", default="relu", type=str,
                        help="CNN activation function")
    parser.add_argument("--mlp_activation_fn", default="relu", type=str,
                        help="MLP activation function")
    parser.add_argument("--mlp_num_layers", default=1, type=int,
                        help="number of MLP layers")
    parser.add_argument("--mlp_hidden_size", default=128, type=int,
                        help="MLP layer hidden size")
    parser.add_argument("--mlp_num_fan_out", default=1, type=int,
                        help="MLP layer number of fan out")
    parser.add_argument("--bin_size", default=30, type=int,
                        help="DRMM matching histogram bin size")
    parser.add_argument("--hist_mode", default="CH", type=str,
                        help="DRMM histogram mode: CH, LCH, NH")
    parser.add_argument("--embedding_trainable", default=True, type=bool,
                        help="MatchPyramid whether embedding is trainable")
    parser.add_argument("--kernel_count", default="16,32", type=str,
                        help="MatchPyramid kernel count for each CNN layer")
    parser.add_argument("--dpool_size", default="3,10", type=str,
                        help="MatchPyramid dynamic pooling size for each CNN layer")


    args = parser.parse_args()

    return args


def get_generator(args, data_pack, embedding_matrix, stage):
    shuffle = True if stage == "train" else False
    batch_size = args.train_batch_size if stage == "train" else args.eval_batch_size
    generator = mz.DataGenerator(data_pack, batch_size=batch_size, shuffle=shuffle)
    # if args.model == "arci":
    #     generator = mz.PairDataGenerator(data_pack, num_dup=args.num_dup, num_neg=args.num_neg,
    #                                    batch_size=args.train_batch_size)
    # elif args.model == "drmm":
    #     generator = mz.HistogramPairDataGenerator(data_pack, embedding_matrix, args.bin_size, hist_mode=args.hist_mode,
    #                                             num_dup=args.num_dup, num_neg=args.num_neg, batch_size=args.train_batch_size)
    # elif args.model == "match_pyramid":
    #     generator = mz.DPoolPairDataGenerator(data_pack,
    #                                         fixed_length_left=args.max_seq_len, fixed_length_right=args.max_seq_len,
    #                                         num_dup=args.num_dup, num_neg=args.num_neg, batch_size=args.train_batch_size)
    # else:
    #     raise ValueError("Unsupported model names: {}".format(args.model))

    return generator


def main():
    args = get_parser()

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.mask_whole_word:
        tokenizer.do_basic_tokenize = False

    mz.USER_DATA_DIR = args.input_dir

    # load data & preprocess
    preprocessor = mz.preprocessors.BertPreprocessor(tokenizer, fixed_length_left=args.max_seq_len,
                                                     fixed_length_right=args.max_seq_len, remove_stop_words=False)
    if args.do_train:
        train_pack = mz.datasets.ccks2022_task9.load_data(args.input_dir, 'train')
        train_pack_processed = preprocessor.fit_transform(train_pack, verbose=0)
        if args.optimizer == "adam":
            optimizer = Adam(lr=args.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {args.optimizer}")
    if args.do_eval:
        valid_pack = mz.datasets.ccks2022_task9.load_data(args.input_dir, 'valid')
        valid_pack_processed = preprocessor.transform(valid_pack, verbose=0)
    if args.do_pred:
        predict_pack = mz.datasets.ccks2022_task9.load_data(args.input_dir, 'test')
        predict_pack_processed = preprocessor.transform(predict_pack, verbose=0)

    # rank task & eval metric
    task = mz.tasks.Classification(num_classes=2)
    task.metrics = [
        mz.metrics.Precision(),
        mz.metrics.Recall(),
        mz.metrics.F1()
    ]
    # if args.model == "drmm":
    #     ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=args.num_neg))
    # else:
    #     ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
    # ranking_task.metrics = [
    #     mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
    #     mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    #     mz.metrics.MeanAveragePrecision(),
    # ]

    # model param
    if args.model == "arci":
        model = mz.models.ArcI(optimizer=optimizer)
        model.params['input_shapes'] = preprocessor.context['input_shapes']
        model.params['task'] = task
        model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
        model.params['embedding_output_dim'] = args.embedding_dim
        # sanity check
        num_filters = ast.literal_eval(args.num_filters)
        pool_size = ast.literal_eval(args.pool_size)
        kernel_size = ast.literal_eval(args.kernel_size)
        assert args.num_blocks == len(num_filters)
        assert args.num_blocks == len(pool_size)
        assert args.num_blocks == len(kernel_size)
        model.params['num_blocks'] = args.num_blocks
        model.params['left_filters'] = num_filters
        model.params['left_kernel_sizes'] = kernel_size
        model.params['left_pool_sizes'] = pool_size
        model.params['right_filters'] = num_filters
        model.params['right_kernel_sizes'] = kernel_size
        model.params['right_pool_sizes'] = pool_size
        model.params['conv_activation_func'] = args.conv_activation_fn
        model.params['mlp_num_layers'] = args.mlp_num_layers
        model.params['mlp_num_units'] = args.mlp_hidden_size
        model.params['mlp_num_fan_out'] = args.mlp_num_fan_out
        model.params['mlp_activation_func'] = args.mlp_activation_fn # "relu"
        model.params['dropout_rate'] = args.dropout
    elif args.model == "drmm":
        model = mz.models.DRMM(optimizer=optimizer)
        model.params['input_shapes'] = [[args.max_seq_len, ], [args.max_seq_len, args.bin_size, ]]
        model.params['task'] = task
        model.params['mask_value'] = 0
        model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
        model.params['embedding_output_dim'] = args.embedding_dim
        model.params['mlp_num_layers'] = args.mlp_num_layers
        model.params['mlp_num_units'] = args.mlp_hidden_size
        model.params['mlp_num_fan_out'] = args.mlp_num_fan_out
        model.params['mlp_activation_func'] = args.mlp_activation_fn # 'tanh'
    elif args.model == "match_pyramid":
        model = mz.models.MatchPyramid(optimizer=optimizer)
        model.params['input_shapes'] = preprocessor.context['input_shapes']
        model.params['task'] = task
        model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
        model.params['embedding_output_dim'] = args.embedding_dim
        model.params['embedding_trainable'] = args.embedding_trainable
        kernel_size = ast.literal_eval(args.kernel_count) # [[3, 3], [3, 3]]
        kernel_count = args.kernel_count.split(",") # [16, 32]
        dpool_size = args.dpool_size.split(",") # [3, 10]
        # sanity check
        assert args.num_block == len(kernel_size)
        assert args.num_block == len(kernel_count)
        assert args.num_block == len(dpool_size)
        model.params['num_blocks'] = args.num_blocks
        model.params['kernel_count'] = kernel_count
        model.params['kernel_size'] = kernel_size
        model.params['dpool_size'] = dpool_size
        model.params['dropout_rate'] = args.dropout
    else:
        raise ValueError("Unsupported model names: {}".format(args.model))

    model.guess_and_fill_missing_params()
    model.build()
    model.compile()
    # model.backend.summary()

    # load embedding
    start_time = time.time()
    bert_embedding = mz.datasets.embeddings.load_bert_embedding(args.model_name_or_path)
    embedding_matrix = bert_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
    if args.model == "drmm":
        # normalize the word embedding for fast histogram generating.
        l2_norm = np.sqrt((embedding_matrix*embedding_matrix).sum(axis=1))
        embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
    model.load_embedding_matrix(embedding_matrix)
    logger.info("Finished loading embedding, time taken: {}s".format(time.time()-start_time))

    # evaluate
    evaluate = None
    if args.do_eval:
        if args.model == "arci":
            dev_x, dev_y = valid_pack_processed[:].unpack()
        else:
            try:
                dev_generator = get_generator(args, valid_pack_processed, embedding_matrix, stage="valid")
            except ValueError as ve:
                logger.error("Error in get generator!", ve)
            dev_x, dev_y = dev_generator[:]
        evaluate = mz.callbacks.EvaluateAllMetrics(model, x=dev_x, y=dev_y, batch_size=args.eval_batch_size,
                                                   model_save_path=args.output_dir.format(model=args.model))
    # train
    if args.do_train:
        # remove previous model directory
        output_dir = args.output_dir.format(model=args.model)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        try:
            train_generator = get_generator(args, train_pack_processed, embedding_matrix, stage="train")
        except ValueError as ve:
            logger.error("Error in get generator!", ve)

        history = model.fit_generator(train_generator, epochs=args.n_epoch, callbacks=[evaluate],
                                      workers=args.num_workers, use_multiprocessing=args.use_multiprocessing)
        # # save model
        # model.save(output_dir)

    if args.do_pred:
        try:
            test_generator = get_generator(args, predict_pack_processed, embedding_matrix, stage="test")
        except ValueError as ve:
            logger.error("Error in get generator!", ve)

        test_x, test_y = test_generator[:]
        prediction = model.predict(test_x)


if __name__ == "__main__":
    main()
