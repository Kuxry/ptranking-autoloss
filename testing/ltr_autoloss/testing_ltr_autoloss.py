# coding=utf-8

import argparse
import torch
import os
import sys
import numpy as np


import logging
from torch.optim.lr_scheduler import CosineAnnealingLR



from ptranking.ltr_autoloss.utils.global_p import *
from ptranking.ltr_autoloss import BaseRunner
#只是调用了这个文件，还需调用文件下的这个构造函数

from ptranking.ltr_autoloss.controller import Controller
from ptranking.ltr_autoloss.loss_formula import LossFormula
from ptranking.base import loss_ranker
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator

def main():
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--gpu', type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO, help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='./log/log_1.txt', help='Logging file path')
    parser.add_argument('--result_file', type=str, default='../result/result.npy', help='Result file path')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed of numpy and pytorch')
    parser.add_argument('--model_name', type=str, default='BiasedMF', help='Choose model to run.')
    # parser.add_argument('--model_path', type=str, help='Model save path.',
    #                     default=os.path.join(MODEL_DIR, 'biasedMF.pt'))  # '%s/%s.pt' % (model_name, model_name)))
    parser.add_argument('--controller_model_path', type=str, help='Controller Model save path.',
                        default=os.path.join(MODEL_DIR, 'controller.pt'))
    parser.add_argument('--shared_cnn_model_path', type=str, help='Shared CNN Model save path.',
                        default=os.path.join(MODEL_DIR, 'loss_formula.pt'))
    parser.add_argument('--formula_path', type=str, help='Loss Formula save path.',
                        default=os.path.join(MODEL_DIR, 'Formula.txt'))
    parser.add_argument('--child_num_layers', type=int, default=12)
    parser.add_argument('--child_num_branches', type=int, default=8)  # different layers
    parser.add_argument('--child_out_filters', type=int, default=36)
    parser.add_argument('--sample_branch_id', action='store_true')
    parser.add_argument('--sample_skip_id', action='store_true')
    parser.add_argument('--search_loss', action='store_true', help="To search a loss or verify a loss")
    parser.add_argument('--train_with_optim', action='store_true')
    parser.add_argument('--child_grad_bound', type=float, default=5.0)
    parser.add_argument('--smooth_coef', type=float, default=1e-6)
    parser.add_argument('--layers', type=str, default='[64, 16]',
                        help="Size of each layer. (For Deep RS Model.)")
    parser.add_argument('--loss_func', type=str, default='BCE',
                        help='Loss Function. Choose from ["BCE", "MSE", "Hinge", "Focal", "MaxR", "SumR", "LogMin"]')

    parser = BaseRunner.baserunner.parse_runner_args(parser)
    parser = Controller.parse_Ctrl_args(parser)
    parser = LossFormula.parse_Formula_args(parser)
    args, extras = parser.parse_known_args()

    # ======================

    """
    >>> Learning-to-Rank Models <<<
    (1) Optimization based on Empirical Risk Minimization
    -----------------------------------------------------------------------------------------
    | Pointwise | RankMSE                                                                   |
    -----------------------------------------------------------------------------------------
    | Pairwise  | RankNet                                                                   |
    -----------------------------------------------------------------------------------------
    | Listwise  | LambdaRank % ListNet % ListMLE % RankCosine %  ApproxNDCG %  WassRank     |
    |           | STListNet  % LambdaLoss                                                   |
    -----------------------------------------------------------------------------------------   


    >>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | LETTOR    | MQ2007_Super %  MQ2008_Super %  MQ2007_Semi %  MQ2008_Semi                |
    -----------------------------------------------------------------------------------------
    | MSLRWEB   | MSLRWEB10K %  MSLRWEB30K                                                  |
    -----------------------------------------------------------------------------------------
    | Yahoo_LTR | Set1 % Set2                                                               |
    -----------------------------------------------------------------------------------------
    | ISTELLA_LTR | Istella_S % Istella % Istella_X                                         |
    -----------------------------------------------------------------------------------------

    """

    cuda = 1  # the gpu id, e.g., 0 or 1, otherwise, set it as None indicating to use cpu

    debug = False  # in a debug mode, we just check whether the model can operate

    config_with_json = True  # specify configuration with json files or not

    reproduce = False  # given pre-trained models, reproduce experiments

    models_to_run = [
        'RankMSE',
        # 'RankNet',
        # 'LambdaRank',
        # 'ListNet',
    ]

    evaluator = LTREvaluator(cuda=cuda)

    if config_with_json:  # specify configuration with json files
        # the directory of json files
        dir_json = '../ltr_adhoc/json'

        # for model_id in models_to_run:
        #     evaluator.run(debug=debug, model_id=model_id, config_with_json=config_with_json, dir_json=dir_json)

    else:  # specify configuration manually
        ''' pointsf | listsf, namely the type of neural scoring function '''
        sf_id = 'pointsf'

        ''' Selected dataset '''
        # data_id = 'Set1'
        # data_id = 'MSLRWEB30K'
        data_id = 'MQ2008_Super'

        ''' By grid_search, we can explore the effects of different hyper-parameters of a model '''
        grid_search = False

        ''' Location of the adopted data '''
        dir_data = '../data/MQ2008/'


        ''' Output directory '''

        dir_output = '../log/'

        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, sf_id=sf_id, grid_search=grid_search,
                          data_id=data_id, dir_data=dir_data, dir_output=dir_output, reproduce=reproduce)


    # ======================

    # testing
    # data_id = 'MQ2008_Super'
    # dir_data = './data/MQ2008/'
    #
    data_id = 'MSLRWEB10K'
    dir_data = './data/MSLR-WEB10K/'
    model_id = 'RankMSE'  # RankMSE, RankNet, LambdaRank
    # all_result = []
    # avg_scores=np.zeros(1)
    for fold in range(2, 3):
        # random seed & gpu
        if args.random_seed == None:
            args.random_seed = np.random.randint(1 << 32)
        torch.manual_seed(args.random_seed)
        print("random seed:", args.random_seed)
        logging.info(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        controller = Controller(search_for=args.search_for,
                                search_whole_channels=True,
                                num_layers=args.child_num_layers + 3,
                                num_branches=args.child_num_branches,
                                out_filters=args.child_out_filters,
                                lstm_size=args.controller_lstm_size,
                                lstm_num_layers=args.controller_lstm_num_layers,
                                tanh_constant=args.controller_tanh_constant,
                                temperature=None,
                                skip_target=args.controller_skip_target,
                                skip_weight=args.controller_skip_weight,
                                entropy_weight=args.controller_entropy_weight,
                                bl_dec=args.controller_bl_dec,
                                num_aggregate=args.controller_num_aggregate,
                                model_path=args.controller_model_path,
                                sample_branch_id=args.sample_branch_id,
                                sample_skip_id=args.sample_skip_id)
        controller = controller.cuda()

        loss_formula = LossFormula(num_layers=args.child_num_layers + 3,
                                   num_branches=args.child_num_branches,
                                   out_filters=args.child_out_filters,
                                   keep_prob=args.child_keep_prob,
                                   model_path=args.shared_cnn_model_path,
                                   epsilon=args.epsilon)
        loss_formula = loss_formula.cuda()

        # https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
        controller_optimizer = torch.optim.Adam(params=controller.parameters(),
                                                lr=args.controller_lr,
                                                betas=(0.0, 0.999),
                                                eps=1e-3)
        # logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=args.log_file, level=args.verbose)
        logging.info(vars(args))

        # from ltr import DataProcessor_epoch
        # train_data, test_data, validation_data = DataProcessor_epoch("MSLRWEB10K", "./data/MSLR-WEB10K/", 2)

        ###保存模型参数

        ltr = loss_ranker(model_path="./model/ltr.pt")
        # mse_ranker=evaluation_ltr(data_id=data_id, dir_data=dir_data, model_id=model_id, batch_size=100,fold=fold)
        #
        #
        # mse_ranker.save_model(model_path="./model/ltr.pt")

        ###
        #model = NeuralRanker()

        #ltr.load_model(model_path="./model/ltr.pt")

        # list(ltr.named_parameters())
        # list(mse_ranker.named_parameters())[]


        model = ltr

        #model.init() #不初始效果好



        # use gpu
        if torch.cuda.device_count() > 0:
            # model = model.to('cuda:0')
            model = model.cuda()

        runner = BaseRunner.baserunner(optimizer=args.optimizer, learning_rate=args.lr, epoch=args.epoch, batch_size=args.batch_size,
						eval_batch_size=args.eval_batch_size, dropout=args.dropout, l2=args.l2, metrics=args.metric,
						check_epoch=args.check_epoch, early_stop=args.early_stop,
						loss_formula=loss_formula,controller=controller, controller_optimizer=controller_optimizer, args=args)


        runner.fold_train(model, search_loss=True, data_id=data_id, dir_data=dir_data, model_id=model_id, fold=fold)
        #tmp_result = runner.fold_train( data_id=data_id, dir_data=dir_data, model_id=model_id, fold=fold)
        # avg_scores= np.add(avg_scores,tmp_result)

        #all_result.append(tmp_result)
    # avg_scores = np.divide(avg_scores, 5)
    # logging.info(avg_scores)
    # print(avg_scores)




    #
    # controller = Controller(search_for=args.search_for,
    #                         search_whole_channels=True,
    #                         num_layers=args.child_num_layers + 3,
    #                         num_branches=args.child_num_branches,
    #                         out_filters=args.child_out_filters,
    #                         lstm_size=args.controller_lstm_size,
    #                         lstm_num_layers=args.controller_lstm_num_layers,
    #                         tanh_constant=args.controller_tanh_constant,
    #                         temperature=None,
    #                         skip_target=args.controller_skip_target,
    #                         skip_weight=args.controller_skip_weight,
    #                         entropy_weight=args.controller_entropy_weight,
    #                         bl_dec=args.controller_bl_dec,
    #                         num_aggregate=args.controller_num_aggregate,
    #                         model_path=args.controller_model_path,
    #                         sample_branch_id=args.sample_branch_id,
    #                         sample_skip_id=args.sample_skip_id)
    # controller = controller.cuda()
    #
    # loss_formula = LossFormula(num_layers=args.child_num_layers + 3,
    #                            num_branches=args.child_num_branches,
    #                            out_filters=args.child_out_filters,
    #                            keep_prob=args.child_keep_prob,
    #                            model_path=args.shared_cnn_model_path,
    #                            epsilon=args.epsilon)
    # loss_formula = loss_formula.cuda()
    #
    # # https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
    # controller_optimizer = torch.optim.Adam(params=controller.parameters(),
    #                                         lr=args.controller_lr,
    #                                         betas=(0.0, 0.999),
    #                                         eps=1e-3)
    # # logging
    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)
    # logging.basicConfig(filename=args.log_file, level=args.verbose)
    # logging.info(vars(args))
    #
    # runner = BaseRunner.baserunner(optimizer=args.optimizer, learning_rate=args.lr, epoch=args.epoch, batch_size=args.batch_size,
    #                 eval_batch_size=args.eval_batch_size, dropout=args.dropout, l2=args.l2, metrics=args.metric,
    #                 check_epoch=args.check_epoch, early_stop=args.early_stop,
    #                 loss_formula=loss_formula,controller=controller, controller_optimizer=controller_optimizer, args=args)
    # runner.train(search_loss=True, data_id=data_id, dir_data=dir_data, model_id=model_id)


if __name__ == '__main__':


    main()
