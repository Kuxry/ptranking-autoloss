# coding=utf-8

import torch
import logging
from time import time

import torch.nn.functional as F
from ptranking.ltr_autoloss.utils import utils, global_p

from tqdm import tqdm
import numpy as np
import os
import copy
import datetime

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, accuracy_score

import ptranking.ltr_autoloss.controller
import ptranking.ltr_autoloss.loss_formula


from ptranking.ltr_adhoc.eval.ltr import LTREvaluator
from torch.utils.tensorboard import SummaryWriter


# count=0



class baserunner():
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=1e-4,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--optimizer', type=str, default='GD',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metric', type=str, default="AUC",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--skip_eval', type=int, default=0,
                            help='number of epochs without evaluation')
        parser.add_argument('--skip_rate', type=float, default=1.005, help='bad loss skip rate')
        parser.add_argument('--rej_rate', type=float, default=1.005, help='bad training reject rate')
        parser.add_argument('--skip_lim', type=float, default=1e-5, help='bad loss skip limit')
        parser.add_argument('--rej_lim', type=float, default=1e-5, help='bad training reject limit')
        parser.add_argument('--lower_bound_zero_gradient', type=float, default=1e-4,
                            help='bound to check zero gradient')
        parser.add_argument('--search_train_epoch', type=int, default=1,
                            help='epoch num for training when searching loss')
        parser.add_argument('--step_train_epoch', type=int, default=1, help='epoch num for training each step')

        return parser

    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='AUC,RMSE', check_epoch=10, early_stop=1, controller=None,
                 loss_formula=None, controller_optimizer=None, args=None, gpu=True, device="cuda:0"):
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2

        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        self.train_results, self.valid_results, self.test_results = [], [], []

        self.controller = controller
        self.loss_formula = loss_formula
        self.controller_optimizer = controller_optimizer
        self.args = args
        self.print_prediction = {}
        self.gpu, self.device = gpu, device

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def eva_termination(self, model):
        """
        检查是否终止训练，基于验证集
        :param model: 模型
        :return: 是否终止训练
        """
        metric = self.metrics[0]
        valid = self.valid_results
        epoch_num =100
        # 如果已经训练超过epoch_num轮，且评价指标越小越好，且评价已经连续10轮非减
        if len(valid) > epoch_num and metric in utils.LOWER_METRIC_LIST and utils.strictly_increasing(valid[-10:]):
            return True
        # 如果已经训练超过epoch_num轮，且评价指标越大越好，且评价已经连续10轮非增
        elif len(valid) > epoch_num and metric not in utils.LOWER_METRIC_LIST and utils.strictly_decreasing(valid[-10:]):
            return True

        # 训练好结果离当前已经epoch_num轮以上了
        elif len(valid) - valid.index(utils.best_result(metric, valid)) > epoch_num:
            return True
        return False

    def fit(self, model, train_data, loss_fun=None, sample_arc=None, regularizer=True):

        accumulate_size=0

        # writer = SummaryWriter('./data/tensorboard')
        model.train_mode()
        # sum_loss=[]
        # global count
        # avg_loss=0
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            accumulate_size += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)
            #if self.args.search_loss else tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1)
            model.optimizer.zero_grad()
            batch_preds = model(batch_q_doc_vectors)
            #loss=torch.tensor(0.5)
            label=batch_std_labels
            pred=batch_preds
            smooth_coef=1e-6


            #loss = torch.mean(1 / (((label+smooth_coef + 1) + ((pred+smooth_coef) * (label+smooth_coef))))) 0.47
            #loss =torch.mean((label-pred)**2) 0.53
            loss =torch.mean((torch.log((pred+ 1 / (label+smooth_coef))))**2) #0.51

            loss=loss+ model.l2() * self.l2_weight

            if loss_fun is not None and sample_arc is not None:
                loss = loss_fun(batch_preds, batch_std_labels, sample_arc)
                if regularizer:
                    loss += model.l2() * self.l2_weight
            # avg_loss+=loss
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 50)


            model.optimizer.step()
            # ndcg1=pre_evaluate(model,test_data,False,False)
            # count=count+1
            # writer.add_scalar("ndcg1", ndcg1,count)

        # avg_loss=avg_loss/100
        # sum_loss.append(avg_loss)
        model.eval_mode()

    def fold_train(self, model,search_loss=False, model_id=None, data_id=None, dir_data=None, skip_eval=0, fold=None):
        logging.info(fold)

        # writer = SummaryWriter('./data/tensorboard')

        #model = NeuralRanker()
        #model.init()  # initialize or reset with the same random initialization

        LTRdata = LTREvaluator()

        # 获得训练、验证、测试数据，epoch=-1不shuffle
        train_data, test_data, validation_data = LTRdata.load_data(eval_dict, data_dict, fold)

        self._check_time(start=True)  # 记录初始时间
        #validation_data=train_data
        # 训练之前的模型效果
        init_train = pre_evaluate(model, train_data) \
            if train_data is not None else [-1.0] * len(self.metrics)
        init_valid = pre_evaluate(model, validation_data) \
            if validation_data is not None else [-1.0] * len(self.metrics)
        init_test = pre_evaluate(model, test_data) \
            if test_data is not None else [-1.0] * len(self.metrics)

        # 打印当前时间
        logging.info(datetime.datetime.now())

        logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (
        utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
        self._check_time()) + ','.join(self.metrics))

        min_reward = torch.tensor(-1.0).cuda()
        #train_with_optim = True

        last_search_cnt = self.controller.num_aggregate * self.args.controller_train_steps
        try:
            for epoch in range(self.epoch):
                self._check_time()
                self.loss_formula.eval()
                self.controller.zero_grad()
                if self.args.search_loss:
                    start_auc = pre_evaluate(model, validation_data,False,True,False,False)  #
                    baseline = torch.tensor(start_auc).cuda()
                    cur_model = copy.deepcopy(model)
                    grad_dict = dict()
                    test_pred = torch.rand(20).cuda() * 0.8 + 0.1  # change range here
                    test_label = torch.rand(20).cuda()
                    test_pred.requires_grad = True
                    max_reward = min_reward.clone().detach()
                    best_arc = None

                    for i in tqdm(range(last_search_cnt), leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100,
                                  mininterval=1):
                        while True:
                            reward = None
                            self.controller()  # perform forward pass to generate a new architecture
                            sample_arc = self.controller.sample_arc
                            if test_pred.grad is not None:
                                test_pred.grad.data.zero_()
                            test_loss = self.loss_formula(test_pred, test_label, sample_arc, small_epsilon=True)
                            try:
                                test_loss.backward()
                            except RuntimeError:
                                pass
                            if test_pred.grad is None or torch.norm(test_pred.grad,float('inf')) < self.args.lower_bound_zero_gradient:
                                reward = min_reward.clone().detach()
                            if reward is None:
                                for key, value in grad_dict.items():
                                    if torch.norm(test_pred.grad - key,
                                                  float('inf')) < self.args.lower_bound_zero_gradient:
                                        reward = value.clone().detach()
                                        break
                            if reward is None:#找不到了，使用整体vald上的ndcg作为梯度上的reward

                                model.zero_grad()
                                for j in range(self.args.search_train_epoch):
                                    last_batch = self.fit(model, train_data, loss_fun=self.loss_formula,sample_arc=sample_arc, regularizer=False)
                                reward = torch.tensor(pre_evaluate(model, validation_data, True, True,False,False)).cuda()
                                grad_dict[test_pred.grad.clone().detach()] = reward.clone().detach()#对随机生成的梯度在上面找梯度
                                #环境就是这个dict 把这个环境进行扩充 评分是用ndcg
                                model = copy.deepcopy(cur_model)
                            if reward < baseline - self.args.skip_lim:
                                reward = min_reward.clone().detach()
                                reward += self.args.controller_entropy_weight * self.controller.sample_entropy
                            else:
                                if reward > max_reward:
                                    max_reward = reward.clone().detach()
                                    if self.args.train_with_optim:
                                        best_arc = copy.deepcopy(sample_arc)
                                reward += self.args.controller_entropy_weight * self.controller.sample_entropy
                                baseline -= (1 - self.args.controller_bl_dec) * (baseline - reward)
                            baseline = baseline.detach()

                            ctrl_loss = -1 * self.controller.sample_log_prob * (reward - baseline)
                            ctrl_loss /= self.controller.num_aggregate
                            if (i + 1) % self.controller.num_aggregate == 0:
                                ctrl_loss.backward()
                                grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
                                                                           self.args.child_grad_bound)
                                self.controller_optimizer.step()
                                # print("step")
                                self.controller.zero_grad()
                            else:
                                ctrl_loss.backward(retain_graph=True)
                            break
                    self.controller.eval()

                    logging.info(
                        'Best auc during controller train: %.3f; Starting auc: %.3f' % (max_reward.item(), start_auc))
                    last_search_cnt = 0
                    if self.args.train_with_optim and best_arc is not None and max_reward > start_auc - self.args.rej_lim:
                        sample_arc = copy.deepcopy(best_arc)
                        for j in range(self.args.search_train_epoch):
                            last_batch = self.fit(model, train_data, loss_fun=self.loss_formula, sample_arc=sample_arc)
                        new_auc = torch.tensor(pre_evaluate(model, validation_data, True, True,True,False)).cuda()
                        print('Optimal: ',
                              self.loss_formula.log_formula(sample_arc=sample_arc, id=self.loss_formula.num_layers - 1))
                    else:
                        grad_dict = dict()
                        self.controller.zero_grad()
                        while True:
                            with torch.no_grad():
                                self.controller(sampling=True)
                                last_search_cnt += 1
                            sample_arc = self.controller.sample_arc
                            if test_pred.grad is not None:
                                test_pred.grad.data.zero_()
                            test_loss = self.loss_formula(test_pred, test_label, sample_arc, small_epsilon=True)
                            try:
                                test_loss.backward()
                            except RuntimeError:
                                pass
                            if test_pred.grad is None or torch.norm(test_pred.grad,
                                                                    float('inf')) < self.args.lower_bound_zero_gradient:
                                continue
                            dup_flag = False
                            for key in grad_dict.keys():
                                if torch.norm(test_pred.grad - key, float('inf')) < self.args.lower_bound_zero_gradient:
                                    dup_flag = True
                                    break
                            if dup_flag:
                                continue
                            print(self.loss_formula.log_formula(sample_arc=sample_arc,
                                                                id=self.loss_formula.num_layers - 1))
                            grad_dict[test_pred.grad.clone().detach()] = True
                            model = copy.deepcopy(cur_model)
                            model.zero_grad()
                            for j in range(self.args.search_train_epoch):
                                last_batch = self.fit(model, train_data, loss_fun=self.loss_formula,
                                                      sample_arc=sample_arc)
                            new_auc = torch.tensor(pre_evaluate(model, validation_data, True, True,False,True)).cuda()
                            if new_auc > start_auc - self.args.rej_lim:  #
                                break
                            print('Epoch %d: Reject!' % (epoch + 1))

                    last_search_cnt = max(last_search_cnt // 10,
                                          self.controller.num_aggregate * self.args.controller_train_steps)
                    if last_search_cnt % self.controller.num_aggregate != 0:
                        last_search_cnt = (last_search_cnt // self.controller.num_aggregate + 1) * self.controller.num_aggregate
                    logging.info(
                        self.loss_formula.log_formula(sample_arc=sample_arc, id=self.loss_formula.num_layers - 1))
                    self.controller.train()
                else:
                    last_batch = self.fit(model, train_data, loss_fun=None, sample_arc=None)
                training_time = self._check_time()

                if epoch >= skip_eval:
                    metrics = self.metrics[0:1]
                    train_result = pre_evaluate(model, train_data) \
                        if train_data is not None else [-1.0] * len(self.metrics)
                    valid_result = pre_evaluate(model, validation_data) \
                        if validation_data is not None else [-1.0] * len(self.metrics)
                    test_result = pre_evaluate(model, test_data) \
                        if test_data is not None else [-1.0] * len(self.metrics)
                    testing_time = self._check_time()

                    self.train_results.append(train_result)
                    self.valid_results.append(valid_result)
                    self.test_results.append(test_result)

                    # out
                    # writer.add_scalar("train", train_result.item(), epoch)
                    # writer.add_scalar("valida", valid_result.item(), epoch)
                    # writer.add_scalar("test", test_result.item(), epoch)

                    # 输出当前模型效果
                    logging.info("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
                                 % (epoch + 1, training_time, utils.format_metric(train_result),
                                    utils.format_metric(valid_result), utils.format_metric(test_result),
                                    testing_time) + ','.join(self.metrics))

                    if not self.args.search_loss:
                        print("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
                              % (epoch + 1, training_time, utils.format_metric(train_result),
                                 utils.format_metric(valid_result), utils.format_metric(test_result),
                                 testing_time) + ','.join(self.metrics))
                    # 如果当前效果是最优的，保存模型，基于验证集
                    if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                        model.save_model()
                        self.controller.save_model()
                        self.loss_formula.save_model()
                    # 检查是否终止训练，基于验证集
                    if self.args.search_loss == False and self.eva_termination(model) and self.early_stop == 1:
                        logging.info("Early stop at %d based on validation result." % (epoch + 1))
                        break
                if epoch < skip_eval:
                    logging.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))


        except KeyboardInterrupt:
            print("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()
                self.controller.save_model()
                self.loss_formula.save_model()

        # Find the best validation result across iterations
        best_valid_score = utils.best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        logging.info("Best Iter(validation)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        # best_val_result = [self.train_results[best_epoch], self.valid_results[best_epoch], self.test_results[best_epoch]]

        best_test_score = utils.best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        logging.info("Best Iter(test)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))

        best_test_result = [self.train_results[best_epoch], self.valid_results[best_epoch],
                            self.test_results[best_epoch]]
        model.load_model()
        self.controller.load_model()
        self.loss_formula.load_model()
        # writer.close()
        return [best_test_result]


    # def train(self, search_loss=False,model_id=None,data_id=None, dir_data=None,skip_eval=0):
    #
    #     #writer = SummaryWriter('./data/tensorboard')
    #
    #     model=NeuralRanker()
    #     model.init()  # initialize or reset with the same random initialization
    #
    #
    #
    #
    #     # 获得训练、验证、测试数据，epoch=-1不shuffle
    #     #train_data, test_data,validation_data = DataProcessor(data_id, dir_data)
    #     self._check_time(start=True)  # 记录初始时间
    #
    #     # 训练之前的模型效果
    #     init_train = pre_evaluate_fold(model,data_id, dir_data,data_type='train') \
    #         #if train_data is not None else [-1.0] * len(self.metrics)
    #     init_valid = pre_evaluate_fold(model,data_id, dir_data,data_type='valid') \
    #         #if validation_data is not None else [-1.0] * len(self.metrics)
    #     init_test = pre_evaluate_fold(model,data_id, dir_data,data_type='test') \
    #         #if test_data is not None else [-1.0] * len(self.metrics)
    #
    #     # 打印当前时间
    #     logging.info(datetime.datetime.now())
    #
    #     logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
    #         self._check_time()) + ','.join(self.metrics))
    #
    #     min_reward = torch.tensor(-1.0).cuda()
    #     train_with_optim = False
    #
    #     last_search_cnt = self.controller.num_aggregate * self.args.controller_train_steps
    #     try:
    #         for epoch in range(self.epoch):
    #             self.loss_formula.eval()
    #             self.controller.zero_grad()
    #             if search_loss:
    #                 start_auc=pre_evaluate_fold(model,data_id, dir_data,data_type='train')#
    #                 baseline = torch.tensor(start_auc).cuda()
    #                 cur_model= copy.deepcopy(model)
    #                 grad_dict = dict()
    #                 test_pred = torch.rand(20).cuda() * 0.8 + 0.1  # change range here
    #                 test_label = torch.rand(20).cuda()
    #                 test_pred.requires_grad = True
    #                 max_reward = min_reward.clone().detach()
    #                 best_arc = None
    #
    #                 for i in tqdm(range(last_search_cnt), leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100,mininterval=1):
    #                     while True:
    #                         reward = None
    #                         self.controller()  # perform forward pass to generate a new architecture
    #                         sample_arc = self.controller.sample_arc
    #                         if test_pred.grad is not None:
    #                             test_pred.grad.data.zero_()
    #                         test_loss = self.loss_formula(test_pred, test_label, sample_arc, small_epsilon=True)
    #                         try:
    #                             test_loss.backward()
    #                         except RuntimeError:
    #                             pass
    #                         if test_pred.grad is None or torch.norm(test_pred.grad,float('inf')) < self.args.lower_bound_zero_gradient:
    #                             reward = min_reward.clone().detach()
    #                         if reward is None:
    #                             for key, value in grad_dict.items():
    #                                 if torch.norm(test_pred.grad - key, float('inf')) < self.args.lower_bound_zero_gradient:
    #                                     reward = value.clone().detach()
    #                                     break
    #                         if reward is None:
    #                             model.zero_grad()
    #                             for j in range(self.args.search_train_epoch):
    #                                 last_batch = self.fit(model,data_id, dir_data,loss_fun=self.loss_formula, sample_arc=sample_arc, regularizer=False)
    #                             reward = torch.tensor(pre_evaluate_fold(model,data_id, dir_data,data_type='train',reward_check=True,result_check=True)).cuda()
    #                             grad_dict[test_pred.grad.clone().detach()] = reward.clone().detach()
    #                             model = copy.deepcopy(cur_model)
    #                         if reward < baseline - self.args.skip_lim:
    #                             reward = min_reward.clone().detach()
    #                             reward += self.args.controller_entropy_weight * self.controller.sample_entropy
    #                         else:
    #                             if reward > max_reward:
    #                                 max_reward = reward.clone().detach()
    #                                 if self.args.train_with_optim:
    #                                     best_arc = copy.deepcopy(sample_arc)
    #                             reward += self.args.controller_entropy_weight * self.controller.sample_entropy
    #                             baseline -= (1 - self.args.controller_bl_dec) * (baseline - reward)
    #                         baseline = baseline.detach()
    #
    #                         ctrl_loss = -1 * self.controller.sample_log_prob * (reward - baseline)
    #                         ctrl_loss /= self.controller.num_aggregate
    #                         if (i + 1) % self.controller.num_aggregate == 0:
    #                             ctrl_loss.backward()
    #                             grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
    #                                                                        self.args.child_grad_bound)
    #                             self.controller_optimizer.step()
    #                             #print("step")
    #                             self.controller.zero_grad()
    #                         else:
    #                             ctrl_loss.backward(retain_graph=True)
    #                         break
    #                 self.controller.eval()
    #
    #                 logging.info('Best auc during controller train: %.3f; Starting auc: %.3f' % (max_reward.item(), start_auc))
    #                 last_search_cnt = 0
    #                 if self.args.train_with_optim and best_arc is not None and max_reward > start_auc - self.args.rej_lim:
    #                     sample_arc = copy.deepcopy(best_arc)
    #                     for j in range(self.args.search_train_epoch):
    #                         last_batch = self.fit(model,data_id, dir_data,loss_fun=self.loss_formula, sample_arc=sample_arc, regularizer=False)
    #                     new_auc = torch.tensor(pre_evaluate_fold(model,data_id, dir_data,data_type='train',reward_check=True,result_check=True)).cuda()
    #                     print('Optimal: ', self.loss_formula.log_formula(sample_arc=sample_arc, id=self.loss_formula.num_layers - 1))
    #                 else:
    #                     grad_dict = dict()
    #                     self.controller.zero_grad()
    #                     while True:
    #                         with torch.no_grad():
    #                             self.controller(sampling=True)
    #                             last_search_cnt += 1
    #                         sample_arc = self.controller.sample_arc
    #                         if test_pred.grad is not None:
    #                             test_pred.grad.data.zero_()
    #                         test_loss = self.loss_formula(test_pred, test_label, sample_arc, small_epsilon=True)
    #                         try:
    #                             test_loss.backward()
    #                         except RuntimeError:
    #                             pass
    #                         if test_pred.grad is None or torch.norm(test_pred.grad,float('inf')) < self.args.lower_bound_zero_gradient:
    #                             continue
    #                         dup_flag = False
    #                         for key in grad_dict.keys():
    #                             if torch.norm(test_pred.grad - key, float('inf')) < self.args.lower_bound_zero_gradient:
    #                                 dup_flag = True
    #                                 break
    #                         if dup_flag:
    #                             continue
    #                         print(self.loss_formula.log_formula(sample_arc=sample_arc,id=self.loss_formula.num_layers - 1))
    #                         grad_dict[test_pred.grad.clone().detach()] = True
    #                         model = copy.deepcopy(cur_model)
    #                         model.zero_grad()
    #                         for j in range(self.args.search_train_epoch):
    #                             last_batch = self.fit(model,data_id, dir_data,loss_fun=self.loss_formula, sample_arc=sample_arc, regularizer=False)
    #                         new_auc =torch.tensor(pre_evaluate_fold(model,data_id, dir_data,data_type='train',reward_check=True,result_check=True)).cuda()
    #                         if new_auc > start_auc - self.args.rej_lim:#
    #                             break
    #                         print('Epoch %d: Reject!' % (epoch + 1))
    #
    #                 last_search_cnt = max(last_search_cnt // 10,self.controller.num_aggregate * self.args.controller_train_steps)
    #                 if last_search_cnt % self.controller.num_aggregate != 0:
    #                     last_search_cnt = (last_search_cnt // self.controller.num_aggregate + 1) * self.controller.num_aggregate
    #                 logging.info(self.loss_formula.log_formula(sample_arc=sample_arc, id=self.loss_formula.num_layers - 1))
    #                 self.controller.train()
    #             else:
    #                 last_batch = self.fit(model,data_id, dir_data,loss_fun=self.loss_formula, sample_arc=sample_arc, regularizer=False)
    #             training_time = self._check_time()
    #
    #             if epoch >= skip_eval:
    #                 metrics = self.metrics[0:1]
    #                 train_result = pre_evaluate_fold(model,data_id, dir_data,data_type='train') \
    #                     #if train_data is not None else [-1.0] * len(self.metrics)
    #                 valid_result = pre_evaluate_fold(model,data_id, dir_data,data_type='valid') \
    #                     #if validation_data is not None else [-1.0] * len(self.metrics)
    #                 test_result = pre_evaluate_fold(model,data_id, dir_data,data_type='test') \
    #                     #if test_data is not None else [-1.0] * len(self.metrics)
    #                 testing_time = self._check_time()
    #
    #
    #                 self.train_results.append(train_result)
    #                 self.valid_results.append(valid_result)
    #                 self.test_results.append(test_result)
    #
    #
    #                 #out
    #                 # writer.add_scalar("train", train_result.item(), epoch)
    #                 # writer.add_scalar("valida", valid_result.item(), epoch)
    #                 # writer.add_scalar("test", test_result.item(), epoch)
    #
    #                 # 输出当前模型效果
    #                 logging.info("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
    #                              % (epoch + 1, training_time, utils.format_metric(train_result),
    #                                 utils.format_metric(valid_result), utils.format_metric(test_result),
    #                                 testing_time) + ','.join(self.metrics))
    #
    #                 if not self.args.search_loss:
    #                     print("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
    #                           % (epoch + 1, training_time, utils.format_metric(train_result),
    #                              utils.format_metric(valid_result), utils.format_metric(test_result),
    #                              testing_time) + ','.join(self.metrics))
    #                 # 如果当前效果是最优的，保存模型，基于验证集
    #                 if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
    #                     model.save_model()
    #                     self.controller.save_model()
    #                     self.loss_formula.save_model()
    #                 # 检查是否终止训练，基于验证集
    #                 if self.args.search_loss == False and self.eva_termination(model) and self.early_stop == 1:
    #                     logging.info("Early stop at %d based on validation result." % (epoch + 1))
    #                     break
    #             if epoch < skip_eval:
    #                 logging.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))
    #
    #
    #     except KeyboardInterrupt:
    #         print("Early stop manually")
    #         save_here = input("Save here? (1/0) (default 0):")
    #         if str(save_here).lower().startswith('1'):
    #             model.save_model()
    #             self.controller.save_model()
    #             self.loss_formula.save_model()
    #
    #     # Find the best validation result across iterations
    #     best_valid_score = utils.best_result(self.metrics[0], self.valid_results)
    #     best_epoch = self.valid_results.index(best_valid_score)
    #     logging.info("Best Iter(validation)= %5d\t train= %s valid= %s test= %s [%.1f s] "
    #                  % (best_epoch + 1,
    #                     utils.format_metric(self.train_results[best_epoch]),
    #                     utils.format_metric(self.valid_results[best_epoch]),
    #                     utils.format_metric(self.test_results[best_epoch]),
    #                     self.time[1] - self.time[0]) + ','.join(self.metrics))
    #     best_test_score = utils.best_result(self.metrics[0], self.test_results)
    #     best_epoch = self.test_results.index(best_test_score)
    #     logging.info("Best Iter(test)= %5d\t train= %s valid= %s test= %s [%.1f s] "
    #                  % (best_epoch + 1,
    #                     utils.format_metric(self.train_results[best_epoch]),
    #                     utils.format_metric(self.valid_results[best_epoch]),
    #                     utils.format_metric(self.test_results[best_epoch]),
    #                     self.time[1] - self.time[0]) + ','.join(self.metrics))
    #     model.load_model()
    #     self.controller.load_model()
    #     self.loss_formula.load_model()

def pre_evaluate(model, data, reward_check=False, result_check=False, loop_check=False, loop_check2=False):
    ranker = model
    fold_num = 5
    cutoffs = [1, 5, 10]
    l2r_cv_avg_scores = np.zeros(len(cutoffs))  # fold average

    # presort
    torch_fold_ndcg_ks = ranker.ndcg_at_ks(test_data=data, ks=cutoffs, presort=True, device='cpu')
    # torch_fold_ndcg_ks=torch_fold_ndcg_ks.cpu()
    fold_ndcg_ks = torch_fold_ndcg_ks.data.numpy()
    l2r_cv_avg_scores = np.add(l2r_cv_avg_scores, fold_ndcg_ks)  # sum for later cv-performance

    eval_prefix = str(fold_num) + '-fold average scores:'
    # print(eval_prefix, metric_results_to_string(list_scores=l2r_cv_avg_scores, list_cutoffs=cutoffs))  # print either cv or average performance

    ndcg1 = l2r_cv_avg_scores[0]
    if reward_check is True:
        print("reward_check")
    if result_check is True:
        print("ndcg1:", ndcg1)
    if loop_check is True:
        print("out loop")
    if loop_check2 is True:
        print("out loop2")
    return ndcg1

#
# def pre_evaluate_fold(model,data_id=None, dir_data=None,data_type=None,reward_check=False,result_check=False):
#
#     ranker = model
#     fold_num = 5
#     cutoffs = [1,3,5]
#     l2r_cv_avg_scores = np.zeros(len(cutoffs)) # fold average
#
#     for fold_k in range(1, fold_num + 1):  # evaluation over k-fold data
#
#
#         train_data, test_data, vali_data = load_multiple_data(data_id=data_id, dir_data=dir_data, fold_k=fold_k)
#         # test_data = None
#         if data_type=='train':
#             torch_fold_ndcg_ks = ranker.ndcg_at_ks(test_data=train_data, ks=cutoffs, device='cpu', presort=True)
#         if data_type=='valid':
#             torch_fold_ndcg_ks = ranker.ndcg_at_ks(test_data=vali_data, ks=cutoffs, device='cpu', presort=True)
#         if data_type=='test':
#             torch_fold_ndcg_ks = ranker.ndcg_at_ks(test_data=test_data, ks=cutoffs, device='cpu', presort=True)
#
#         fold_ndcg_ks = torch_fold_ndcg_ks.data.numpy()
#         l2r_cv_avg_scores = np.add(l2r_cv_avg_scores, fold_ndcg_ks)  # sum for later cv-performance
#
#     l2r_cv_avg_scores = np.divide(l2r_cv_avg_scores, fold_num)
#     ndcg1=l2r_cv_avg_scores[0]
#     if reward_check is True:
#         print("reward_check")
#     if result_check is True:
#         print(ndcg1)
#
#     return ndcg1
