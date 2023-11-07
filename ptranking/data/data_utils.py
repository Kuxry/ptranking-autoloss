#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import os
import random
import numpy as np
from pathlib import Path
from enum import Enum, unique, auto

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import torch
import torch.utils.data as data

from ptranking.ltr_adhoc.util.bin_utils import batch_count
from ptranking.utils.numpy.np_extensions import np_arg_shuffle_ties
from ptranking.ltr_adhoc.util.one_hot_utils import get_one_hot_reprs
from ptranking.utils.bigdata.BigPickle import pickle_save, pickle_load

## Supported datasets and formats ##

MSLETOR_SEMI  = ['MQ2007_Semi', 'MQ2008_Semi']
MSLETOR_LIST  = ['MQ2007_List', 'MQ2008_List']
MSLETOR_SUPER = ['MQ2007_Super', 'MQ2008_Super']
MSLETOR       = ['MQ2007_Super', 'MQ2008_Super', 'MQ2007_Semi', 'MQ2008_Semi', 'MQ2007_List', 'MQ2008_List']

'''
The dataset used in the IRGAN paper, which is a revised version of MQ2008_Semi by adding some document vectors per query
in order to mimic unlabeled documents. Unfortunately, the details on how to generate these personally added documents
are not described.
'''
IRGAN_MQ2008_SEMI = ['IRGAN_MQ2008_Semi']

MSLRWEB       = ['MSLRWEB10K', 'MSLRWEB30K']

YAHOO_LTR     = ['Set1', 'Set2']
YAHOO_LTR_5Fold     = ['5FoldSet1', '5FoldSet2']

ISTELLA_LTR   = ['Istella_S', 'Istella', 'Istella_X']
ISTELLA_MAX = 1000000 # As ISTELLA contain extremely large features, e.g., 1.79769313486e+308, we replace features of this kind with a constant 1000000

GLTR_LIBSVM = ['LTR_LibSVM', 'LTR_LibSVM_K']
GLTR_LETOR  = ['LETOR', 'LETOR_K']

"""
GLTR refers to General Learning-to-rank, thus
GLTR_LIBSVM and GLTR_LETOR refer to general learning-to-rank datasets given in the formats of libsvm and LETOR, respectively.

The suffix '_K' indicates that the dataset consists of K folds in order to perform k-fold cross validation.

For GLTR_LIBSVM, it is defined as follows, where features with zero values are not included.
<ground-truth label int> qid:<query_id int> [<feature_id int>:<feature_value float>]

For example:

4 qid:105 2:0.4 8:0.7 50:0.5
1 qid:105 5:0.5 30:0.7 32:0.4 48:0.53
0 qid:210 4:0.9 38:0.01 39:0.5 45:0.7
1 qid:210 1:0.2 8:0.9 31:0.93 40:0.6

The above sample dataset includes two queries, the query “105” has 2 documents, the corresponding ground-truth labels are 4 and 1, respectively.

For GLTR_LETOR, it is defined as follows, where features with zero values are still included and the number of features per row must be the same.
<ground-truth label int> qid:<query_id int> [<feature_id int>:<feature_value float>]

4 qid:105 1:0.4 2:0.7  3:0.5
1 qid:105 1:0.5 2:0.7  3:0.4
0 qid:210 1:0.9 2:0.01 3:0.5
1 qid:210 1:0.2 2:0.9  3:0.93
"""

## supported feature normalization ##
SCALER_LEVEL = ['QUERY', 'DATASET']
SCALER_ID    = ['MinMaxScaler', 'RobustScaler', 'StandardScaler', "SLog1P"]

@unique
class MASK_TYPE(Enum):
    """ Supported ways of masking labels """
    rand_mask_all = auto()
    rand_mask_rele = auto()


@unique
class LABEL_TYPE(Enum):
    """ The types of labels of supported datasets """
    MultiLabel = auto()
    Permutation = auto()


@unique
class SPLIT_TYPE(Enum):
    """ The split-part of a dataset """
    Train = auto()
    Test = auto()
    Validation = auto()

class SymmetricLog1pScaler(object):
    """
    Symmetric Log1p Transformation
    {
    author = {Zhuang, Honglei and Wang, Xuanhui and Bendersky, Michael and Najork, Marc},
    title = {Feature Transformation for Neural Ranking Models},
    booktitle = {Proceedings of the 43rd SIGIR Conference},
    pages = {1649–1652}
    }
    """
    @staticmethod
    def fit_transform(X):
        return np.sign(X) * np.log(1.0 + np.abs(X))


def get_data_meta(data_id=None):
    """ Get the meta-information corresponding to the specified dataset """
    if data_id in MSLRWEB:
        max_rele_level = 4
        label_type = LABEL_TYPE.MultiLabel
        num_features = 136
        has_comment = False
        fold_num = 5

    elif data_id in MSLETOR_SUPER:
        max_rele_level = 2
        label_type = LABEL_TYPE.MultiLabel
        num_features = 46
        has_comment = True
        fold_num = 5

    elif data_id in MSLETOR_SEMI:
        max_rele_level = 2
        label_type = LABEL_TYPE.MultiLabel
        num_features = 46
        has_comment = True
        fold_num = 5

    elif data_id in MSLETOR_LIST:
        max_rele_level = None
        label_type = LABEL_TYPE.Permutation
        num_features = 46
        has_comment = True
        fold_num = 5

    elif data_id in YAHOO_LTR:
        max_rele_level = 4
        label_type = LABEL_TYPE.MultiLabel
        num_features = 700 # libsvm format, rather than uniform number
        has_comment = False
        fold_num = 1

    elif data_id in YAHOO_LTR_5Fold:
        max_rele_level = 4
        label_type = LABEL_TYPE.MultiLabel
        num_features = 700  # libsvm format, rather than uniform number
        has_comment = False
        fold_num = 5

    elif data_id in ISTELLA_LTR:
        max_rele_level = 4
        label_type = LABEL_TYPE.MultiLabel
        num_features = 220  # libsvm format, rather than uniform number
        fold_num = 1
        if data_id in ['Istella_S', 'Istella']:
            has_comment = False
        else:
            has_comment = True
    else:
        raise NotImplementedError

    data_meta = dict(num_features=num_features, has_comment=has_comment, label_type=label_type,
                     max_rele_level=max_rele_level, fold_num=fold_num)
    return data_meta

def get_scaler(scaler_id):
    """ Initialize the scaler-object correspondingly """
    assert scaler_id in SCALER_ID
    if scaler_id == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_id == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_id == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_id == 'SLog1P':
        scaler = SymmetricLog1pScaler()

    return scaler

def get_scaler_setting(data_id, scaler_id=None):
    """
    A default scaler-setting for loading a dataset
    :param data_id:
    :param grid_search: used for grid-search
    :return:
    """
    ''' According to {Introducing {LETOR} 4.0 Datasets}, "QueryLevelNorm version: Conduct query level normalization based on data in MIN version. This data can be directly used for learning. We further provide 5 fold partitions of this version for cross fold validation".
     --> Thus there is no need to perform query_level_scale again for {MQ2007_super | MQ2008_super | MQ2007_semi | MQ2008_semi}
     --> But for {MSLRWEB10K | MSLRWEB30K}, the query-level normalization is ## not conducted yet##.
     --> For {Yahoo_LTR_Set_1 | Yahoo_LTR_Set_1 }, the query-level normalization is already conducted.
     --> For Istella! LETOR, the query-level normalization is not conducted yet.
         We note that ISTELLA contains extremely large features, e.g., 1.79769313486e+308, we replace features of this kind with a constant 1000000.
    '''
    if scaler_id is None:
        if data_id in MSLRWEB or data_id in ISTELLA_LTR:
            scale_data = True
            scaler_id = 'StandardScaler'  # ['MinMaxScaler', 'StandardScaler']
            scaler_level = 'QUERY'  # SCALER_LEVEL = ['QUERY', 'DATASET']
        else:
            scale_data = False
            scaler_id = None
            scaler_level = None
    else:
        scale_data = True
        scaler_level = 'QUERY'

    return scale_data, scaler_id, scaler_level

def get_buffer_file_name(data_id, file, data_dict, presort=None):
    """ Generate the file name """
    min_rele  = data_dict['min_rele']
    if min_rele is not None and min_rele > 0:
        fi_suffix = '_'.join(['MiR', str(min_rele)])
    else:
        fi_suffix = ''

    min_docs = data_dict['min_docs']
    if min_docs is not None and min_docs > 0:
        if len(fi_suffix)>0:
            fi_suffix = '_'.join([fi_suffix, 'MiD', str(min_docs)])
        else:
            fi_suffix = '_'.join(['MiD', str(min_docs)])

    res_suffix = ''
    if data_dict['binary_rele']:
        res_suffix += '_B'
    if data_dict['unknown_as_zero']:
        res_suffix += '_UO'

    pq_suffix = '_'.join([fi_suffix, 'PerQ']) if len(fi_suffix) > 0 else 'PerQ'

    assert presort is not None
    if presort: pq_suffix = '_'.join([pq_suffix, 'PreSort'])

    # plus scaling
    scale_data   = data_dict['scale_data']
    scaler_id    = data_dict['scaler_id'] if 'scaler_id' in data_dict else None
    scaler_level = data_dict['scaler_level'] if 'scaler_level' in data_dict else None
    if scale_data:
        assert scaler_id is not None and scaler_id in SCALER_ID and scaler_level in SCALER_LEVEL
        if 'DATASET' == scaler_level:
            pq_suffix = '_'.join([pq_suffix, 'DS', scaler_id])
        else:
            pq_suffix = '_'.join([pq_suffix, 'QS', scaler_id])

    if data_id in YAHOO_LTR:
        perquery_file = file[:file.find('.txt')].replace(data_id.lower() + '.', 'Buffered' + data_id + '/') + '_' + pq_suffix + res_suffix + '.np'
    elif data_id in ISTELLA_LTR:
        perquery_file = file[:file.find('.txt')].replace(data_id, 'Buffered_' + data_id) + '_' + pq_suffix + res_suffix + '.np'
    else:
        perquery_file = file[:file.find('.txt')].replace('Fold', 'BufferedFold') + '_' + pq_suffix + res_suffix +'.np'

    return perquery_file

## ---------------------------------------------------- ##
""" processing on letor datasets """

def _parse_docid(comment):
    parts = comment.strip().split()
    return parts[2]

def _parse_qid_tok(tok):
    assert tok.startswith('qid:')
    return tok[4:]

def iter_lines(lines, has_targets=True, one_indexed=True, missing=0.0, has_comment=False):
    """
    Transforms an iterator of lines to an iterator of LETOR rows. Each row is represented by a (x, y, qid, comment) tuple.
    Parameters
    ----------
    lines : iterable of lines Lines to parse.
    has_targets : bool, optional, i.e., the relevance label
        Whether the file contains targets. If True, will expect the first token  every line to be a real representing
        the sample's target (i.e. score). If False, will use -1 as a placeholder for all targets.
    one_indexed : bool, optional, i.e., whether the index of the first feature is 1
        Whether feature ids are one-indexed. If True, will subtract 1 from each feature id.
    missing : float, optional
        Placeholder to use if a feature value is not provided for a sample.
    Yields
    ------
    x : array of floats Feature vector of the sample.
    y : float Target value (score) of the sample, or -1 if no target was parsed.
    qid : object Query id of the sample. This is currently guaranteed to be a string.
    comment : str Comment accompanying the sample.
    """
    for line in lines:
        #print(line)
        if has_comment:
            data, _, comment = line.rstrip().partition('#')
            toks = data.split()
        else:
            toks = line.rstrip().split()

        num_features = 0
        feature_vec = np.repeat(missing, 8)
        std_score = -1.0
        if has_targets:
            std_score = float(toks[0])
            toks = toks[1:]

        qid = _parse_qid_tok(toks[0])

        for tok in toks[1:]:
            fid, _, val = tok.partition(':')
            fid = int(fid)
            val = float(val)
            if one_indexed:
                fid -= 1

            assert fid >= 0
            while len(feature_vec) <= fid:
                orig = len(feature_vec)
                feature_vec.resize(len(feature_vec) * 2)
                feature_vec[orig:orig * 2] = missing

            feature_vec[fid] = val
            num_features = max(fid + 1, num_features)

        assert num_features > 0
        feature_vec.resize(num_features)

        if has_comment:
            yield (feature_vec, std_score, qid, comment)
        else:
            yield (feature_vec, std_score, qid)

def parse_letor(source, has_targets=True, one_indexed=True, missing=0.0, has_comment=False):
    """
    Parses a LETOR dataset from `source`.
    Parameters
    ----------
    source : string or iterable of lines String, file, or other file-like object to parse.
    has_targets : bool, optional
    one_indexed : bool, optional
    missing : float, optional
    Returns
    -------
    X : array of arrays of floats Feature matrix (see `iter_lines`).
    y : array of floats Target vector (see `iter_lines`).
    qids : array of objects Query id vector (see `iter_lines`).
    comments : array of strs Comment vector (see `iter_lines`).
    """
    max_width = 0
    feature_vecs, std_scores, qids = [], [], []
    if has_comment:
        comments = []

    it = iter_lines(source, has_targets=has_targets, one_indexed=one_indexed, missing=missing, has_comment=has_comment)
    if has_comment:
        for f_vec, s, qid, comment in it:
            feature_vecs.append(f_vec)
            std_scores.append(s)
            qids.append(qid)
            comments.append(comment)
            max_width = max(max_width, len(f_vec))
    else:
        for f_vec, s, qid in it:
            feature_vecs.append(f_vec)
            std_scores.append(s)
            qids.append(qid)
            max_width = max(max_width, len(f_vec))

    assert max_width > 0
    all_features_mat = np.ndarray((len(feature_vecs), max_width), dtype=np.float64)
    all_features_mat.fill(missing)
    for i, x in enumerate(feature_vecs):
        all_features_mat[i, :len(x)] = x

    all_labels_vec = np.array(std_scores)

    if has_comment:
        docids = [_parse_docid(comment) for comment in comments]
        #features, std_scores, qids, docids
        return all_features_mat, all_labels_vec, qids, docids
    else:
        # features, std_scores, qids
        return all_features_mat, all_labels_vec, qids

def clip_query_data(qid, list_docids=None, feature_mat=None, std_label_vec=None, binary_rele=False,
                    unknown_as_zero=False, clip_query=None, min_docs=None, min_rele=1, presort=None):
    """ Clip the data associated with the same query if required """
    if binary_rele: std_label_vec = np.clip(std_label_vec, a_min=-10, a_max=1) # to binary labels
    if unknown_as_zero: std_label_vec = np.clip(std_label_vec, a_min=0, a_max=10) # convert unknown as zero

    if clip_query:
        if feature_mat.shape[0] < min_docs: # skip queries with documents that are fewer the pre-specified min_docs
            return None
        if (std_label_vec > 0).sum() < min_rele:
            # skip queries with no standard relevant documents, since there is no meaning for both training and testing.
            return None

    assert presort is not None
    if presort:
        '''
        Possible advantages: 1> saving time for evaluation; 
        2> saving time for some models, say the ones need optimal ranking
        '''
        des_inds = np_arg_shuffle_ties(std_label_vec, descending=True)  # sampling by shuffling ties
        feature_mat, std_label_vec = feature_mat[des_inds], std_label_vec[des_inds]
        '''
        if list_docids is None:
            list_docids = None
        else:
            list_docids = []
            for ind in des_inds:
                list_docids.append(list_docids[ind])
        '''
    return (qid, feature_mat, std_label_vec)

def iter_queries(in_file, presort=None, data_dict=None, scale_data=None, scaler_id=None, perquery_file=None, buffer=True):
    '''
    Transforms an iterator of rows to an iterator of queries (i.e., a unit of all the documents and labels associated
    with the same query). Each query is represented by a (qid, feature_mat, std_label_vec) tuple.
    :param in_file:
    :param has_comment:
    :param query_level_scale: perform query-level scaling, say normalization
    :param scaler: MinMaxScaler | RobustScaler
    :param unknown_as_zero: if not labled, regard the relevance degree as zero
    :return:
    '''
    assert presort is not None
    if os.path.exists(perquery_file): return pickle_load(perquery_file)

    if scale_data: scaler = get_scaler(scaler_id=scaler_id)
    min_docs, min_rele = data_dict['min_docs'], data_dict['min_rele']
    unknown_as_zero, binary_rele, has_comment = data_dict['unknown_as_zero'], data_dict['binary_rele'], data_dict['has_comment']

    clip_query = False
    if min_rele is not None and min_rele > 0:
        clip_query = True
    if min_docs is not None and min_docs > 0:
        clip_query = True

    list_Qs = []
    print(in_file)
    with open(in_file, encoding='iso-8859-1') as file_obj:
        dict_data = dict()
        if has_comment:
            all_features_mat, all_labels_vec, qids, docids = parse_letor(file_obj.readlines(), has_comment=True)

            for i in range(len(qids)):
                f_vec = all_features_mat[i, :]
                std_s = all_labels_vec[i]
                qid = qids[i]
                docid = docids[i]

                if qid in dict_data:
                    dict_data[qid].append((std_s, docid, f_vec))
                else:
                    dict_data[qid] = [(std_s, docid, f_vec)]

            del all_features_mat
            # unique qids
            seen = set()
            seen_add = seen.add
            # sequential unique id
            qids_unique = [x for x in qids if not (x in seen or seen_add(x))]

            for qid in qids_unique:
                tmp = list(zip(*dict_data[qid]))

                list_labels_per_q = tmp[0]
                if data_dict['data_id'] in MSLETOR_LIST:
                    ''' convert the original rank-position into grade-labels '''
                    ranking_size = len(list_labels_per_q)
                    list_labels_per_q = [ranking_size-r for r in list_labels_per_q]

                #list_docids_per_q = tmp[1]
                list_features_per_q = tmp[2]
                feature_mat = np.vstack(list_features_per_q)

                if scale_data:
                    if data_dict['data_id'] in ISTELLA_LTR:
                        # due to the possible extremely large features, e.g., 1.79769313486e+308
                        feature_mat = scaler.fit_transform(np.clip(feature_mat, a_min=None, a_max=ISTELLA_MAX))
                    else:
                        feature_mat = scaler.fit_transform(feature_mat)

                Q = clip_query_data(qid=qid, feature_mat=feature_mat, std_label_vec=np.array(list_labels_per_q),
                                    binary_rele=binary_rele, unknown_as_zero=unknown_as_zero, clip_query=clip_query,
                                    min_docs=min_docs, min_rele=min_rele, presort=presort)
                if Q is not None:
                    list_Qs.append(Q)
        else:
            if data_dict['data_id'] in YAHOO_LTR:
                all_features_mat, all_labels_vec, qids = parse_letor(file_obj.readlines(), has_comment=False, one_indexed=False)
            else:
                all_features_mat, all_labels_vec, qids = parse_letor(file_obj.readlines(), has_comment=False)

            for i in range(len(qids)):
                f_vec = all_features_mat[i, :]
                std_s = all_labels_vec[i]
                qid = qids[i]

                if qid in dict_data:
                    dict_data[qid].append((std_s, f_vec))
                else:
                    dict_data[qid] = [(std_s, f_vec)]

            del all_features_mat
            # unique qids
            seen = set()
            seen_add = seen.add
            # sequential unique id
            qids_unique = [x for x in qids if not (x in seen or seen_add(x))]

            for qid in qids_unique:
                tmp = list(zip(*dict_data[qid]))
                list_labels_per_q = tmp[0]
                if data_dict['data_id'] in MSLETOR_LIST:
                    ''' convert the original rank-position into grade-labels '''
                    ranking_size = len(list_labels_per_q)
                    list_labels_per_q = [ranking_size-r for r in list_labels_per_q]

                list_features_per_q = tmp[1]
                feature_mat = np.vstack(list_features_per_q)

                if scale_data:
                    if data_dict['data_id'] in ISTELLA_LTR:
                        # due to the possible extremely large features, e.g., 1.79769313486e+308
                        feature_mat = scaler.fit_transform(np.clip(feature_mat, a_min=None, a_max=ISTELLA_MAX))
                    else:
                        feature_mat = scaler.fit_transform(feature_mat)

                Q = clip_query_data(qid=qid, feature_mat=feature_mat, std_label_vec=np.array(list_labels_per_q),
                                    binary_rele=binary_rele, unknown_as_zero=unknown_as_zero, clip_query=clip_query,
                                    min_docs=min_docs, min_rele=min_rele, presort=presort)
                if Q is not None:
                    list_Qs.append(Q)

    if buffer:
        assert perquery_file is not None
        parent_dir = Path(perquery_file).parent
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        pickle_save(list_Qs, file=perquery_file)

    return list_Qs

## ---------------------------------------------------- ##

class LTRDataset(data.Dataset):
    """
    Loading the specified dataset as torch.utils.data.Dataset.
    We assume that checking the meaningfulness of given loading-setting is conducted beforehand.
    """
    def __init__(self, split_type, file, data_id=None, data_dict=None, eval_dict=None, presort=False, hot=False, buffer=True):
        assert data_id is not None or data_dict is not None
        if data_dict is None: data_dict = self.get_default_data_dict(data_id=data_id)

        self.hot = hot

        ''' data property '''
        self.label_type = data_dict['label_type']

        ''' split-specific settings '''
        self.split_type = split_type
        self.presort = presort
        self.data_id = data_dict['data_id']

        if data_dict['data_id'] in MSLETOR or data_dict['data_id'] in MSLRWEB \
                or data_dict['data_id'] in YAHOO_LTR or data_dict['data_id'] in YAHOO_LTR_5Fold \
                or data_dict['data_id'] in ISTELLA_LTR \
                or data_dict['data_id'] == 'IRGAN_MQ2008_Semi': # supported datasets

            perquery_file = get_buffer_file_name(data_id=data_id, file=file, data_dict=data_dict, presort=self.presort)

            if hot:
                torch_perquery_file = perquery_file.replace('.np', '_Hot.torch')
            else:
                torch_perquery_file = perquery_file.replace('.np', '.torch')

            if eval_dict is not None:
                mask_label, mask_ratio, mask_type = eval_dict['mask_label'], eval_dict['mask_ratio'], eval_dict['mask_type']
                if mask_label:
                    mask_label_str = '_'.join([mask_type, 'Ratio', '{:,g}'.format(mask_ratio)])
                    torch_perquery_file = torch_perquery_file.replace('.torch', '_'+mask_label_str+'.torch')
            else:
                mask_label = False

            if os.path.exists(torch_perquery_file):
                print('loading buffered file ...')
                self.list_torch_Qs = pickle_load(torch_perquery_file)
            else:
                self.list_torch_Qs = []

                scale_data = data_dict['scale_data']
                scaler_id = data_dict['scaler_id'] if 'scaler_id' in data_dict else None
                list_Qs = iter_queries(in_file=file, presort=self.presort, data_dict=data_dict, scale_data=scale_data,
                                       scaler_id=scaler_id, perquery_file=perquery_file, buffer=buffer)

                list_inds = list(range(len(list_Qs)))
                for ind in list_inds:
                    qid, doc_reprs, doc_labels = list_Qs[ind]

                    torch_q_doc_vectors = torch.from_numpy(doc_reprs).type(torch.FloatTensor)
                    #torch_q_doc_vectors = torch.unsqueeze(torch_q_doc_vectors, dim=0)  # a default batch size of 1

                    torch_std_labels = torch.from_numpy(doc_labels).type(torch.FloatTensor)
                    #torch_std_labels = torch.unsqueeze(torch_std_labels, dim=0) # a default batch size of 1

                    if mask_label: # masking TODO to be double-checked
                        if MASK_TYPE[mask_type] == MASK_TYPE.rand_mask_rele:
                            torch_batch_rankings, torch_batch_std_labels = random_mask_rele_labels(
                                batch_ranking=torch_batch_rankings, batch_label=torch_batch_std_labels,
                                mask_ratio=mask_ratio, mask_value=0, presort=self.presort)

                        elif MASK_TYPE[mask_type] == MASK_TYPE.rand_mask_all:
                            masked_res = random_mask_all_labels(batch_ranking=torch_batch_rankings,
                                batch_label=torch_batch_std_labels, mask_ratio=mask_ratio, mask_value=0,
                                                                presort=self.presort)
                            if masked_res is not None:
                                torch_batch_rankings, torch_batch_std_labels = masked_res
                            else:
                                continue
                        else:
                            raise NotImplementedError
                    if hot: # TODO to be updated
                        assert mask_label is not True # not supported since it is rarely used.
                        max_rele_level = data_dict['max_rele_level']
                        assert max_rele_level is not None

                        torch_batch_std_hot_labels = get_one_hot_reprs(torch_batch_std_labels)
                        batch_cnts = batch_count(batch_std_labels=torch_batch_std_labels, max_rele_grade=max_rele_level, descending=True)

                        self.list_torch_Qs.append((qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts))
                    else:
                        self.list_torch_Qs.append((qid, torch_q_doc_vectors, torch_std_labels))
                #buffer
                #print('Num of q:', len(self.list_torch_Qs))
                if buffer:
                    parent_dir = Path(torch_perquery_file).parent
                    if not os.path.exists(parent_dir):
                        os.makedirs(parent_dir)
                    pickle_save(self.list_torch_Qs, torch_perquery_file)
        else:
            raise NotImplementedError

    def get_default_data_dict(self, data_id, scaler_id=None):
        ''' a default setting for loading a dataset '''
        min_docs = 1
        min_rele = 1 # with -1, it means that we don't care with dumb queries that has no relevant documents. Say, for checking the statistics of an original dataset
        scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=data_id, scaler_id=scaler_id)

        train_presort = False if data_id in MSLETOR_SEMI else True

        data_dict = dict(data_id=data_id, min_docs=min_docs, min_rele=min_rele, binary_rele=False,unknown_as_zero=False,
                         train_presort=train_presort, validation_presort=True, test_presort=True,
                         train_batch_size=1, validation_batch_size=1, test_batch_size=1,
                         scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

        data_meta = get_data_meta(data_id=data_id)
        data_dict.update(data_meta)

        return data_dict

    def __len__(self):
        return len(self.list_torch_Qs)

    def __getitem__(self, index):
        qid, torch_batch_rankings, torch_batch_std_labels = self.list_torch_Qs[index]
        return qid, torch_batch_rankings, torch_batch_std_labels

    def iter_hot(self):
        list_inds = list(range(len(self.list_torch_Qs)))
        for ind in list_inds:
            qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts = self.list_torch_Qs[ind]
            yield qid, torch_batch_rankings, torch_batch_std_labels, torch_batch_std_hot_labels, batch_cnts

## Customize Sampler for Batch Processing ##

def pre_allocate_batch(dict_univ_bin, num_docs_per_batch):
    '''
    Based on the expected number of documents to process within a single batch, we merge the queries that have the same number of documents to form a batch
    @param dict_univ_bin: [unique_value, bin of index]
    @param num_docs_per_batch:
    @return:
    '''
    list_batch_inds = []

    if 1 == num_docs_per_batch: # a simple but time-consuming per-query processing, namely the batch_size is always one
        for univ in dict_univ_bin:
            bin = dict_univ_bin[univ]
            for index in bin:
                single_ind_as_batch = [index]
                list_batch_inds.append(single_ind_as_batch)

        return list_batch_inds
    else:
        for univ in dict_univ_bin:
            bin = dict_univ_bin[univ]
            bin_length = len(bin)

            if univ * bin_length < num_docs_per_batch: # merge all queries as one batch
                list_batch_inds.append(bin)
            else:
                if univ < num_docs_per_batch: # split with an approximate value
                    num_inds_per_batch = num_docs_per_batch // univ
                    for i in range(0, bin_length, num_inds_per_batch):
                        sub_bin = bin[i: min(i+num_inds_per_batch, bin_length)]
                        list_batch_inds.append(sub_bin)
                else: # one single query as a batch
                    for index in bin:
                        single_ind_as_batch = [index]
                        list_batch_inds.append(single_ind_as_batch)

        return list_batch_inds

class LETORSampler(data.Sampler):
    '''
    Customized sampler for LETOR datasets based on the observation that:
    though the number of documents per query may differ, there are many queries that have the same number of documents, especially with a big dataset.
    '''
    def __init__(self, data_source, rough_batch_size=None):
        list_num_docs = []
        for qid, torch_batch_rankings, torch_batch_std_labels in data_source:
            list_num_docs.append(torch_batch_std_labels.size(0))

        dict_univ_bin = {}
        for ind, univ in enumerate(list_num_docs):
            if univ in dict_univ_bin:
                dict_univ_bin[univ].append(ind)
            else:
                bin = [ind]
                dict_univ_bin[univ] = bin

        self.list_batch_inds = pre_allocate_batch(dict_univ_bin=dict_univ_bin, num_docs_per_batch=rough_batch_size)

    def __iter__(self):
        for batch_inds in self.list_batch_inds:
            yield batch_inds

class LETORPercentSampler(data.Sampler):
    '''
    Customized sampler for LETOR datasets
    '''
    def __init__(self, data_source, percent=.01):
        '''
        @param data_source: dataset to sample from
        @param percent: the ratio of being used part
        '''
        num_queries = data_source.__len__()
        #print('num_queries', num_queries)
        num_used_queries = int(num_queries * percent)
        self.list_used_inds = list(np.random.permutation(num_queries)[0:num_used_queries])

    def __iter__(self):
        for ind in self.list_used_inds:
            yield [ind]

## ------ loading data for ltr_gbm ----- ##

class GBMDataset():
    """
    Loading the specified dataset as gradient boosting machines.
    We assume that checking the meaningfulness of given loading-setting is conducted beforehand.
    """
    def __init__(self, split_type, file, data_id=None, data_dict=None, eval_dict=None, presort=False, buffer=True):
        assert data_id is not None or data_dict is not None
        if data_dict is None: data_dict = self.get_default_data_dict(data_id=data_id)

        ''' data property '''
        self.label_type = data_dict['label_type']

        ''' split-specific settings '''
        self.split_type = split_type
        self.presort = presort
        self.data_id = data_dict['data_id']

        if data_dict['data_id'] in MSLETOR or data_dict['data_id'] in MSLRWEB \
                or data_dict['data_id'] in YAHOO_LTR or data_dict['data_id'] in YAHOO_LTR_5Fold \
                or data_dict['data_id'] in ISTELLA_LTR \
                or data_dict['data_id'] == 'IRGAN_MQ2008_Semi': # supported datasets

            perquery_file = get_buffer_file_name(data_id=data_id, file=file, data_dict=data_dict, presort=self.presort)

            if eval_dict is not None:
                mask_label, mask_ratio, mask_type = eval_dict['mask_label'], eval_dict['mask_ratio'], eval_dict['mask_type']
                if mask_label:
                    mask_label_str = '_'.join([mask_type, 'Ratio', '{:,g}'.format(mask_ratio)])
            else:
                mask_label = False

            self.list_torch_Qs = []

            scale_data = data_dict['scale_data']
            scaler_id = data_dict['scaler_id'] if 'scaler_id' in data_dict else None
            list_Qs = iter_queries(in_file=file, presort=self.presort, data_dict=data_dict, scale_data=scale_data,
                                   scaler_id=scaler_id, perquery_file=perquery_file, buffer=buffer)

            num_queries, num_all_docs, num_features = len(list_Qs), 0, data_dict['num_features']
            for _, _, doc_labels in list_Qs:
                ranking_size = len(doc_labels)
                num_all_docs += ranking_size

            self.group = np.empty((num_queries,))
            self.target = np.empty((num_all_docs,))
            self.data = np.empty((num_all_docs, num_features))
            head = 0
            for i, entry in enumerate(list_Qs):
                qid, doc_reprs, doc_labels = entry
                if mask_label: # masking TODO to be added
                    '''
                    if MASK_TYPE[mask_type] == MASK_TYPE.rand_mask_rele:
                        torch_batch_rankings, torch_batch_std_labels = random_mask_rele_labels(
                            batch_ranking=torch_batch_rankings, batch_label=torch_batch_std_labels,
                            mask_ratio=mask_ratio, mask_value=0, presort=self.presort)
                    elif MASK_TYPE[mask_type] == MASK_TYPE.rand_mask_all:
                        masked_res = random_mask_all_labels(batch_ranking=torch_batch_rankings,
                            batch_label=torch_batch_std_labels, mask_ratio=mask_ratio, mask_value=0, presort=self.presort)
                        if masked_res is not None:
                            torch_batch_rankings, torch_batch_std_labels = masked_res
                        else:
                            continue
                    else:
                        raise NotImplementedError
                    '''
                ranking_size = len(doc_labels)
                tail = head + ranking_size
                self.group[i] = ranking_size
                self.data[head:tail, :] = doc_reprs
                self.target[head:tail] = doc_labels
                head = tail
        else:
            raise NotImplementedError

    def get_default_data_dict(self, data_id, scaler_id=None):
        ''' a default setting for loading a dataset '''
        min_docs = 1
        min_rele = 1 # with -1, it means that we don't care with dumb queries that has no relevant documents. Say, for checking the statistics of an original dataset
        scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=data_id, scaler_id=scaler_id)

        train_presort = False if data_id in MSLETOR_SEMI else True

        data_dict = dict(data_id=data_id, min_docs=min_docs, min_rele=min_rele, binary_rele=False,unknown_as_zero=False,
                         train_presort=train_presort, validation_presort=True, test_presort=True,
                         train_batch_size=1, validation_batch_size=1, test_batch_size=1,
                         scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)

        data_meta = get_data_meta(data_id=data_id)
        data_dict.update(data_meta)

        return data_dict

    def get_data(self):
        return (self.data, self.target, self.group)

## ------ loading letor data as libsvm data ----- ##

def get_buffer_file_name_libsvm(in_file, data_id=None, eval_dict=None, need_group=True):
    """ get absolute paths of data file and group file """

    if data_id in MSLETOR or data_id in MSLRWEB:
        buffer_prefix       = in_file.replace('Fold', 'BufferedFold')
        file_buffered_data  = buffer_prefix.replace('txt', 'data')
        if need_group: file_buffered_group = buffer_prefix.replace('txt', 'group')
    elif data_id in YAHOO_LTR:
        buffer_prefix       = in_file[:in_file.find('.txt')].replace(data_id.lower() + '.', 'Buffered' + data_id + '/')
        file_buffered_data  = buffer_prefix  + '.data'
        if need_group: file_buffered_group = buffer_prefix  + '.group'
    elif data_id in ISTELLA_LTR:
        buffer_prefix       = in_file[:in_file.find('.txt')].replace(data_id, 'Buffered_' + data_id)
        file_buffered_data  = buffer_prefix  + '.data'
        if need_group: file_buffered_group = buffer_prefix  + '.group'
    else:
        raise NotImplementedError

    if eval_dict is not None and eval_dict['mask_label']:
        mask_ratio = eval_dict['mask_ratio']
        mask_type = eval_dict['mask_type']
        mask_label_str = '_'.join([mask_type, 'Ratio', '{:,g}'.format(mask_ratio)])
        file_buffered_data = file_buffered_data.replace('.data', '_'+mask_label_str+'.data')
        file_buffered_group = file_buffered_group.replace('.group', '_'+mask_label_str+'.group')

    if need_group:
        return file_buffered_data, file_buffered_group
    else:
        return file_buffered_data

def letor_to_libsvm(doc_reprs=None, doc_labels=None, output_feature=None, output_group=None, need_group=False):
    ''' convert query-level letor-data to libsvm data '''
    num_docs = doc_reprs.shape[0]
    if need_group: output_group.write(str(num_docs) + "\n") # group file
    for i in range(num_docs): # per document only include nonzero features
        feats = doc_reprs[i, :].tolist()
        libsvm_feats = []
        for key, val in enumerate(feats):
            if val != 0.0: libsvm_feats.append(':'.join([str(key+1), str(val)]))
        output_feature.write(str(doc_labels[i]) + " " + " ".join(libsvm_feats) + "\n")


def load_letor_data_as_libsvm_data(in_file, split_type=None, data_id=None, min_docs=None, min_rele=None,
                                   data_dict=None, eval_dict=None, need_group=True, presort=None, scaler_id=None):
    """
    Load data by firstly converting letor data as libsvm data
    :param in_file:
    :param min_docs:
    :param min_rele:
    :param data_id:
    :param eval_dict:
    :param need_group: required w.r.t. xgboost, lightgbm
    :return:
    """
    assert data_id is not None or data_dict is not None
    if data_dict is None:
        scale_data, scaler_id, scaler_level = get_scaler_setting(data_id=data_id, scaler_id=scaler_id)
        data_dict = dict(data_id=data_id, min_docs=min_docs, min_rele=min_rele, binary_rele=False,
                         unknown_as_zero = False, scale_data=scale_data, scaler_id=scaler_id, scaler_level=scaler_level)
        data_meta = get_data_meta(data_id=data_id)
        data_dict.update(data_meta)
    elif data_id is None:
        data_id = data_dict['data_id']

    if need_group:
        file_buffered_data, file_buffered_group = get_buffer_file_name_libsvm(in_file, data_id=data_id, eval_dict=eval_dict, need_group=True)
    else:
        file_buffered_data = get_buffer_file_name_libsvm(in_file, data_id=data_id, eval_dict=eval_dict, need_group=False)

    if os.path.exists(file_buffered_data):
        if need_group:
            return file_buffered_data, file_buffered_group
        else:
            return file_buffered_data
    else:
        parent_dir = Path(file_buffered_data).parent
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        output_feature = open(file_buffered_data, "w")
        if need_group: output_group = open(file_buffered_group, "w")

        perquery_file = get_buffer_file_name(data_id=data_id, file=in_file, data_dict=data_dict, presort=presort)
        list_Qs = iter_queries(in_file=in_file, data_dict=data_dict, scale_data=data_dict['scale_data'],
                               scaler_id=data_dict['scaler_id'], perquery_file=perquery_file, buffer=True, presort=presort)

        if eval_dict is not None and eval_dict['mask_label'] and split_type==SPLIT_TYPE.Train:
            if MASK_TYPE.rand_mask_rele == MASK_TYPE[eval_dict['mask_type']]:
                for qid, doc_reprs, doc_labels in list_Qs:
                    doc_labels = np_random_mask_rele_labels(batch_label=doc_labels, mask_ratio=eval_dict['mask_ratio'], mask_value=0)
                    if doc_labels is not None:
                        letor_to_libsvm(doc_reprs=doc_reprs.astype(np.float32), doc_labels=doc_labels.astype(np.int),
                                        output_feature=output_feature, output_group=output_group, need_group=need_group)
            elif MASK_TYPE.rand_mask_all == MASK_TYPE[eval_dict['mask_type']]:
                for qid, doc_reprs, doc_labels in list_Qs:
                    doc_labels = np_random_mask_all_labels(batch_label=doc_labels, mask_ratio=eval_dict['mask_ratio'], mask_value=0)
                    if doc_labels is not None:
                        letor_to_libsvm(doc_reprs=doc_reprs.astype(np.float32), doc_labels=doc_labels.astype(np.int),
                                        output_feature=output_feature, output_group=output_group, need_group=need_group)
            else:
                raise NotImplementedError
        else:
            for qid, doc_reprs, doc_labels in list_Qs:
                letor_to_libsvm(doc_reprs=doc_reprs.astype(np.float32), doc_labels=doc_labels.astype(np.int),
                                output_feature=output_feature, output_group=output_group, need_group=need_group)

        output_group.close()
        output_feature.close()

    if need_group:
        return file_buffered_data, file_buffered_group
    else:
        return file_buffered_data

#######################
# Masking Application #
#######################

torch_zero = torch.FloatTensor([0.0])
def random_mask_all_labels(batch_ranking, batch_label, mask_ratio, mask_value=0, presort=False):
    '''
    Mask the ground-truth labels with the specified ratio as '0'. Meanwhile, re-sort according to the labels if required.
    :param doc_reprs:
    :param doc_labels:
    :param mask_ratio: the ratio of labels to be masked
    :param mask_value:
    :param presort:
    :return:
    '''

    size_ranking = batch_label.size(1)
    num_to_mask = int(size_ranking*mask_ratio)
    mask_ind = np.random.choice(size_ranking, size=num_to_mask, replace=False)

    batch_label[:, mask_ind] = mask_value

    if torch.gt(batch_label, torch_zero).any(): # whether the masked one includes explicit positive labels
        if presort: # re-sort according to the labels if required
            std_labels = torch.squeeze(batch_label)
            sorted_labels, sorted_inds = torch.sort(std_labels, descending=True)

            batch_label = torch.unsqueeze(sorted_labels, dim=0)
            batch_ranking = batch_ranking[:, sorted_inds, :]

        return batch_ranking, batch_label
    else:
        return None


def random_mask_rele_labels(batch_ranking, batch_label=None, mask_ratio=None, mask_value=0, presort=False):
    '''
    Mask the ground-truth labels with the specified ratio as '0'. Meanwhile, re-sort according to the labels if required.
    :param doc_reprs:
    :param doc_labels:
    :param mask_ratio: the ratio of labels to be masked
    :param mask_value:
    :param presort:
    :return:
    '''

    assert 1 == batch_label.size(0) # todo for larger batch-size, need to per-dimension masking

    # squeeze for easy process
    docs, labels = torch.squeeze(batch_ranking, dim=0), torch.squeeze(batch_label)

    all_rele_inds = torch.gt(labels, torch_zero).nonzero()
    num_rele = all_rele_inds.size()[0]

    num_to_mask = int(num_rele*mask_ratio)
    mask_inds = np.random.choice(num_rele, size=num_to_mask, replace=False)

    rele_inds_to_mask = all_rele_inds[mask_inds, 0] # the 0-column corresponds to original rele index since all_rele_inds.size()=(num_rele, 1)

    batch_label[:, rele_inds_to_mask] = mask_value

    if torch.gt(batch_label, torch_zero).any(): # whether the masked one includes explicit positive labels
        if presort: # re-sort according to the labels if required
            std_labels = torch.squeeze(batch_label)
            sorted_labels, sorted_inds = torch.sort(std_labels, descending=True)

            batch_label = torch.unsqueeze(sorted_labels, dim=0)
            batch_ranking = batch_ranking[:, sorted_inds, :]

        return batch_ranking, batch_label
    else:
        # only supports enough rele labels
        raise NotImplementedError


def np_random_mask_all_labels(batch_label, mask_ratio, mask_value=0):
    '''
    Mask the ground-truth labels with the specified ratio as '0'.
    '''
    size_ranking = len(batch_label)
    num_to_mask = int(size_ranking*mask_ratio)
    mask_ind = np.random.choice(size_ranking, size=num_to_mask, replace=False)

    batch_label[mask_ind] = mask_value

    if np.greater(batch_label, 0.0).any(): # whether the masked one includes explicit positive labels
        return batch_label
    else:
        return None


def np_random_mask_rele_labels(batch_label, mask_ratio, mask_value=0):
    '''
    Mask the ground-truth labels with the specified ratio as '0'.
    '''
    all_rele_inds = np.greater(batch_label, 0).nonzero()[0] # due to one-dimension
    #print('all_rele_inds', all_rele_inds)
    num_rele = all_rele_inds.shape[0]
    #print('num_rele', num_rele)

    num_to_mask = int(num_rele*mask_ratio)
    mask_inds = np.random.choice(num_rele, size=num_to_mask, replace=False)
    #print('mask_inds', mask_inds)

    rele_inds_to_mask = all_rele_inds[mask_inds]
    #print('rele_inds_to_mask', rele_inds_to_mask)sss

    batch_label[rele_inds_to_mask] = mask_value

    if np.greater(batch_label, 0.0).any(): # whether the masked one includes explicit positive labels
        return batch_label
    else:
        return None

