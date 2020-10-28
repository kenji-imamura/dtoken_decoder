# The original part of the fairseq: Copyright (c) Facebook, Inc. and its affiliates.
# The modified and additional parts:
# Copyright (c) 2020 National Institute of Information and Communications Technology.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import copy
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from .dtoken_generator import DoubleTokenSequenceGenerator

@register_task('dtoken_translation')
class DoubleTokenTranslationTask(TranslationTask):
    """
    Translation task for double-token autoregressive decoding.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """
    def build_generator(self, args):
        return DoubleTokenSequenceGenerator(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            sampling=getattr(args, 'sampling', False),
            sampling_topk=getattr(args, 'sampling_topk', -1),
            sampling_topp=getattr(args, 'sampling_topp', -1.0),
            temperature=getattr(args, 'temperature', 1.),
            diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            no_incremental=getattr(args, 'no_incremental', False),
        )


    def _arrange_sample(self, sample, tgt_dict):
        """
        Arrange the original mini-batch for the double-token decoder.
        For compatibility, <s> (index=0) is used as the BOS and EOS tokens,
        and </s> (index=2) is used as the end-of-decoding token.
        """
        new_sample = copy.deepcopy(sample)
        target = new_sample['target']
        target = target.masked_fill(target.eq(tgt_dict.eos_index), tgt_dict.bos_index)
        target = torch.cat([target.new(target.size(0), 1).fill_(tgt_dict.bos_index), target], dim=1)
        len    = target.ne(tgt_dict.pad_index).sum(1)
        new_len = len.max() - 1
        if new_len % 2 != 0: new_len += 1

        target_list = []
        prev_token_list = []
        for ii in range(target.size(0)):
            x = target[ii, 0:len[ii]]
            x = torch.cat([x.unsqueeze(-1), x.flip(0).unsqueeze(-1)], dim=1)
            x = x.view(-1)
            if x.size(0) < new_len + 2: x.resize_(new_len + 2)
            x[len[ii]:].fill_(tgt_dict.pad_index)
            x[len[ii]] = tgt_dict.eos_index
            if len[ii] % 2 == 0: x[len[ii]+1] = tgt_dict.eos_index

            prev_token_list.append(x[:new_len].unsqueeze(0))
            target_list.append(x[2:new_len+2].unsqueeze(0))

        new_sample['target'] = torch.cat(target_list, dim=0).detach()
        new_sample['net_input']['prev_output_tokens'] = torch.cat(prev_token_list, dim=0).detach()
        new_sample['ntokens'] = int(new_sample['target'].ne(tgt_dict.pad_index).sum())
        return new_sample

    def _reorder_tokens(self, tensor):
        """
        Reorder the output of the double-token decoder to the original order.
        """
        width = tensor.size(-1) - 1
        f_order = torch.arange(0, width, 2, device=tensor.device)
        b_order = torch.arange(1, width, 2, device=tensor.device).flip(0)
        return torch.cat([tensor.index_select(-1, f_order),
                          tensor.index_select(-1, b_order),
                          tensor[-1].unsqueeze(0)], dim=-1)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        new_sample = self._arrange_sample(sample, model.decoder.dictionary)
        return super().train_step(new_sample, model, criterion, optimizer,
                                  ignore_grad=ignore_grad)

    def valid_step(self, sample, model, criterion):
        new_sample = self._arrange_sample(sample, model.decoder.dictionary)
        return super().valid_step(new_sample, model, criterion)

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        finalized = super().inference_step(
            generator, models, sample, prefix_tokens=prefix_tokens)
        for sent in finalized:
            for hypo in sent:
                hypo['tokens'] = self._reorder_tokens(hypo['tokens'])
        return finalized
    
