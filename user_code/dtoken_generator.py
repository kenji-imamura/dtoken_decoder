# The original part of the fairseq: Copyright (c) Facebook, Inc. and its affiliates.
# The modified and additional parts:
# Copyright (c) 2020 National Institute of Information and Communications Technology.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
from fairseq.search import Search

class DoubleTokenSequenceGenerator(SequenceGenerator):
    def __init__(self, tgt_dict, no_incremental=False, **kwargs):
        super().__init__(tgt_dict, **kwargs)
        self.bos = tgt_dict.bos()
        self.no_incremental = no_incremental

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = DoubleTokenEnsembleModel(models)
        return self._generate(model, sample, **kwargs)

    @torch.no_grad()
    def _generate(
        self,
        model,
        sample,
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 2).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size, max_len + 3).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        if bos_token is None:
            tokens[:, 0] = self.bos
            tokens[:, 1] = self.bos 
        else:
            tokens[:, 0] = bos_token
            tokens[:, 1] = bos_token
        attn, attn_buf = None, None

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 3 * beam_size  # 3 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size or step == max_len:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, pre_eos_tokens, pre_eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.size(0)

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 2:step+3]  # skip the first two indices, which are BOS and EOS
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 2:step+3] if attn is not None else None
            
            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]

            # insert pre_eos_tokens and scores
            assert not tokens_clone.eq(self.eos).any()
            if pre_eos_tokens is not None:
                tokens_clone[:, step - 1] = pre_eos_tokens
                pos_scores[:, step - 1] = pre_eos_scores
            tokens_clone[:, step] = self.eos
            pos_scores[:, step] = eos_scores

            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]
                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())

            newly_finished_idx = []
            newly_finished_sent = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    newly_finished_idx.append(unfin_idx)
                    newly_finished_sent.append(sent)
            return newly_finished_idx, newly_finished_sent

        reorder_state = None
        batch_idxs = None
        looping = True
        for step in range(0, max_len + 1, 2):  # one extra step for EOS marker
            if not looping: break

            #####################################
            # Compute probability distribution of each token
            #####################################
            bi_tokens = tokens[:, :step + 2]
            avg_probs, avg_attn = model.forward_decoder(
                bi_tokens, encoder_outs, temperature=self.temperature,
            )
            avg_probs[:, :, self.pad] = -math.inf  # never select pad
            avg_probs[:, :, self.unk] -= self.unk_penalty  # apply unk penalty
            
            # handle max/min length constraints
            if step >= max_len:
                avg_probs[:, :, :self.eos] = -math.inf
                avg_probs[:, :, self.eos + 1:] = -math.inf
            if step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                avg_probs[:, :, self.eos] = -math.inf

            # Record attention scores
            if avg_attn is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                # Because attn[:. :, 0] was used. I closed up them.
                attn[:, :, step:step+2].copy_(avg_attn.transpose(1, 2))

            # buffersを設定
            scores = scores.type_as(avg_probs)
            scores_buf = scores_buf.type_as(avg_probs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            #####################################
            # Select Top-K candidates according to the beam
            #####################################
            cand_scores, cand_indices, cand_beams = self._search_step(
                step,
                avg_probs.view(bsz, -1, 2, self.vocab_size).contiguous(),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                max_len=max_len,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # or candidates with a score of -inf
            dtoken_eos_mask = None
            finalized_sents = set()
            fin_sent_set = set()
            pre_eos_tokens = buffer('pre_eos_tokens')
            for ii in range(2):
                eos_mask = (cand_indices.eq(self.eos) & cand_scores.ne(-math.inf))[:, :, ii]
                if ii == 0:
                    dtoken_eos_mask = eos_mask
                else:
                    eos_mask &= ~dtoken_eos_mask
                    dtoken_eos_mask |= eos_mask

                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )

                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_indices[:, :beam_size, 0],
                        mask=eos_mask[:, :beam_size],
                        out=pre_eos_tokens,
                    )
                    torch.masked_select(
                        cand_scores[:, :beam_size, :],
                        mask=eos_mask[:, :beam_size].unsqueeze(2),
                        out=eos_scores,
                    )
                    eos_scores = eos_scores.view(-1, 2)
                    fin_idx, fin_sent = finalize_hypos(step + ii, eos_bbsz_idx, eos_scores[:, ii],
                                                       pre_eos_tokens if ii > 0 else None,
                                                       eos_scores[:, 0] if ii > 0 else None,
                    )
                    finalized_sents |= set(fin_idx)
                    fin_sent_set |= set(fin_sent)

            for sent in fin_sent_set: finished[sent] = True
            num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                looping = False
                break
            assert step < max_len

            #####################################
            # Shrink the beam by removing finalized sentences
            #####################################
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(list(finalized_sents))] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                dtoken_eos_mask = dtoken_eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]

                bbsz_offsets.resize_(new_bsz, 1)	# shrink

                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)

                bsz = new_bsz
            else:
                batch_idxs = None

            #####################################
            # Extend the beam and reorder buffers
            #####################################

            # Set active_mask so that values > cand_size indicate eos
            # and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            active_mask = buffer('active_mask')
            torch.add(
                dtoken_eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:dtoken_eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist, active_hypos)
            )

            # active indices
            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )

            # save scores of the active hypotheses
            active_scores1 = torch.gather(
                cand_scores[:, :, 0].view(bsz, -1), dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )
            active_scores2 = torch.gather(
                cand_scores[:, :, 1].view(bsz, -1), dim=1, index=active_hypos,
                out=scores[:, step+1].view(bsz, beam_size),
            )
            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores2.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 2], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 2],
            )
            torch.gather(
                cand_indices[:, :, 0].view(bsz, -1), dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 2],
            )
            torch.gather(
                cand_indices[:, :, 1].view(bsz, -1), dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 3],
            )

            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores[:, :, 0].view(bsz, -1), dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )
            torch.gather(
                cand_scores[:, :, 1].view(bsz, -1), dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )
            
            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder decoder internal states based on the prev choice of beams
            reorder_state = active_bbsz_idx.clone()
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(reorder_state)
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)

            continue

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
        return finalized


    def _search_step(self, step, lprobs, scores, max_len=1024):
        self.search._init_buffers(lprobs)
        bsz, beam_size, _, vocab_size = lprobs.size()

        ################
        # Forward (left) token
        ################
        f_lprobs = None
        if step == 0:
            f_lprobs = lprobs[:, ::beam_size, 0, :].view(bsz, 1, -1)
        else:
            # make probs contain cumulative scores for each hypothesis
            f_lprobs = lprobs[:, :, 0, :].view(bsz, beam_size, -1).contiguous()
            f_lprobs.add_(scores[:, :, step - 1].view(bsz, beam_size, 1))

        torch.topk(
            f_lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                f_lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.search.scores_buf, self.search.indices_buf),
        )
        torch.div(self.search.indices_buf, vocab_size, out=self.search.beams_buf)
        self.search.indices_buf.fmod_(vocab_size)

        scores_buf = self.search.scores_buf.clone()
        indices_buf = self.search.indices_buf.clone()
        beams_buf = self.search.beams_buf.clone()

        ################
        # Backward (right) token
        ################
        b_lprobs = None
        if step == 0:
            b_lprobs = lprobs[:, ::beam_size, 1, :].view(bsz, 1, -1).contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            b_lprobs = lprobs[:, :, 1, :].view(bsz, beam_size, -1).contiguous()

        offset_step = b_lprobs.size(1)
        offset = (torch.arange(0, bsz) * offset_step).unsqueeze(1).type_as(beams_buf)
        offset = offset + beams_buf
        b_lprobs = b_lprobs.view(bsz * offset_step, -1)[offset]

        # If the forward token is EOS, the backward token is also EOS.
        eos_mask = indices_buf.eq(self.eos)
        if eos_mask.any():
            b_lprobs[:, :, :].masked_fill_(eos_mask.unsqueeze(2), -math.inf)
            b_lprobs[:, :, self.eos].masked_fill_(eos_mask, 0.0)
        b_lprobs.add_(scores_buf.unsqueeze(-1))
        torch.topk(
            b_lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 3,
                b_lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.search.scores_buf, self.search.indices_buf),
        )
        torch.div(self.search.indices_buf, vocab_size, out=self.search.beams_buf)
        self.search.indices_buf.fmod_(vocab_size)

        beams_ret = beams_buf.gather(1, self.search.beams_buf)
        scores_ret = torch.stack([scores_buf.gather(1, self.search.beams_buf),
                                  self.search.scores_buf], 2)
        indices_ret = torch.stack([indices_buf.gather(1, self.search.beams_buf),
                                   self.search.indices_buf], 2)

        return scores_ret, indices_ret, beams_ret


class DoubleTokenEnsembleModel(EnsembleModel):
    """A wrapper around an ensemble of models."""

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_two(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            out_probs, out_attn = self._decode_two(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(out_probs)
            if out_attn is not None:
                if avg_attn is None:
                    avg_attn = out_attn
                else:
                    avg_attn.add_(out_attn)

        avg_probs = torch.logsumexp(
            torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn


    def _decode_two(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -2:, :]

        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        return probs[:, -2:, :], attn[:, -2:, :] if attn is not None else None
