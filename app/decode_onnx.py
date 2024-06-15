import os
import math
import logging
import warnings
from typing import Optional, List, Dict
from dataclasses import dataclass, field

import k2
import torch
import kaldifeat
import onnxruntime as ort
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence
from icefall import NgramLm, NgramLmStateCost

class OnnxModel:
    def __init__(
        self,
        encoder_model_filename: str,
        decoder_model_filename: str,
        joiner_model_filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 4
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.init_encoder(encoder_model_filename)
        self.init_decoder(decoder_model_filename)
        self.init_joiner(joiner_model_filename)

    def init_encoder(self, encoder_model_filename: str):
        self.encoder = ort.InferenceSession(
            encoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_decoder(self, decoder_model_filename: str):
        self.decoder = ort.InferenceSession(
            decoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        decoder_meta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(decoder_meta["context_size"])
        self.vocab_size = int(decoder_meta["vocab_size"])
        self.blank_id = 0

        logging.info(f"context_size: {self.context_size}")
        logging.info(f"vocab_size: {self.vocab_size}")

    def init_joiner(self, joiner_model_filename: str):
        self.joiner = ort.InferenceSession(
            joiner_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        joiner_meta = self.joiner.get_modelmeta().custom_metadata_map
        self.joiner_dim = int(joiner_meta["joiner_dim"])

        logging.info(f"joiner_dim: {self.joiner_dim}")

    def run_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> [torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
          x_lens:
            A 2-D tensor of shape (N,). Its dtype is torch.int64
        Returns:
          Return a  containing:
            - encoder_out, its shape is (N, T', joiner_dim)
            - encoder_out_lens, its shape is (N,)
        """
        out = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])

    def run_decoder(self, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
          decoder_input:
            A 2-D tensor of shape (N, context_size)
        Returns:
          Return a 2-D tensor of shape (N, joiner_dim)
        """
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: decoder_input.numpy()},
        )[0]

        return torch.from_numpy(out)

    def run_joiner(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            A 2-D tensor of shape (N, joiner_dim)
          decoder_out:
            A 2-D tensor of shape (N, joiner_dim)
        Returns:
          Return a 2-D tensor of shape (N, vocab_size)
        """
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out.numpy(),
                self.joiner.get_inputs()[1].name: decoder_out.numpy(),
            },)[0]

        return torch.from_numpy(out)
    

class OnnxLM:
    def __init__(self, filename: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.LM = ort.InferenceSession(
            filename,
            sess_options=session_opts,
        )

        meta_data = self.LM.get_modelmeta().custom_metadata_map
        self.sos_id = int(meta_data["sos_id"])
        self.eos_id = int(meta_data["eos_id"])
        self.vocab_size = int(meta_data["vocab_size"])
        self.num_layers = int(meta_data["num_layers"])
        self.hidden_size = int(meta_data["hidden_size"])
        print(meta_data)
    
    def __call__(self, x: torch.Tensor, state: Optional[torch.Tensor]=None) -> torch.Tensor:
        if state is not None:
            h, c = state
        else:
            h = torch.zeros(self.num_layers, 1, self.hidden_size).to(torch.float32)
            c = torch.zeros(self.num_layers, 1, self.hidden_size).to(torch.float32)
        
        out = self.LM.run(
            [
                self.LM.get_outputs()[0].name,
                self.LM.get_outputs()[1].name,
                self.LM.get_outputs()[2].name,
            ],
            {
                self.LM.get_inputs()[0].name: x.numpy(),
                self.LM.get_inputs()[1].name: h.numpy(),
                self.LM.get_inputs()[2].name: c.numpy(),
            },
        )
        return torch.from_numpy(out[0]), \
               (torch.from_numpy(out[1]), torch.from_numpy(out[2]))
    

@dataclass
class Hypothesis:
    # The preDicted tokens so far.
    # Newly preDicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    log_prob: torch.Tensor

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    # the lm score for next token given the current ys
    lm_score: Optional[torch.Tensor] = None

    # the RNNLM states (h and c in LSTM)
    state: Optional[List[torch.Tensor]] = None

    # N-gram LM state
    state_cost: Optional[NgramLmStateCost] = None

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))
    

class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A Dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.

        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.

        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy
            torch.logaddexp(old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.

        Caution:
          `self` is modified **in-place**.

        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: torch.Tensor) -> "HypothesisList":
        """Remove all Hypotheses whose log_prob is less than threshold.

        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.

        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `log_prob` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = List(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].log_prob / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(Dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)
    

def get_hyps_shape(hyps: List[HypothesisList]): #-> k2.RaggedShape:
    """Return a ragged shape with axes [utt][num_hyps].

    Args:
      hyps:
        len(hyps) == batch_size. It contains the current hypothesis for
        each utterance in the batch.
    Returns:
      Return a ragged shape with 2 axes [utt][num_hyps]. Note that
      the shape is on CPU.
    """
    num_hyps = [len(h) for h in hyps]

    # torch.cumsum() is inclusive sum, so we put a 0 at the beginning
    # to get exclusive sum later.
    num_hyps.insert(0, 0)

    num_hyps = torch.tensor(num_hyps)
    row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)
    ans = k2.ragged.create_ragged_shape2(
        row_splits=row_splits, cached_tot_size=row_splits[-1].item()
    )
    return ans


def modified_beam_search_LODR(
    model: OnnxModel,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LODR_lm: NgramLm,
    LODR_lm_scale: float,
    LM: OnnxLM,
    lm_scale: float = 0.4,
    beam: int = 4,
) -> List[List[int]]:
    """This function implements LODR (https://arxiv.org/abs/2203.16776) with
    `modified_beam_search`. It uses a bi-gram language model as the estimate
    of the internal language model and subtracts its score during shallow fusion
    with an external language model. This implementation uses a RNNLM as the
    external language model.

    Args:
        model (Transducer):
            The transducer model
        encoder_out (torch.Tensor):
            Encoder output in (N,T,C)
        encoder_out_lens (torch.Tensor):
            A 1-D tensor of shape (N,), containing the number of
            valid frames in encoder_out before padding.
        LODR_lm:
            A low order n-gram LM, whose score will be subtracted during shallow fusion
        LODR_lm_scale:
            The scale of the LODR_lm
        LM:
            A neural net LM, e.g an RNNLM or transformer LM
        beam (int, optional):
            Beam size. Defaults to 4.

    Returns:
      Return a List-of-List of token IDs. ans[i] is the decoding results
      for the i-th utterance.

    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)
    assert LM is not None

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    
    blank_id = model.blank_id
    sos_id = LM.sos_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    # get initial lm score and lm state by scoring the "sos" token
    sos_token = torch.tensor([[sos_id]]).to(torch.int64).to("cpu")
    # lens = torch.tensor([1]).to("cpu")
    init_score, init_states = LM(sos_token)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[blank_id] * context_size,
                log_prob=torch.zeros(1, dtype=torch.float32, device="cpu"),
                state=init_states,  # state of the NN LM
                lm_score=init_score.reshape(-1),
                state_cost=NgramLmStateCost(
                    LODR_lm
                ),  # state of the source domain ngram
            )
        )
    # print("PACKED", packed_encoder_out.data.shape, '\n', packed_encoder_out.data, '\n')
    # encoder_out = model.encoder_proj(packed_encoder_out.data)
    # print("AFTER ENCODER PROJ")
    # print(packed_encoder_out.data.shape)
    # print(packed_encoder_out.data)
    offset = 0
    finalized_B = []
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = packed_encoder_out.data[start:end]  # get batch
        current_encoder_out = current_encoder_out
        # print("\nENCODER DATA CHANGE")
        # print(current_encoder_out.shape)
        # print(current_encoder_out)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to("cpu")

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device="cpu",
            dtype=torch.int64,
        )  # (num_hyps, context_size)
        # print("\nDECODER INPUT")
        # print(decoder_input)
        decoder_out = model.run_decoder(decoder_input)
        # print("\nDECODER OUT")
        # print(decoder_out.shape)
        # print(decoder_out)
        # decoder_out = model.decoder_proj(decoder_out)
        # print("\nAFTER DECODER PROJ")
        # print(decoder_out.shape)
        # print(decoder_out)
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )
        # print("\nLAST ENCODER CHANGE")
        # print(current_encoder_out.shape)
        # print(current_encoder_out)
        logits = model.run_joiner(
            current_encoder_out,
            decoder_out,
        )  # (num_hyps, 1, 1, vocab_size)
        
        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        # print("\nPROBS")
        # print(log_probs.shape)
        # print(log_probs)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)
        """
        for all hyps with a non-blank new token, score this token.
        It is a little confusing here because this for-loop
        looks very similar to the one below. Here, we go through all
        top-k tokens and only add the non-blanks ones to the token_list.
        LM will score those tokens given the LM states. Note that
        the variable `scores` is the LM score after seeing the new
        non-blank token.
        """
        token_list = []
        hs = []
        cs = []
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()
            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_token = topk_token_indexes[k]
                if new_token not in (blank_id, unk_id):
                    # if LM.lm_type == "rnn":
                    token_list.append([new_token])
                    # store the LSTM states
                    hs.append(hyp.state[0])
                    cs.append(hyp.state[1])
                    # else:
                    #     # for transformer LM
                    #     token_list.append(
                    #         [sos_id] + hyp.ys[context_size:] + [new_token]
                    #     )

        # forward NN LM to get new states and scores
        if len(token_list) != 0:
            x_lens = torch.tensor([len(tokens) for tokens in token_list]).to("cpu")
            # if LM.lm_type == "rnn":
            tokens_to_score = (
                torch.tensor(token_list).to(torch.int64).to().reshape(-1, 1)
            )
            hs = torch.cat(hs, dim=1).to()
            cs = torch.cat(cs, dim=1).to()
            state = (hs, cs)
            # else:
            #     # for transformer LM
            #     tokens_List = [torch.tensor(tokens) for tokens in token_list]
            #     tokens_to_score = (
            #         torch.nn.utils.rnn.pad_sequence(
            #             tokens_List, batch_first=True, padding_value=0.0
            #         )
            #         .to()
            #         .to(torch.int64)
            #     )

            #     state = None

            scores, lm_states = LM(tokens_to_score, state)

        count = 0  # index, used to locate score and lm states
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                ys = hyp.ys[:]

                # current score of hyp
                lm_score = hyp.lm_score
                state = hyp.state

                hyp_log_prob = topk_log_probs[k]  # get score of current hyp
                new_token = topk_token_indexes[k]
                if new_token not in (blank_id, unk_id):

                    ys.append(new_token)
                    state_cost = hyp.state_cost.forward_one_step(new_token)

                    # calculate the score of the latest token
                    current_ngram_score = state_cost.lm_score - hyp.state_cost.lm_score

                    assert current_ngram_score <= 0.0, (
                        state_cost.lm_score,
                        hyp.state_cost.lm_score,
                    )
                    # score = score + TDLM_score - LODR_score
                    # LODR_LM_scale should be a negative number here
                    hyp_log_prob += (
                        lm_score[new_token] * lm_scale
                        + LODR_lm_scale * current_ngram_score
                    )  # add the lm score

                    lm_score = scores[count]
                    # if LM.lm_type == "rnn":
                    state = (
                        lm_states[0][:, count, :].unsqueeze(1),
                        lm_states[1][:, count, :].unsqueeze(1),
                    )
                    count += 1
                else:
                    state_cost = hyp.state_cost

                new_hyp = Hypothesis(
                    ys=ys,
                    log_prob=hyp_log_prob,
                    state=state,
                    lm_score=lm_score,
                    state_cost=state_cost,
                )
                B[i].add(new_hyp)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
    return ans


class Onnx_asr_model:
    def __init__(self, weights_path: str, beam_depth: int = 12):
        self.weights_path = weights_path
        torch.set_num_threads(8)

        self.model = OnnxModel(
            encoder_model_filename=os.path.join(weights_path, "am-onnx", "encoder.onnx"),
            decoder_model_filename=os.path.join(weights_path, "am-onnx", "decoder.onnx"),
            joiner_model_filename=os.path.join(weights_path, "am-onnx", "joiner.onnx"),
        )

        sp = spm.SentencePieceProcessor()
        sp.load(os.path.join(weights_path, "lang", "bpe.model"))
        self.sp = sp

        opts = kaldifeat.FbankOptions()
        opts.device = "cpu"
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80

        self.fbank = kaldifeat.Fbank(opts)

        self.LM = OnnxLM(os.path.join(weights_path, "lm", "lm.onnx"),)

        self.ngram_lm = NgramLm(
            os.path.join(weights_path, "lm", "2gram.fst.txt"),
            backoff_id=500,
            is_binary=False,
        )

        self.beam_depth=beam_depth

    @torch.no_grad()
    def transcribe(self, audio):

        features = self.fbank(audio)
        feature_lengths = [f.size(0) for f in features]

        features = pad_sequence(
            features,
            batch_first=True,
            padding_value=math.log(1e-10),
        )

        feature_lengths = torch.tensor(feature_lengths, dtype=torch.int64)
        encoder_out, encoder_out_lens = self.model.run_encoder(features, feature_lengths)

        hyps = modified_beam_search_LODR(
            model=self.model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=self.beam_depth,
            LODR_lm=self.ngram_lm,
            LODR_lm_scale=0,
            LM=self.LM,
            lm_scale=0,
        )

        words = self.sp.decode(hyps)
        return words
