import itertools as it
from typing import List

import numpy as np
import torch
from fairseq.data import Dictionary

try:
    from flashlight.lib.sequence.criterion import (CpuViterbiPath,
                                                   get_data_ptr_as_bytes)
    from flashlight.lib.text.decoder import (LM, CriterionType, KenLM,
                                             LexiconDecoder,
                                             LexiconDecoderOptions, LMState,
                                             SmearingMode, Trie)
    from flashlight.lib.text.dictionary import create_word_dict, load_words
except:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object


class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        # print(args)
        self.nbest = args["nbest"]

        # criterion-specific init
        if args["criterion"] == "ctc":
            self.criterion_type = CriterionType.CTC
            self.blank = (
                tgt_dict.index("<ctc_blank>")
                if "<ctc_blank>" in tgt_dict.indices
                else tgt_dict.bos()
            )
            if "<sep>" in tgt_dict.indices:
                self.silence = tgt_dict.index("<sep>")
            elif "|" in tgt_dict.indices:
                self.silence = tgt_dict.index("|")
            else:
                self.silence = tgt_dict.eos()
            self.asg_transitions = None
        elif args.criterion == "asg_loss":
            self.criterion_type = CriterionType.ASG
            self.blank = -1
            self.silence = -1
            self.asg_transitions = args.asg_transitions
            self.max_replabel = args.max_replabel
            assert len(self.asg_transitions) == self.vocab_size ** 2
        else:
            raise RuntimeError(f"unknown criterion: {args.criterion}")

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        # return emissions
        return self.decode(emissions)

    def run_decoder(self, emissions):
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        model = models  ## change here
        encoder_out = model(**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(
                    encoder_out
                )  # no need to normalize emissions
            else:
                emissions = model.module.get_normalized_probs(
                    encoder_out, log_probs=True
                )  ## to run on multiple GPU
        elif self.criterion_type == CriterionType.ASG:
            emissions = encoder_out["encoder_out"]
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        if self.criterion_type == CriterionType.CTC:
            idxs = filter(lambda x: x != self.blank, idxs)
        elif self.criterion_type == CriterionType.ASG:
            idxs = filter(lambda x: x >= 0, idxs)
            idxs = unpack_replabels(list(idxs), self.tgt_dict, self.max_replabel)
        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
            for b in range(B)
        ]


class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.unit_lm = getattr(args, "unit_lm", False)

        if args["lexicon"]:
            self.lexicon = load_words(args["lexicon"])
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")
            self.lm = KenLM(args["kenlm_model"], self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)
            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]
                    assert (
                        tgt_dict.unk() not in spelling_idxs
                    ), f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args["beam"],
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args["beam_threshold"],
                lm_weight=args["lm_weight"],
                word_score=args["word_score"],
                unk_score=args["unk_weight"],
                sil_score=args["sil_weight"],
                log_add=False,
                criterion_type=self.criterion_type,
            )

            if self.asg_transitions is None:
                pass
                # self.asg_transitions = torch.FloatTensor(N, N).zero_()
                self.asg_transitions = []

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                self.asg_transitions,
                self.unit_lm,
            )
        else:
            assert (
                args.unit_lm
            ), "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import (LexiconFreeDecoder,
                                                     LexiconFreeDecoderOptions)

            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

    def get_timesteps(self, token_idxs: List[int]) -> List[int]:
        """Returns frame numbers corresponding to every non-blank token.
        Parameters
        ----------
        token_idxs : List[int]
            IDs of decoded tokens.
        Returns
        -------
        List[int]
            Frame numbers corresponding to every non-blank token.
        """
        timesteps = []
        for i, token_idx in enumerate(token_idxs):
            if token_idx == self.blank:
                continue
            if i == 0 or token_idx != token_idxs[i - 1]:
                timesteps.append((i, token_idx))
        return timesteps

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(result.tokens),
                        "score": result.score,
                        "timesteps": self.get_timesteps(result.tokens),
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    }
                    for result in nbest_results
                ]
            )
        return hypos


def get_args(lexicon_path, lm_path, BEAM=128, LM_WEIGHT=2, WORD_SCORE=-1):
    args = {}
    args["lexicon"] = lexicon_path
    args["kenlm_model"] = lm_path
    args["beam"] = BEAM
    args["beam_threshold"] = 25
    args["lm_weight"] = LM_WEIGHT
    args["word_score"] = WORD_SCORE
    args["unk_weight"] = -np.inf
    args["sil_weight"] = 0
    args["nbest"] = 1
    args["criterion"] = "ctc"
    args["labels"] = "ltr"
    return args


# HARDCODING ARGS HERE #
def load_decoder(dict_path, lexicon_path, lm_path, decoder):
    target_dict = Dictionary.load(dict_path)
    args = get_args(lexicon_path, lm_path)

    if decoder == "viterbi":
        generator = W2lViterbiDecoder(args, target_dict)
    else:
        generator = W2lKenLMDecoder(args, target_dict)

    return generator
