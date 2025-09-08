
import torch
from torchtext.data.metrics import bleu_score

def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
	"""
	Greedy decoding for sequence generation.
	Args:
		model: The trained Transformer model.
		src: Source input tensor (batch_size, src_seq_len).
		src_mask: Source mask tensor.
		max_len: Maximum length of the generated sequence.
		start_symbol: The index of the start token.
		device: torch.device.
	Returns:
		output: Generated sequence tensor (batch_size, max_len)
	"""
	memory = model.encode(src, src_mask)
	batch_size = src.size(0)
	ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src).to(device)
	for i in range(max_len-1):
		out = model.decode(memory, src_mask, ys, torch.ones(ys.size(0), ys.size(1)).to(device))
		prob = model.generator(out[:, -1])
		_, next_word = torch.max(prob, dim=1)
		next_word = next_word.unsqueeze(1)
		ys = torch.cat([ys, next_word], dim=1)
	return ys

def compute_bleu(candidate_corpus, reference_corpus, max_n=4):
	"""
	Compute BLEU score for a batch of predictions and references.
	Args:
		candidate_corpus: List of candidate sentences (list of list of tokens)
		reference_corpus: List of reference sentences (list of list of list of tokens)
		max_n: Maximum n-gram order for BLEU.
	Returns:
		BLEU score (float)
	"""
	return bleu_score(candidate_corpus, reference_corpus, max_n=max_n)
