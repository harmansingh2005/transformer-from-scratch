import torch
from torchtext.data.metrics import bleu_score

def greedy_decode(
	model,
	src,
	src_mask,
	max_len,
	start_symbol,
	device,
	eos_symbol=None,
	pad_id_src=0,
	pad_id_tgt=0,
):
	"""
	Greedy decoding for sequence generation.
	Supports both the repository's Transformer API (pad_id based) and
	a fallback mask-based API used in tests' DummyModel.

	Args:
		model: The model providing encode/decode/generator.
		src: Source tensor (batch_size, src_seq_len).
		src_mask: Source mask (kept for backward compatibility; unused for pad-id API).
		max_len: Maximum length to generate (including the start token position).
		start_symbol: BOS token id to start decoding.
		device: torch.device.
		eos_symbol: Optional EOS token id for early stopping (default: None).
		pad_id_src: Pad id for source (default: 0).
		pad_id_tgt: Pad id for target (default: 0).

	Returns:
		Tensor of shape (batch_size, L) where L <= max_len if early-stopped,
		else L == max_len.
	"""
	batch_size = src.size(0)
	ys = torch.full((batch_size, 1), start_symbol, dtype=src.dtype, device=device)

	# Encode using pad-id API if available (returns tuple), else use mask-based API
	use_pad_api = False
	enc_res = None
	try:
		enc_res = model.encode(src, pad_id_src)
	except TypeError:
		enc_res = None

	if isinstance(enc_res, tuple):
		memory = enc_res[0]
		use_pad_api = True
	elif enc_res is not None and not isinstance(enc_res, tuple):
		# Likely DummyModel returned a tensor; treat as mask-based API
		memory = model.encode(src, src_mask)
		use_pad_api = False
	else:
		# Directly fallback to mask-based API
		memory = model.encode(src, src_mask)
		use_pad_api = False

	for _ in range(max_len - 1):
		if use_pad_api:
			dec_out = model.decode(ys, memory, src, pad_id_src, pad_id_tgt)
			logits_step = model.generator(dec_out)[:, -1, :]
		else:
			# Fallback: tests' DummyModel returns logits directly from decode
			tgt_mask = torch.ones(ys.size(0), ys.size(1), device=device)
			out = model.decode(memory, src_mask, ys, tgt_mask)
			logits_step = model.generator(out[:, -1])

		next_token = torch.argmax(logits_step, dim=1, keepdim=True)
		ys = torch.cat([ys, next_token], dim=1)

		if eos_symbol is not None:
			# Stop if all sequences in batch have produced EOS
			if torch.all(next_token.squeeze(1) == eos_symbol):
				break

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
