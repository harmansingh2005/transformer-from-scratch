
import torch
import unittest
import sys
sys.path.append('c:/Users/harma/OneDrive/Desktop/Transformer LLM/src')
import eval as eval_module

class DummyModel:
	def encode(self, src, src_mask):
		return src
	def decode(self, memory, src_mask, ys, tgt_mask):
		# Return dummy logits: batch_size x seq_len x vocab_size
		batch_size = ys.size(0)
		seq_len = ys.size(1)
		vocab_size = 5
		# Always predict next token as 1
		logits = torch.zeros(batch_size, seq_len, vocab_size)
		logits[:, -1, 1] = 10.0
		return logits
	def generator(self, out):
		# out: batch_size x vocab_size
		return out

class TestEvalGreedyBLEU(unittest.TestCase):
	def setUp(self):
		self.device = torch.device('cpu')
		self.model = DummyModel()
		self.src = torch.tensor([[1,2,3]])
		self.src_mask = torch.ones(1, 3)
		self.max_len = 5
		self.start_symbol = 0

	def test_greedy_decode(self):
		ys = eval_module.greedy_decode(self.model, self.src, self.src_mask, self.max_len, self.start_symbol, self.device)
		self.assertEqual(ys.shape, (1, self.max_len))
		# All tokens after start should be 1 (see DummyModel)
		self.assertTrue(torch.all(ys[0, 1:] == 1))

	def test_compute_bleu(self):
		candidate = [["the", "cat", "sat"]]
		reference = [[["the", "cat", "sat"]]]
		bleu = eval_module.compute_bleu(candidate, reference)
		self.assertAlmostEqual(bleu, 1.0, places=4)

if __name__ == "__main__":
	unittest.main()
