
"""
Prediction script for the Transformer model.

This script loads a trained Transformer model from a checkpoint and generates 
predictions using greedy decoding.
"""

import sys
import torch
import argparse
import sentencepiece as spm
from pathlib import Path

from .model import Transformer
from .data import SPECIALS


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, verbose=False):
    """
    Load the trained model and SentencePiece tokenizer.
    
    Args:
        checkpoint_path (str): Path to the .pt checkpoint file
        tokenizer_path (str): Path to the .model SentencePiece file
    
    Returns:
        tuple: (model, tokenizer, device)
            - model: Loaded Transformer model in eval mode
            - tokenizer: SentencePiece processor
            - device: Device the model is loaded on
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    if verbose:
        print(f"Loading checkpoint from {checkpoint_path}...", file=sys.stderr)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model hyperparameters from checkpoint
    vocab_size = checkpoint["vocab_size"]
    hparams = checkpoint["hparams"]
    
    if verbose:
        print(f"Model hyperparameters: {hparams}", file=sys.stderr)
        print(f"Vocabulary size: {vocab_size}", file=sys.stderr)
    
    # Create model with saved hyperparameters
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=hparams["d_model"],
        num_layers_enc=hparams["num_layers_enc"],
        num_layers_dec=hparams["num_layers_dec"],
        num_heads=hparams["num_heads"],
        d_ff=hparams["d_ff"],
        dropout=hparams["dropout"]
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    if verbose:
        print(f"Model loaded on device: {device}", file=sys.stderr)
    
    # Load tokenizer
    if verbose:
        print(f"Loading tokenizer from {tokenizer_path}...", file=sys.stderr)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if verbose:
        print(f"Tokenizer vocabulary size: {tokenizer.vocab_size()}", file=sys.stderr)
    
    return model, tokenizer, device


def encode_text(text, tokenizer, max_len=128):
    """
    Tokenize and encode input text for the model.
    
    Args:
        text (str): Input text string
        tokenizer: SentencePiece processor
        max_len (int): Maximum sequence length
    
    Returns:
        torch.Tensor: Encoded input tokens with shape [1, seq_len]
    """
    # Encode text to token IDs
    token_ids = tokenizer.encode(text.strip(), out_type=int)
    
    # Add BOS and EOS tokens
    token_ids = [SPECIALS["bos"]] + token_ids + [SPECIALS["eos"]]
    
    # Truncate if sequence is too long
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len-1] + [SPECIALS["eos"]]
    
    # Convert to tensor and add batch dimension
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def decode_tokens(token_ids, tokenizer):
    """
    Convert token IDs back to readable text.
    
    Args:
        token_ids (list): List of token IDs
        tokenizer: SentencePiece processor
    
    Returns:
        str: Decoded text string
    """
    if not token_ids:
        return ""
    
    # Filter out special tokens and invalid tokens
    filtered_tokens = [
        token_id for token_id in token_ids 
        if (token_id not in [SPECIALS["pad"], SPECIALS["bos"], SPECIALS["eos"], SPECIALS["unk"]] 
            and 0 <= token_id < tokenizer.vocab_size())
    ]
    
    if not filtered_tokens:
        return ""
    
    try:
        decoded_text = tokenizer.decode(filtered_tokens)
        return decoded_text.strip()
    except Exception as e:
        print(f"Warning: Error decoding tokens {filtered_tokens}: {e}", file=sys.stderr)
        return ""


def generate_prediction(model, src_tokens, tokenizer, device, max_len=128, verbose=False):
    """
    Generate translation using greedy decoding.
    
    Args:
        model: Transformer model
        src_tokens: Source tokens tensor [1, src_len]
        tokenizer: SentencePiece processor
        device: Device to run on
        max_len (int): Maximum generation length
        verbose (bool): Print debug information
    
    Returns:
        list: Generated token IDs (excluding special tokens used internally)
    """
    model.eval()
    src_tokens = src_tokens.to(device)
    pad_id = SPECIALS["pad"]
    
    if verbose:
        print(f"Input tokens: {src_tokens[0].cpu().tolist()}", file=sys.stderr)
        src_text = decode_tokens(src_tokens[0].cpu().tolist(), tokenizer)
        print(f"Input text reconstruction: '{src_text}'", file=sys.stderr)
    
    with torch.no_grad():
        # Encode source sequence
        memory, _ = model.encode(src_tokens, pad_id)
        
        if verbose:
            print(f"Encoded memory shape: {memory.shape}", file=sys.stderr)
        
        # Initialize target sequence with BOS token
        tgt_tokens = torch.tensor([[SPECIALS["bos"]]], dtype=torch.long, device=device)
        generated_ids = []
        
        # Generate tokens one by one
        for step in range(max_len - 1):
            # Get decoder output for current target sequence
            decoder_output = model.decode(tgt_tokens, memory, src_tokens, pad_id, pad_id)
            
            # Generate next token logits
            logits = model.generator(decoder_output)  # [1, tgt_len, vocab_size]
            next_token_logits = logits[:, -1, :]  # Get logits for next position
            
            # Greedy selection: choose token with highest probability
            next_token_id = next_token_logits.argmax(dim=-1).item()
            
            if verbose:
                # Show top candidates for debugging
                probs = torch.softmax(next_token_logits, dim=-1)
                top_probs, top_indices = probs.topk(5)
                candidates = []
                for idx, prob in zip(top_indices[0], top_probs[0]):
                    token_text = tokenizer.decode([idx.item()]) if idx.item() < tokenizer.vocab_size() else f"<UNK:{idx.item()}>"
                    candidates.append(f"{token_text}({prob:.3f})")
                print(f"Step {step+1}: Selected token ID {next_token_id} | Top-5: {candidates}", file=sys.stderr)
            
            # Stop generation if EOS token is produced
            if next_token_id == SPECIALS["eos"]:
                if verbose:
                    print(f"EOS token generated, stopping at step {step+1}", file=sys.stderr)
                break
            
            # Add token to generated sequence
            generated_ids.append(next_token_id)
            
            # Update target sequence for next iteration
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            tgt_tokens = torch.cat([tgt_tokens, next_token_tensor], dim=1)
            
            if verbose:
                current_text = decode_tokens(generated_ids, tokenizer)
                print(f"  Current output: '{current_text}'", file=sys.stderr)
        
        if verbose:
            print(f"Generation finished. Total tokens: {len(generated_ids)}", file=sys.stderr)
    
    return generated_ids


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate predictions using trained Transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello, how are you?"
  echo "Good morning" | %(prog)s
  %(prog)s "Translate this" --verbose
        """
    )
    
    parser.add_argument(
        "text", 
        nargs="?", 
        help="Input text to translate (if not provided, reads from stdin)"
    )
    parser.add_argument(
        "--checkpoint", 
        default="checkpoints/transformer_best_full.pt",
        help="Path to model checkpoint (default: checkpoints/transformer_best_full.pt)"
    )
    parser.add_argument(
        "--tokenizer", 
        default="spm_shared.model",  # Use existing tokenizer for now
        help="Path to SentencePiece model (default: spm_shared.model)"
    )
    parser.add_argument(
        "--max-len", 
        type=int, 
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Print debug information during generation"
    )
    parser.add_argument(
        "--only-output",
        action="store_true",
        help="Print only the final translation line with no extra logs"
    )
    
    args = parser.parse_args()
    
    # Get input text
    if args.text:
        input_text = args.text
    else:
        # Read from stdin
        input_text = sys.stdin.read().strip()
    
    if not input_text:
        print("Error: No input text provided", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Resolve file paths
    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / args.checkpoint
    tokenizer_path = project_root / args.tokenizer
    
    # Verify files exist
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer file not found at {tokenizer_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Determine verbosity (suppress logs if only-output requested)
        verbose = args.verbose and not args.only_output

        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer(checkpoint_path, tokenizer_path, verbose=verbose)
        
        # Encode input text
        src_tokens = encode_text(input_text, tokenizer, args.max_len)
        
        if verbose:
            print(f"Processing input: '{input_text}'", file=sys.stderr)
            print(f"Encoded to {src_tokens.shape[1]} tokens", file=sys.stderr)
        
        # Generate prediction
        if verbose:
            print("Generating prediction...", file=sys.stderr)
        predicted_tokens = generate_prediction(
            model, src_tokens, tokenizer, device, args.max_len, verbose
        )
        
        # Decode prediction to text
        output_text = decode_tokens(predicted_tokens, tokenizer)
        
        if verbose:
            print(f"Generated {len(predicted_tokens)} tokens", file=sys.stderr)
            print(f"Final translation: '{output_text}'", file=sys.stderr)
        
        # Output result
        if output_text:
            print(f"final translation: {output_text}")
        else:
            print("Warning: Generated empty output", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()