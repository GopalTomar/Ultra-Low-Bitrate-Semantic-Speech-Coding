"""
Compression Engine - Further compresses text tokens
Uses entropy coding and tokenization for ultra-low bitrate
"""

import logging
import json
import zlib
import base64
from typing import Dict, Tuple, Optional, List  # <--- FIXED IMPORTS HERE
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCompressor:
    """Compresses text using various algorithms"""
    
    def __init__(self):
        self.compression_stats = []
    
    def compress(self, text: str, method: str = "zlib") -> bytes:
        """
        Compress text using specified method
        """
        if not text:
            return b''
        
        original_size = len(text.encode('utf-8'))
        
        if method == "zlib":
            compressed = zlib.compress(text.encode('utf-8'), level=9)
        elif method == "bz2":
            import bz2
            compressed = bz2.compress(text.encode('utf-8'))
        elif method == "lzma":
            import lzma
            compressed = lzma.compress(text.encode('utf-8'))
        else:
            raise ValueError(f"Unknown compression method: {method}")
        
        compressed_size = len(compressed)
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        logger.info(f"Compressed: {original_size} → {compressed_size} bytes ({ratio:.2f}:1)")
        
        self.compression_stats.append({
            'method': method,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': ratio
        })
        
        return compressed
    
    def decompress(self, compressed: bytes, method: str = "zlib") -> str:
        """
        Decompress bytes back to text
        """
        if not compressed:
            return ''
        
        if method == "zlib":
            decompressed = zlib.decompress(compressed)
        elif method == "bz2":
            import bz2
            decompressed = bz2.decompress(compressed)
        elif method == "lzma":
            import lzma
            decompressed = lzma.decompress(compressed)
        else:
            raise ValueError(f"Unknown compression method: {method}")
        
        return decompressed.decode('utf-8')
    
    def get_stats(self) -> Dict:
        """Get compression statistics"""
        if not self.compression_stats:
            return {}
        
        total_original = sum(s['original_size'] for s in self.compression_stats)
        total_compressed = sum(s['compressed_size'] for s in self.compression_stats)
        
        return {
            'total_original_bytes': total_original,
            'total_compressed_bytes': total_compressed,
            'average_ratio': total_original / total_compressed if total_compressed > 0 else 0,
            'total_savings_percent': ((total_original - total_compressed) / total_original * 100) if total_original > 0 else 0
        }


class TokenCompressor:
    """
    Advanced compression using tokenization
    Reduces vocabulary and applies entropy coding
    """
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
    
    def build_vocabulary(self, text_corpus: list):
        """
        Build vocabulary from corpus
        """
        word_freq = {}
        
        for text in text_corpus:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Assign indices (most frequent = smaller index)
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        logger.info(f"Built vocabulary: {self.vocab_size} unique words")
    
    def tokenize(self, text: str) -> list:
        """
        Convert text to token indices
        """
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Unknown token (use special index -1 or vocab_size)
                tokens.append(self.vocab_size)
        
        return tokens
    
    def detokenize(self, tokens: list) -> str:
        """
        Convert token indices back to text
        """
        words = []
        
        for token in tokens:
            if token in self.reverse_vocab:
                words.append(self.reverse_vocab[token])
            else:
                words.append("<UNK>")  # Unknown token
        
        return " ".join(words)
    
    def compress_tokens(self, tokens: list) -> bytes:
        """
        Compress token list using variable-length encoding
        """
        if not tokens:
            return b''
            
        if max(tokens) < 256:
            # Each token fits in 1 byte
            return bytes(tokens)
        elif max(tokens) < 65536:
            # Each token needs 2 bytes
            import struct
            return b''.join(struct.pack('H', t) for t in tokens)
        else:
            # Need 4 bytes per token
            import struct
            return b''.join(struct.pack('I', t) for t in tokens)
    
    def decompress_tokens(self, compressed: bytes, bytes_per_token: int = 1) -> list:
        """
        Decompress bytes back to token list
        """
        if not compressed:
            return []
            
        if bytes_per_token == 1:
            return list(compressed)
        elif bytes_per_token == 2:
            import struct
            return [struct.unpack('H', compressed[i:i+2])[0] 
                    for i in range(0, len(compressed), 2)]
        else:
            import struct
            return [struct.unpack('I', compressed[i:i+4])[0] 
                    for i in range(0, len(compressed), 4)]


class SemanticPacket:
    """
    Packet structure for transmitting semantic data
    Includes metadata and compressed payload
    """
    
    def __init__(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        language: str = "en",
        timestamp: Optional[float] = None
    ):
        import time
        
        self.text = text
        self.speaker_id = speaker_id
        self.language = language
        self.timestamp = timestamp or time.time()
        self.compressed_data = None
        self.metadata = {}
    
    def pack(self, compressor: TextCompressor) -> bytes:
        """
        Pack packet into bytes for transmission
        """
        # Compress text
        self.compressed_data = compressor.compress(self.text)
        
        # Create packet structure
        packet = {
            'text_compressed': base64.b64encode(self.compressed_data).decode('utf-8'),
            'speaker_id': self.speaker_id,
            'language': self.language,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'original_length': len(self.text)
        }
        
        # Serialize to JSON
        json_str = json.dumps(packet)
        return json_str.encode('utf-8')
    
    @staticmethod
    def unpack(packet_bytes: bytes, compressor: TextCompressor) -> 'SemanticPacket':
        """
        Unpack bytes back to SemanticPacket
        """
        # Deserialize JSON
        packet_dict = json.loads(packet_bytes.decode('utf-8'))
        
        # Decompress text
        compressed_data = base64.b64decode(packet_dict['text_compressed'])
        text = compressor.decompress(compressed_data)
        
        # Reconstruct packet
        packet = SemanticPacket(
            text=text,
            speaker_id=packet_dict.get('speaker_id'),
            language=packet_dict.get('language', 'en'),
            timestamp=packet_dict.get('timestamp')
        )
        packet.metadata = packet_dict.get('metadata', {})
        
        return packet
    
    def calculate_overhead(self) -> Dict:
        """Calculate packet overhead"""
        text_size = len(self.text.encode('utf-8'))
        compressed_size = len(self.compressed_data) if self.compressed_data else 0
        
        # Full packet size
        full_packet = self.pack(TextCompressor())
        packet_size = len(full_packet)
        
        overhead = packet_size - compressed_size
        
        return {
            'text_size': text_size,
            'compressed_size': compressed_size,
            'packet_size': packet_size,
            'overhead_bytes': overhead,
            'overhead_percent': (overhead / packet_size * 100) if packet_size > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    # Test text compression
    test_text = "The quick brown fox jumps over the lazy dog. " * 10
    
    print("=== Text Compression ===")
    compressor = TextCompressor()
    
    compressed_zlib = compressor.compress(test_text, method="zlib")
    decompressed = compressor.decompress(compressed_zlib, method="zlib")
    
    assert test_text == decompressed, "Decompression failed!"
    print("✓ Compression/Decompression successful")
    
    stats = compressor.get_stats()
    print(f"Average Ratio: {stats['average_ratio']:.2f}:1")
    print(f"Savings: {stats['total_savings_percent']:.1f}%")
    
    # Test semantic packet
    print("\n=== Semantic Packet ===")
    packet = SemanticPacket(
        text="Hello, this is a test message.",
        speaker_id="user_123",
        language="en"
    )
    
    packed = packet.pack(compressor)
    unpacked = SemanticPacket.unpack(packed, compressor)
    
    print(f"Original text: '{packet.text}'")
    print(f"Unpacked text: '{unpacked.text}'")
    print(f"Packet size: {len(packed)} bytes")
    
    overhead = packet.calculate_overhead()
    print(f"Overhead: {overhead['overhead_percent']:.1f}%")