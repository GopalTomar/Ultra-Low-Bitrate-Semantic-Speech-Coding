"""
Network Layer - WebSocket-based client/server for transmitting semantic packets
Handles real-time bidirectional communication
"""

import asyncio
import websockets
import logging
import json
from typing import Optional, Callable
from pathlib import Path
import ssl

from config import Config
from compression_engine import SemanticPacket, TextCompressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticServer:
    """
    Server that receives audio, transcribes it, and sends text packets
    """

    def __init__(
        self,
        host: str = Config.SERVER_HOST,
        port: int = Config.SERVER_PORT,
        on_receive_callback: Optional[Callable] = None
    ):
        self.host = host
        self.port = port
        self.on_receive_callback = on_receive_callback
        self.compressor = TextCompressor()
        self.active_connections = set()

    # --- FIX IS HERE: Removed 'path' argument ---
    async def handler(self, websocket):
        """Handle incoming WebSocket connections"""
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")
        self.active_connections.add(websocket)

        try:
            async for message in websocket:
                # Received semantic packet from client
                packet = SemanticPacket.unpack(message, self.compressor)

                logger.info(f"Received from {client_addr}: '{packet.text}'")

                # Call custom callback if provided
                if self.on_receive_callback:
                    response = self.on_receive_callback(packet)

                    if response:
                        # Send response back
                        response_packet = SemanticPacket(
                            text=response,
                            speaker_id="server",
                            language=packet.language
                        )
                        await websocket.send(response_packet.pack(self.compressor))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            self.active_connections.discard(websocket)

    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting server on {self.host}:{self.port}")

        # Optional: Setup SSL
        ssl_context = None
        if Config.USE_ENCRYPTION:
            # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            # ssl_context.load_cert_chain('cert.pem', 'key.pem')
            pass

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            ssl=ssl_context
        ):
            logger.info(f"✓ Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    def run(self):
        """Run the server (blocking)"""
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")


class SemanticClient:
    """
    Client that sends text packets and receives synthesized responses
    """

    def __init__(
        self,
        server_uri: Optional[str] = None,
        on_receive_callback: Optional[Callable] = None
    ):
        if server_uri is None:
            server_uri = f"ws://{Config.SERVER_HOST}:{Config.SERVER_PORT}"

        self.server_uri = server_uri
        self.on_receive_callback = on_receive_callback
        self.compressor = TextCompressor()
        self.websocket = None
        self.connected = False

    async def connect(self):
        """Connect to the server"""
        try:
            logger.info(f"Connecting to {self.server_uri}...")
            self.websocket = await websockets.connect(
                self.server_uri,
                ping_interval=20,
                ping_timeout=10
            )
            self.connected = True
            logger.info("✓ Connected to server")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            raise

    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from server")

    async def send(self, text: str, speaker_id: Optional[str] = None):
        """
        Send text packet to server

        Args:
            text: Text to send
            speaker_id: Optional speaker identifier
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to server")

        # Create and pack semantic packet
        packet = SemanticPacket(
            text=text,
            speaker_id=speaker_id,
            language=Config.TTS_LANGUAGE
        )

        packed = packet.pack(self.compressor)

        logger.info(f"Sending: '{text}' ({len(packed)} bytes)")
        await self.websocket.send(packed)

    async def receive(self) -> SemanticPacket:
        """
        Receive packet from server

        Returns:
            SemanticPacket object
        """
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to server")

        message = await self.websocket.recv()
        packet = SemanticPacket.unpack(message, self.compressor)

        logger.info(f"Received: '{packet.text}'")

        # Call callback if provided
        if self.on_receive_callback:
            self.on_receive_callback(packet)

        return packet

    async def send_and_receive(
        self,
        text: str,
        speaker_id: Optional[str] = None
    ) -> SemanticPacket:
        """
        Send a packet and wait for response

        Args:
            text: Text to send
            speaker_id: Optional speaker ID

        Returns:
            Response packet
        """
        await self.send(text, speaker_id)
        response = await self.receive()
        return response

    async def interactive_session(self):
        """
        Interactive CLI session for testing
        """
        logger.info("\n=== Interactive Session Started ===")
        logger.info("Type messages to send (or 'quit' to exit)\n")

        try:
            while True:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if user_input.strip():
                    # Send message
                    await self.send(user_input)

                    # Wait for response
                    response = await self.receive()
                    print(f"Server: {response.text}\n")

        except KeyboardInterrupt:
            logger.info("\nSession ended by user")


class NetworkMetrics:
    """Track network performance metrics"""

    def __init__(self):
        self.packets_sent = 0
        self.packets_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.latencies = []

    def record_send(self, packet_size: int):
        """Record a sent packet"""
        self.packets_sent += 1
        self.bytes_sent += packet_size

    def record_receive(self, packet_size: int, latency: float):
        """Record a received packet"""
        self.packets_received += 1
        self.bytes_received += packet_size
        self.latencies.append(latency)

    def get_stats(self) -> dict:
        """Get network statistics"""
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0

        return {
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'average_latency_ms': avg_latency * 1000,
            'min_latency_ms': min(self.latencies) * 1000 if self.latencies else 0,
            'max_latency_ms': max(self.latencies) * 1000 if self.latencies else 0
        }

    def print_stats(self):
        """Print network statistics"""
        stats = self.get_stats()
        print("\n=== Network Statistics ===")
        print(f"Packets Sent: {stats['packets_sent']}")
        print(f"Packets Received: {stats['packets_received']}")
        print(f"Data Sent: {stats['bytes_sent']:,} bytes")
        print(f"Data Received: {stats['bytes_received']:,} bytes")
        print(f"Average Latency: {stats['average_latency_ms']:.2f} ms")
        print(f"Min Latency: {stats['min_latency_ms']:.2f} ms")
        print(f"Max Latency: {stats['max_latency_ms']:.2f} ms")


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run as server
        def handle_message(packet: SemanticPacket) -> str:
            """Echo server - sends back what it receives"""
            return f"Echo: {packet.text}"

        server = SemanticServer(on_receive_callback=handle_message)
        server.run()

    else:
        # Run as client
        async def run_client():
            client = SemanticClient()

            try:
                await client.connect()
                await client.interactive_session()
            finally:
                await client.disconnect()

        asyncio.run(run_client())