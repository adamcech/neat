import socket


class MySocketFunctions:

    @staticmethod
    def send(s: socket.socket, data: bytes, header_size: int = 16):
        s.sendall(bytes(f"{len(data):<{header_size}}", 'utf-8') + data)

    @staticmethod
    def recv(s: socket.socket, header_size: int = 16) -> bytes:
        full_msg = b""

        msg = s.recv(2048)
        data_size = int(msg[:header_size])

        full_msg += msg

        while len(full_msg) < header_size + data_size:
            full_msg += s.recv(2048)

        return full_msg[header_size:]
