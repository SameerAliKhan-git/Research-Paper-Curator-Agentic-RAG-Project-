import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("0.0.0.0", 8502))
    print("Port 8502 is FREE!")
    s.close()
except Exception as e:
    print("Port 8502 is BUSY:", e)
