import socket
import socks
import requests

socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
socket.socket = socks.socksocket
print(requests.get('https://www.google.com').text)
