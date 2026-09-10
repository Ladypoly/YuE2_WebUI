"""Open the console in a browser once the server answers on its port."""
import socket
import sys
import time
import webbrowser

port = int(sys.argv[1]) if len(sys.argv) > 1 else 7865
deadline = time.monotonic() + 90

while time.monotonic() < deadline:
    probe = socket.socket()
    probe.settimeout(0.5)
    reachable = probe.connect_ex(("127.0.0.1", port)) == 0
    probe.close()
    if reachable:
        webbrowser.open("http://127.0.0.1:%d" % port)
        break
    time.sleep(0.5)
else:
    print("The console did not start within 90 seconds.", file=sys.stderr)
