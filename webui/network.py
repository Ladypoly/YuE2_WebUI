"""Reaching the console from a phone on the same network, without opening it up.

Bound to localhost the console needs no protection: nothing else can reach it.
Bound to the network it is reachable by everything on that network, and its API
can start runs, move files around and download models, so it asks for a PIN.
The PIN is a lock on the front door of a house on a private street, not an
authentication system: this is plain HTTP on a LAN, so it belongs behind a
router, never on the open internet.

The page itself and its stylesheet stay open, so a phone can load the console
far enough to ask for the PIN; everything underneath is gated.
"""
from __future__ import annotations

import ipaddress
import secrets
import socket
import threading
import time

COOKIE = "yue2_key"
HEADER = "x-yue2-key"
# A wrong PIN is cheap to try, so a burst of them costs the caller a pause.
MAX_ATTEMPTS = 8
LOCKOUT_SECONDS = 300


def new_pin():
    return "%06d" % secrets.randbelow(1000000)


def is_loopback(host):
    if not host:
        return False
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return host == "localhost"


def local_addresses():
    """The addresses this machine is likely reachable at, best guess first.

    A machine with Hyper-V, WSL, a VPN or Bluetooth tethering has several
    addresses and only one of them is the network the phone is on. The address
    the routing table would actually use comes first; link-local addresses are
    dropped, because nothing reaches the console on one of those.
    """
    found = []
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.168.255.255", 9))
        found.append(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in found:
                found.append(address)
    except OSError:
        pass
    usable = []
    for address in found:
        if is_loopback(address):
            continue
        try:
            if ipaddress.ip_address(address).is_link_local:
                continue        # 169.254.x: an interface with no real network
        except ValueError:
            continue
        usable.append(address)
    return usable


def subnet_of(address, bits=24):
    """The /24 the phone shares, for a firewall rule that is not wide open."""
    try:
        return str(ipaddress.ip_network(address + "/%d" % bits, strict=False))
    except ValueError:
        return ""


def firewall_command(port, address):
    """The one elevated command Windows needs before a phone can connect."""
    scope = subnet_of(address)
    if not scope:
        return ""
    return ("New-NetFirewallRule -DisplayName 'YuE2 Console' -Direction Inbound "
            "-Action Allow -Protocol TCP -LocalPort %d -RemoteAddress %s" % (port, scope))


class Gate:
    """Who may call the API, and how badly they have guessed so far."""

    def __init__(self):
        self.pin = ""
        self.lock = threading.Lock()
        self.attempts = {}

    def required(self, host):
        """Loopback is the machine itself; anything else needs the PIN."""
        return bool(self.pin) and not is_loopback(host)

    def blocked_for(self, host):
        with self.lock:
            count, until = self.attempts.get(host, (0, 0.0))
        remaining = until - time.monotonic()
        return int(remaining) if remaining > 0 else 0

    def check(self, host, offered):
        if not self.required(host):
            return True
        if self.blocked_for(host):
            return False
        if offered and secrets.compare_digest(str(offered), self.pin):
            with self.lock:
                self.attempts.pop(host, None)
            return True
        return False

    def note_failure(self, host):
        """Only guesses at the PIN count; a request with no key at all does not."""
        with self.lock:
            count, until = self.attempts.get(host, (0, 0.0))
            count += 1
            if count >= MAX_ATTEMPTS:
                self.attempts[host] = (0, time.monotonic() + LOCKOUT_SECONDS)
            else:
                self.attempts[host] = (count, until)

    def describe(self, port, reveal=False):
        addresses = local_addresses()
        return {"pin_set": bool(self.pin),
                "pin": self.pin if reveal else "",
                "addresses": addresses,
                "urls": ["http://%s:%d/" % (address, port) for address in addresses],
                "firewall": firewall_command(port, addresses[0]) if addresses else "",
                "port": port}
