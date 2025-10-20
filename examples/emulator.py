"""
Simple PUF emulator with hardcoded challenge-response pairs (CRPs).

This module emulates a hardware PUF provisioning service. It exposes utilities to:
- derive a challenge from an Owner ID (OID)
- look up a response R for a given Device ID (DID) and challenge
- derive a master key K = H(R || DID)

Security note: This is for development only. Do NOT ship hardcoded CRPs.
"""

import hmac
import hashlib
from typing import Dict, Tuple, Optional


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


class PUFEmulator:
    """
    Emulates a PUF provisioning backend with a fixed CRP table.

    - The table maps (DID, challenge) -> response bytes
    - challenge is derived as SHA256(OID)
    - master key K is SHA256(response || DID)
    """

    def __init__(self, crp_table: Optional[Dict[Tuple[str, bytes], bytes]] = None):
        if crp_table is None:
            # Default dev CRPs for a couple of devices. Values are deterministic
            # but not secret. Extend as needed for testing.
            self._crps: Dict[Tuple[str, bytes], bytes] = {}
            self._seed_crps()
        else:
            self._crps = crp_table

    @staticmethod
    def derive_challenge(oid: str) -> bytes:
        return sha256(oid.encode("utf-8"))

    def get_response(self, did: str, challenge: bytes) -> Optional[bytes]:
        return self._crps.get((did, challenge))

    def get_master_key(self, oid: str, did: str) -> Optional[bytes]:
        challenge = self.derive_challenge(oid)
        response = self.get_response(did, challenge)
        if response is None:
            return None
        return sha256(response + did.encode("utf-8"))

    def _seed_crps(self):
        # Deterministically populate a few CRPs for demo devices
        demo_devices = [
            "DID-0001-DEMO", "DID-0002-DEMO", "DID-EDGE-A1"
        ]
        owners = [
            "OID-ALPHA", "OID-BETA", "OID-GAMMA"
        ]
        for did in demo_devices:
            for oid in owners:
                challenge = self.derive_challenge(oid)
                # Derive a pseudo response using HMAC to keep deterministic
                h = hmac.new(did.encode("utf-8"), digestmod=hashlib.sha256)
                h.update(challenge)
                response = h.digest()
                self._crps[(did, challenge)] = response
